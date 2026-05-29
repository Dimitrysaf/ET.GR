"""Tests for codification (Level 2). No network, no Ollama, no API key.

Δοκιμές για την κωδικοποίηση: regex parser + smoke test του codifier.
"""
import os
import sys
import tempfile

# Make sure ANTHROPIC_API_KEY is unset for the dormant-Claude assertion.
os.environ.pop("ANTHROPIC_API_KEY", None)

from backend.codification.amendment_parser import (
    parse_amendments,
    claude_parse_amendments,
)


def test_parse_replace():
    s = "Η παρ. 3 του άρθρου 46 του ν. 4622/2019 αντικαθίσταται ως εξής: «Το νέο κείμενο εδώ.»"
    res = parse_amendments(s)
    assert len(res) == 1, res
    a = res[0]
    assert a.target_law_id == "ν. 4622/2019", a.target_law_id
    assert a.operation == "replace", a.operation
    assert a.article == "46", a.article
    assert a.paragraph == "3", a.paragraph
    assert a.new_text == "Το νέο κείμενο εδώ.", repr(a.new_text)


def test_parse_delete():
    s = "Η παρ. 2 του άρθρου 5 του ν. 4622/2019 καταργείται."
    res = parse_amendments(s)
    assert len(res) == 1, res
    a = res[0]
    assert a.operation == "delete", a.operation
    assert a.article == "5", a.article
    assert a.paragraph == "2", a.paragraph
    assert a.new_text is None, repr(a.new_text)


def test_parse_insert():
    s = "Στο άρθρο 12 του ν. 4622/2019 προστίθεται παράγραφος 5 ως εξής: «Νέα παράγραφος.»"
    res = parse_amendments(s)
    assert len(res) == 1, res
    a = res[0]
    assert a.operation == "insert", a.operation
    assert a.article == "12", a.article
    assert a.new_text == "Νέα παράγραφος.", repr(a.new_text)


def test_parse_multiple_sentences_with_inner_period():
    """Πολλές οδηγίες σε σειρά, καθεμία τελειώνει με τελεία ΜΕΣΑ στα «...».

    Regression: ο splitter πρέπει να σπάει στο ".»", αλλιώς οι προτάσεις ενώνονται
    και χάνονται/μπερδεύονται οι τροποποιήσεις (λάθος operation/νόμος).
    """
    s = (
        "Η περ. β της παρ. 1 του άρθρου 9 του ν. 4622/2019 διαγράφεται. "
        "Στο τέλος της παρ. 2 του άρθρου 7 του ν. 4622/2019 προστίθεται εδάφιο "
        "ως εξής: «Νέο εδάφιο κειμένου.» "
        "Το άρθρο 90 του π.δ. 63/2005 αντικαθίσταται ως εξής: «Αντικατάσταση.»"
    )
    res = parse_amendments(s)
    assert len(res) == 3, [(a.operation, a.target_law_id, a.article) for a in res]
    assert res[0].operation == "delete" and res[0].article == "9"
    assert res[1].operation == "insert" and res[1].article == "7"
    assert res[1].new_text == "Νέο εδάφιο κειμένου.", repr(res[1].new_text)
    assert res[2].operation == "replace" and res[2].target_law_id == "π.δ. 63/2005"
    assert res[2].new_text == "Αντικατάσταση.", repr(res[2].new_text)


def test_claude_dormant_without_key():
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert claude_parse_amendments("οτιδήποτε") is None


def test_codify_routes_no_old_text_to_review():
    """Replace χωρίς old_text → πρέπει να μπει στην ουρά ελέγχου χωρίς exception."""
    tmp = tempfile.mkdtemp()
    # Redirect LAWS_DIR/REVIEW_DIR ΠΡΙΝ φορτωθεί ο codifier-εξαρτημένος κώδικας.
    import backend.config as cfg
    cfg.LAWS_DIR = os.path.join(tmp, "laws")
    cfg.REVIEW_DIR = os.path.join(tmp, "review")
    import backend.storage.versioning as ver
    import backend.ai.review_queue as rq
    ver.LAWS_DIR = cfg.LAWS_DIR
    rq.REVIEW_DIR = cfg.REVIEW_DIR

    from backend.codification.codifier import codify

    s = "Η παρ. 3 του άρθρου 46 του ν. 4622/2019 αντικαθίσταται ως εξής: «Νέο κείμενο.»"
    results = codify(s, effective_date="2026-05-29")
    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    assert r.law_id == "ν. 4622/2019"
    # replace χωρίς old_text → δεν εφαρμόζεται, μπαίνει σε review.
    assert r.applied == 0, r.applied
    assert r.queued_for_review == 1, r.queued_for_review


def test_codify_insert_applies():
    """Insert εφαρμόζεται (append) και κάνει snapshot — απλό seeding."""
    tmp = tempfile.mkdtemp()
    import backend.config as cfg
    cfg.LAWS_DIR = os.path.join(tmp, "laws2")
    cfg.REVIEW_DIR = os.path.join(tmp, "review2")
    import backend.storage.versioning as ver
    import backend.ai.review_queue as rq
    ver.LAWS_DIR = cfg.LAWS_DIR
    rq.REVIEW_DIR = cfg.REVIEW_DIR

    from backend.codification.codifier import codify

    s = "Στο άρθρο 12 του ν. 4622/2019 προστίθεται παράγραφος 5 ως εξής: «Νέα παράγραφος.»"
    results = codify(s, effective_date="2026-05-29")
    assert len(results) == 1
    r = results[0]
    assert r.applied == 1, (r.applied, r.notes)
    assert r.current_path is not None


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    print(f"\nAll {len(fns)} tests passed")


if __name__ == "__main__":
    _run()
