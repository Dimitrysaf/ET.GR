"""Codifier — εφαρμογή τροποποιήσεων ΦΕΚ πάνω στο κείμενο νόμου.

Applies parsed FEK amendments to versioned law texts. Each amending FEK is a
"commit" with effective date = publication date.

ΕΙΛΙΚΡΙΝΗΣ ΣΧΕΔΙΑΣΗ / Honest design note:
Οι ελληνικές οδηγίες τροποποίησης σχεδόν ΠΟΤΕ δεν παραθέτουν το ΠΑΛΙΟ κείμενο·
δίνουν μόνο locator ("παρ. 3 του άρθρου 46") + το ΝΕΟ κείμενο. Σε flat-text
αποθήκευση δεν μπορούμε να εντοπίσουμε με ασφάλεια "παρ. 3 του άρθρου 46".
Γι' αυτό οι περισσότερες replace/delete χωρίς old_text οδηγούνται στην ουρά
ανθρώπινου ελέγχου (queued_for_review) — αυτό είναι σωστό και ασφαλές. Το
auto-apply καλύπτει: (α) insert (append/anchor) και (β) replace/delete ΟΤΑΝ
παρατίθεται old_text που βρίσκεται στο κείμενο.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Import-fallback pattern (όπως στα υπόλοιπα modules) ─────────────────────────
try:
    from backend.codification.amendment_parser import (
        ParsedAmendment,
        parse_amendments,
        claude_parse_amendments,
    )
    from backend.ai.diff_engine import apply_amendments
    from backend.ai.review_queue import enqueue_review
    from backend.storage.versioning import get_current_law_text, snapshot_and_update
except Exception:  # pragma: no cover - allow alternate package layout
    from codification.amendment_parser import (  # type: ignore
        ParsedAmendment,
        parse_amendments,
        claude_parse_amendments,
    )
    from ai.diff_engine import apply_amendments  # type: ignore
    from ai.review_queue import enqueue_review  # type: ignore
    from storage.versioning import get_current_law_text, snapshot_and_update  # type: ignore


@dataclass
class CodificationResult:
    law_id: str
    applied: int                 # how many amendments auto-applied
    queued_for_review: int       # how many needed human/Claude confirmation
    diff: str
    current_path: Optional[str]
    version_path: Optional[str]
    notes: List[str] = field(default_factory=list)


def _is_auto_applicable(am: ParsedAmendment) -> bool:
    """True αν η τροποποίηση μπορεί να εφαρμοστεί αυτόματα με ασφάλεια.

    - insert: πάντα (το diff_engine κάνει anchor ή append).
    - replace/delete: μόνο αν υπάρχει old_text (αλλιώς δεν εντοπίζεται).
    Επιπλέον απαιτείται confidence >= 0.6.
    """
    if am.confidence < 0.6:
        return False
    if am.operation == "insert":
        return True
    if am.operation in ("replace", "delete"):
        return bool(am.old_text)
    return False


def _to_diff_amendment(am: ParsedAmendment) -> Dict:
    """Μετατροπή ParsedAmendment → dict που δέχεται το diff_engine."""
    return {
        "operation": am.operation,
        "old_text": am.old_text or "",
        "new_text": am.new_text or "",
        "confidence": am.confidence,
    }


def _review_payload(law_id: str, am: ParsedAmendment, reason: str) -> Dict:
    """Δομημένο payload για την ουρά ανθρώπινου ελέγχου."""
    return {
        "document_id": f"codify-{law_id}",
        "type": "codification_amendment",
        "law_id": law_id,
        "operation": am.operation,
        "article": am.article,
        "paragraph": am.paragraph,
        "periptosi": am.periptosi,
        "edafio": am.edafio,
        "new_text": am.new_text,
        "old_text": am.old_text,
        "confidence": am.confidence,
        "raw_instruction": am.raw_instruction,
        "reason": reason,
    }


def codify(raw_text: str, effective_date: Optional[str], log_hook=None,
           use_claude: bool = False) -> List[CodificationResult]:
    """Κωδικοποίηση: εφαρμογή τροποποιήσεων ΦΕΚ στους νόμους-στόχους.

    1. parsed = parse_amendments(raw_text). Αν κενό και use_claude, δοκίμασε
       claude_parse_amendments (dormant όταν δεν υπάρχει ANTHROPIC_API_KEY).
    2. Ομαδοποίηση ανά target_law_id.
    3. Για κάθε νόμο:
         - auto-applicable (insert· ή replace/delete με old_text & confidence>=0.6)
           → diff_engine.apply_amendments → snapshot_and_update αν άλλαξε.
         - needs-review (replace/delete χωρίς old_text, ή confidence<0.6)
           → enqueue_review με δομημένο payload.
    Δεν σηκώνει ποτέ exception για μία κακή τροποποίηση — μαζεύει notes.
    """
    def _log(msg: str) -> None:
        if log_hook:
            try:
                log_hook("codify", msg)
            except Exception:
                pass

    parsed: List[ParsedAmendment] = []
    try:
        parsed = parse_amendments(raw_text) or []
    except Exception as exc:
        _log(f"parse_amendments failed: {exc}")
        parsed = []

    if not parsed and use_claude:
        try:
            claude_parsed = claude_parse_amendments(raw_text, log_hook=log_hook)
        except Exception as exc:  # pragma: no cover - dormant
            _log(f"claude_parse_amendments failed: {exc}")
            claude_parsed = None
        if claude_parsed:
            parsed = claude_parsed

    if not parsed:
        _log("No amendment instructions found.")
        return []

    # ── Ομαδοποίηση ανά νόμο-στόχο ───────────────────────────────────────────
    by_law: Dict[str, List[ParsedAmendment]] = {}
    for am in parsed:
        by_law.setdefault(am.target_law_id, []).append(am)

    results: List[CodificationResult] = []
    for law_id, ams in by_law.items():
        notes: List[str] = []
        applied = 0
        queued = 0
        diff_text = ""
        current_path: Optional[str] = None
        version_path: Optional[str] = None

        try:
            current = get_current_law_text(law_id)
        except Exception as exc:
            current = ""
            notes.append(f"could not read current law text: {exc}")

        auto_dicts: List[Dict] = []
        for am in ams:
            if _is_auto_applicable(am):
                auto_dicts.append(_to_diff_amendment(am))
            else:
                reason = (
                    "low confidence" if am.confidence < 0.6
                    else "no old_text to locate change in flat law text"
                )
                try:
                    path = enqueue_review(_review_payload(law_id, am, reason))
                    notes.append(f"queued for review ({reason}): {path}")
                except Exception as exc:
                    notes.append(f"failed to enqueue review: {exc}")
                queued += 1

        if auto_dicts:
            try:
                result = apply_amendments(current, auto_dicts)
                notes.extend(result.notes)
                if result.changed:
                    applied = len(auto_dicts)
                    diff_text = result.diff
                    try:
                        current_path, version_path = snapshot_and_update(
                            law_id, result.new_law_text, effective_date, result.diff
                        )
                    except Exception as exc:
                        notes.append(f"snapshot_and_update failed: {exc}")
                else:
                    # Δεν άλλαξε τίποτα — π.χ. replace με old_text που δεν βρέθηκε.
                    for am in ams:
                        if _is_auto_applicable(am) and am.operation in ("replace", "delete"):
                            try:
                                path = enqueue_review(
                                    _review_payload(law_id, am, "auto-apply found no match")
                                )
                                notes.append(f"queued for review (no match): {path}")
                            except Exception as exc:
                                notes.append(f"failed to enqueue review: {exc}")
                            queued += 1
            except Exception as exc:
                notes.append(f"apply_amendments failed: {exc}")

        _log(f"law={law_id} applied={applied} queued={queued}")
        results.append(
            CodificationResult(
                law_id=law_id,
                applied=applied,
                queued_for_review=queued,
                diff=diff_text,
                current_path=current_path,
                version_path=version_path,
                notes=notes,
            )
        )

    return results
