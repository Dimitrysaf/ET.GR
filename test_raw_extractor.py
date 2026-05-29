#!/usr/bin/env python3
"""Test the new faithful raw extractor metadata + archival XML (NO PDF, NO Ollama)."""

from lxml import etree as ET

from backend.fek.raw_extractor import extract_metadata, build_archive_xml

SAMPLE_MASTHEAD = (
    "ΕΦΗΜΕΡΙΔΑ ΤΗΣ ΚΥΒΕΡΝΗΣΕΩΣ\n"
    "ΤΗΣ ΕΛΛΗΝΙΚΗΣ ΔΗΜΟΚΡΑΤΙΑΣ\n"
    "ΤΕΥΧΟΣ ΠΡΩΤΟ    Αρ. Φύλλου 1\n"
    "8 Ιανουαρίου 2026\n"
    "ΠΡΑΞΕΙΣ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ\n"
)


def test_extract_metadata():
    md = extract_metadata(SAMPLE_MASTHEAD)
    assert md["teychos"] == "Α'", md
    assert md["arithmos_fyllou"] == "1", md
    assert md["hmerominia"] == "2026-01-08", md
    assert md["titlos"] == "Εφημερίδα της Κυβερνήσεως της Ελληνικής Δημοκρατίας", md
    assert md["document_type"] == "praxi_ypourgikou_symvouliou", md
    print("✓ extract_metadata OK:", md)


def test_build_archive_xml():
    md = extract_metadata(SAMPLE_MASTHEAD)
    raw_text = "Πρώτη γραμμή\nΔεύτερη γραμμή κειμένου."
    xml_text = build_archive_xml(md, raw_text)

    root = ET.fromstring(xml_text.encode("utf-8"))
    assert root.tag == "FEK", root.tag
    keimeno = root.find("Keimeno")
    assert keimeno is not None and keimeno.text == raw_text, keimeno
    assert root.find("Metadata/Teychos").text == "Α'"
    print("✓ build_archive_xml OK (parses with lxml, contains raw text)")


if __name__ == "__main__":
    test_extract_metadata()
    test_build_archive_xml()
    print("\nAll tests passed.")
