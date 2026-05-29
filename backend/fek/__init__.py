"""
ΦΕΚ Level 1 — πιστή εξαγωγή (faithful extraction).

Αυτό το package κρατά το ΦΕΚ ως πρωτογενή νομική πηγή: RAW κείμενο αυτούσιο +
metadata μέσω regex. ΔΕΝ κάνει AI restructuring ολόκληρου του εγγράφου.
"""

from .raw_extractor import (
    extract_raw_text,
    extract_metadata,
    build_archive_xml,
    build_archive_html,
)

__all__ = [
    "extract_raw_text",
    "extract_metadata",
    "build_archive_xml",
    "build_archive_html",
]
