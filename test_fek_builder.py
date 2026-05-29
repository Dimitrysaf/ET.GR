#!/usr/bin/env python3
"""Test the new FEK cabinet_act XML builder."""

import json
from backend.ai.structurer import to_xml

# Sample cabinet_act JSON that the AI should now produce
sample_cabinet_act = {
    "document_type": "cabinet_act",
    "title": "Στελέχωση του Ιδιαίτερου Γραφείου του Υπουργού Εθνικής Οικονομίας και Οικονομικών Κυριάκου Πιερρακάκη, υπό την ιδιότητά του ως Προέδρου του Eurogroup.",
    "document_id": "ΦΕΚ Α' 1/2026",
    "publication_date": "2026-01-08",
    "document_structure": "cabinet_act",
    "action_info": {
        "number": "39",
        "date": "2025-12-23",
        "title": "Στελέχωση του Ιδιαίτερου Γραφείου του Υπουργού Εθνικής Οικονομίας και Οικονομικών Κυριάκου Πιερρακάκη, υπό την ιδιότητά του ως Προέδρου του Eurogroup."
    },
    "organ": "ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ",
    "aitiologia": {
        "intro": "Έχοντας υπόψη:",
        "items": [
            {
                "number": 1,
                "intro": "Tις διατάξεις:",
                "subitems": [
                    {"letter": "α", "text": "Των παρ. 1 και 2 του άρθρου 45..."},
                    {"letter": "β", "text": "Του άρθρου 90..."},
                    {"letter": "γ", "text": "Του άρθρου 50..."}
                ]
            },
            {
                "number": 2,
                "text": "Την ανάγκη αύξησης κατά επτά (7) των θέσεων συνεργατών..."
            }
        ]
    },
    "decision": {
        "intro": "αποφασίζει:",
        "articles": [
            {
                "number": 1,
                "text": "Αυξάνονται κατά επτά (7) οι θέσεις συνεργατών στο Ιδιαίτερο Γραφείο..."
            },
            {
                "number": 2,
                "text": "Οι ανωτέρω θέσεις δύνανται να καλύπτονται..."
            }
        ],
        "publication_instruction": "Η παρούσα Πράξη να δημοσιευθεί στην Εφημερίδα της Κυβέρνησης."
    },
    "signatures": {
        "prothypourgos": "ΚΥΡΙΑΚΟΣ ΜΗΤΣΟΤΑΚΗΣ",
        "members": [
            "ΚΩΝΣΤΑΝΤΙΝΟΣ ΧΑΤΖΗΔΑΚΗΣ",
            "ΚΥΡΙΑΚΟΣ ΠΙΕΡΡΑΚΑΚΗΣ",
            "ΓΕΩΡΓΙΟΣ ΓΕΡΑΠΕΤΡΙΤΗΣ"
        ]
    },
    "contact_info": {
        "address": "Καποδιστρίου 34, Τ.Κ. 104 32 Αθήνα",
        "phone": "210 5279000",
        "url": "https://eservices.et.gr"
    },
    "metadata": {
        "confidence": 0.95
    },
    "amendments": []
}

print("Testing new FEK cabinet_act XML builder...")
print("=" * 80)

# Debug: check conditions
print(f"document_structure: {sample_cabinet_act.get('document_structure')}")
print(f"action_info exists: {bool(sample_cabinet_act.get('action_info'))}")
print(f"Condition to use new builder: {sample_cabinet_act.get('document_structure') == 'cabinet_act' and sample_cabinet_act.get('action_info')}")
print()

try:
    xml_output = to_xml(sample_cabinet_act)
    print("✓ XML generated successfully!")
    print("\nGenerated XML (first 2000 chars):")
    print("-" * 80)
    print(xml_output[:2000])
    print("-" * 80)
    
    # Verify key elements exist in output
    checks = [
        ("EfimeridaKyverniseos", "Root element"),
        ("AitiolologikoMeros", "Reasoning section"),
        ("ApoфasistikоMeros", "Decision section"),
        ("Ypografes", "Signatures section"),
        ("Prothypourgos", "Prime Minister signature"),
        ("MeliYpourgikoySymbouliou", "Cabinet members"),
        ("Epikoinonia", "Contact info"),
    ]
    
    print("\nElement presence checks:")
    for element, description in checks:
        if element in xml_output:
            print(f"  ✓ {element:30} ({description})")
        else:
            print(f"  ✗ {element:30} (MISSING - {description})")
    
    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
