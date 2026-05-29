import json
from backend.ai.structurer import structure_raw_text

text = """
ΕΦΗΜΕΡΙΔΑΤΗΣ ΚΥΒΕΡΝΗΣΕΩΣΤΗΣ ΕΛΛΗΝΙΚΗΣ ΔΗΜΟΚΡΑΤΙΑΣ
8 Ιανουαρίου 2026 ΤΕΥΧΟΣ ΠΡΩΤΟ Αρ. Φύλλου 1
ΠΡΑΞΕΙΣ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ
Πράξη 39 της 23-12-2025
Στελέχωση του Ιδιαίτερου Γραφείου του Υπουργού Εθνικής Οικονομίας και Οικονομικών Κυριάκου
Πιερρακάκη, υπό την ιδιότητά του ως Προέδρου του Eurogroup.
ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ
Έχοντας υπόψη:
1. Tις διατάξεις:
α) Των παρ. 1 και 2 του άρθρου 45, του άρθρου 46 και ιδίως της παρ. 6 σε συνδυασμό με την παρ. 3 αυτού,
των άρθρων 47, 47Α και 48, καθώς και της περ. 22 του άρθρου 119 του ν.  4622/2019 «Επιτελικό Κράτος»
"""

# We can't really run chat_json here because it needs Ollama, but we can check the chunking logic.
from backend.ai.structurer import _split_into_chunks
chunks = _split_into_chunks(text)
print(f"Chunks: {len(chunks)}")
print(f"Chunk 0 size: {len(chunks[0])}")
