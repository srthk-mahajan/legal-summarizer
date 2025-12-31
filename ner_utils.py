"""
ner_utils.py
Lightweight regex-based Named Entity Extraction for legal judgments.
(No spaCy dependency — fully standalone and NumPy-safe.)
"""

import re


def extract_legal_entities(text: str):
    """
    Extracts key legal entities like Judge names, Case numbers,
    Dates, Statutes, and Citations using regex heuristics only.
    """
    entities = {
        "Judge_Names": set(),
        "Case_Numbers": set(),
        "Dates": set(),
        "Statutes": set(),
        "Citations": set(),
    }

    # --- Judge Names ---
    # Matches patterns like "Justice D.Y. Chandrachud", "Hon'ble Mr. Justice Nariman"
    for match in re.finditer(
        r"(?:Hon'?ble\s+)?(?:Mr\.|Mrs\.|Ms\.)?\s*Justice\s+[A-Z][A-Za-z\.\s,]+",
        text,
        flags=re.I,
    ):
        entities["Judge_Names"].add(match.group(0).strip())

    # --- Case Numbers ---
    for match in re.finditer(
        r"\b(?:Crl\.|Cr\.|Civil|W\.?P\.?|Appeal|S\.?L\.?P\.?)\s*No\.?\s*\d{1,5}/\d{2,4}\b",
        text,
        flags=re.I,
    ):
        entities["Case_Numbers"].add(match.group(0))

    # --- Dates ---
    for match in re.finditer(
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4}\b"
        r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        text,
        flags=re.I,
    ):
        entities["Dates"].add(match.group(0))

    # --- Statutes / Sections ---
    for match in re.finditer(r"\b(?:Section|S\.|Article|Art\.)\s*\d+[A-Z]?\b", text, flags=re.I):
        entities["Statutes"].add(match.group(0))
    acts = [
        "Constitution of India",
        "Indian Penal Code",
        "Code of Criminal Procedure",
        "CrPC",
        "IPC",
        "Evidence Act",
        "Companies Act",
        "Contract Act",
        "IT Act",
    ]
    for act in acts:
        for match in re.finditer(re.escape(act), text, flags=re.I):
            entities["Statutes"].add(match.group(0))

    # --- Citations ---
    for match in re.finditer(
        r"\(?\d{4}\)?\s*\d*\s*(SCC|AIR|CriLJ|All LJ|SCR|Comp Cas|RCR|LLJ)\s*\d+",
        text,
        flags=re.I,
    ):
        entities["Citations"].add(match.group(0))

    # Clean up and convert sets to sorted lists
    for key in entities:
        entities[key] = sorted(list(entities[key]))

    return entities


# ✅ Quick standalone test
if __name__ == "__main__":
    sample = """In (2012) 5 SCC 777, Justice A.K. Goel delivered judgment in Crl. No. 145/2022 
    under Section 482 CrPC. Decided on 12 April 2013 by Hon'ble Mr. Justice Chandrachud."""
    result = extract_legal_entities(sample)
    for k, v in result.items():
        print(f"{k}: {v}")
