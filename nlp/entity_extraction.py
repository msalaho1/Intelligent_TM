"""
this file is used for Entity extraction.

Author: Mohamed Salah AbdelRaouf
Email: salahmo50@gmail.com

"""


import spacy
from spacy.matcher import Matcher
import pandas as pd

# Load the disease-specific model
nlp_md = spacy.load("en_ner_bc5cdr_md")

# Load a more generic model
nlp_nm= spacy.load("en_core_web_sm")

# Define required fields for each intent
REQUIRED_FIELDS = {
    "add patient": ["name", "age", "condition", "gender"],
    "assign medication": ["name", "medication", "dosage", "frequency"],
    "schedule followup": ["name", "date"]
}

# Load valid medications from CSV file
VALID_MEDICATIONS = pd.read_csv("data/valid_medications.csv")["Medication"].tolist()

def extract_entities(text: str) -> dict:
    """
    Extract entities like name, age, medication, dosage, date, gender, and frequency using spaCy and pattern matching.

    Args:
        text (str): The input text to extract entities from.

    Returns:
        dict: A dictionary containing the detected entities such as name, age, medication, dosage, date, gender, and frequency.
    """
    doc_nm = nlp_nm(text)
    doc_md = nlp_md(text)
    matcher = Matcher(nlp_md.vocab)

    # Define patterns for name, age, dosage, date, gender, and frequency
    age_pattern = [{"LIKE_NUM": True}, {"LOWER": "years"}, {"LOWER": "old"}]  # Matches age like '23 years old'
    dosage_pattern = [{"LIKE_NUM": True}, {"LOWER": "mg"}]  # Matches dosage like '500 mg'
    date_pattern = [
        # Matches formats like "20th Jan", "20 January", "20 Jan.", "monday 23th of december 8:30 pm"
        {"LOWER": {"IN": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}, "OP": "?"},
        {"TEXT": {"REGEX": "\\d{1,2}(st|nd|rd|th)?"}},
        {"LOWER": {"IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": ["jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june", "jul", "july", "aug", "august", "sep", "september", "oct", "october", "nov", "november", "dec", "december"]}},
        {"TEXT": {"REGEX": "\\d{1,2}(:\\d{2})?"}, "OP": "?"},
        {"LOWER": {"IN": ["am", "pm"]}, "OP": "?"}
    ]
    gender_pattern = [
        {"LOWER": {"IN": ["male", "female", "man", "woman", "boy", "girl"]}}
    ]
    frequency_pattern = [
        {"LOWER": {"IN": ["once", "twice", "daily", "weekly", "monthly"]}},
        {"LOWER": {"IN": ["a", "per"]}, "OP": "?"},
        {"LOWER": {"IN": ["day", "week", "month"]}, "OP": "?"}
    ]

    # Add patterns to the matcher
    matcher.add("AGE", [age_pattern])
    matcher.add("DOSAGE", [dosage_pattern])
    matcher.add("DATE", [date_pattern])
    matcher.add("GENDER", [gender_pattern])
    matcher.add("FREQUENCY", [frequency_pattern])

    # Apply matcher to the document
    matches = matcher(doc_md)
    entities = {}

    # Extract entities from spaCy NER
    for ent in doc_nm.ents:
        if ent.label_ == "PERSON":
            entities["name"] = ent.text  # Overwrite if a better name is detected
           
    for ent in doc_md.ents:
        if ent.label_ == "DISEASE":
            entities["condition"] = ent.text
        elif ent.label_ == "DATE":
            entities["date"] = ent.text
        elif ent.label_ in ["ORG", "PRODUCT", "CHEMICAL"]:
            entities["medication"] = ent.text
    
    # Extract entities from pattern matches
    for match_id, start, end in matches:
        span = doc_md[start:end]
        label = nlp_md.vocab.strings[match_id]

        if label == "AGE":
            entities["age"] = span.text
        elif label == "GENDER":
            entities["gender"] = span.text
        elif label == "DOSAGE":
            entities["dosage"] = span.text
        elif label == "DATE":
            entities["date"] = span.text
        elif label == "FREQUENCY":
            entities["frequency"] = span.text
    return entities

def validate_entities(intent: str, entities: dict) -> tuple:
    """
    Validates that all required fields for the intent are present and that the given medications are valid.

    Args:
        intent (str): The intent for which validation is being performed.
        entities (dict): A dictionary containing the detected entities.

    Returns:
        tuple: A tuple containing:
            - A list of missing fields (if any).
            - A list of errors (if any, such as invalid medication).
    """
    required_fields = REQUIRED_FIELDS.get(intent, [])
    missing_fields = [field for field in required_fields if field not in entities]
    
    # Validate medication if intent is 'assign medication'
    errors = []
    if intent == "assign medication" and "medication" in entities:
        medication = entities["medication"]
        if medication not in VALID_MEDICATIONS:
            errors.append(f"Invalid medication: {medication}. Please provide a valid medication.")
    return missing_fields, errors
