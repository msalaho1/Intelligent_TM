"""
this file is used for intent extraction 

Author: Mohamed Salah AbdelRaouf
Email: salahmo50@gmail.com

"""

import json
from transformers import pipeline


# Load pre-defined intents
with open("data/intents.json", "r") as f:
    INTENTS = json.load(f)

# Initialize the BART by Meta AI, a zero shot learning classifier for fallback
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
intents = ["add patient", "assign medication", "schedule follow up"]

def detect_intent(text: str, threshold: float = 0.75) -> dict:
    """
    Detect intent based on pre-defined patterns or fallback to a text classifier.

    Args:
        text (str): The input text to classify the intent from.
        threshold (float): The confidence threshold below which intent is considered ambiguous.

    Returns:
        dict: A dictionary containing:
            - 'intent' (str or None): The detected intent or None if ambiguous.
            - 'error' (str, optional): An error message if the intent is ambiguous.
    """

    # Classify the given text according to dataset 
    for intent, patterns in INTENTS.items():
        if any(pattern in text.lower() for pattern in patterns):
            return {"intent": intent, "confidence": 1.0}

    # Fallback to the zero shot learning classifier
    result = classifier(text, candidate_labels=intents)

    # Getting the maximum score index
    max_index = result['scores'].index(max(result['scores']))
    intent = result['labels'][max_index]
    score = result['scores'][max_index]

    if score < threshold:
        return {"intent": None, "error": "Ambiguous intent. Please clarify your command."}
    
    return {"intent": intent, "confidence": score}
