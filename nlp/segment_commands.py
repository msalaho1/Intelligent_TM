"""
this file is used for Segmenting input commands

Author: Mohamed Salah AbdelRaouf
Email: salahmo50@gmail.com

"""

import spacy
from typing import List, Tuple

def segment_input(text: str) -> Tuple[List[str], None]:
    """
    Segments the input text into smaller parts based on specific criteria.

    This function uses SpaCy to tokenize the input text and segments it based on the following rules:
    - If the token is "and", "also", or a punctuation mark, it creates a new segment.
    - The remaining tokens after the last segment are added as the final segment.

    Args:
        text (str): The input text to segment.

    Returns:
        List[str]:
            A list of segmented strings.
    """
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the input text
    doc = nlp(text)
    segments = []
    current_segment = []

    for token in doc:
        current_segment.append(token.text) 

        # Check for segment delimiters: "and", "also", or punctuation
        if token.text.lower() in {"and", "also"} or token.is_punct:
            segments.append(" ".join(current_segment).strip())  # Create a segment
            current_segment = []  # Reset the current segment

    if current_segment:
        segments.append(" ".join(current_segment).strip())

    return segments
