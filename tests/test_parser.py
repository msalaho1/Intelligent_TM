import unittest
from nlp.intent_extraction import detect_intent
from nlp.entity_extraction import extract_entities, validate_entities

class TestParser(unittest.TestCase):
    def test_intent_detection(self):
        """Test intent detection functionality."""
        text = "Add a new patient John Doe."
        result = detect_intent(text)
        self.assertEqual(result["intent"], "add patient")
        
        text = "Schedule a follow-up appointment for next Monday."
        result = detect_intent(text)
        self.assertEqual(result["intent"], "schedule follow up")

        text = "assign a medication Aspirin to the patient?"
        result = detect_intent(text,threshold=0.5)
        self.assertEqual(result["intent"], "assign medication")

        text = "This is an ambiguous command."
        result = detect_intent(text)
        self.assertIsNone(result["intent"])

    def test_entity_extraction(self):
        """Test entity extraction functionality."""
        text = "Assign medication Paracetamol 500mg twice a day for John Doe."
        entities = extract_entities(text)
        self.assertEqual(entities["name"], "John Doe")
        self.assertEqual(entities["medication"], "Paracetamol")
        self.assertEqual(entities["dosage"], "500mg")
        self.assertEqual(entities["frequency"], "twice a day")

        text = "Add a new patient Jane Doe 23 years old female with diabetes."
        entities = extract_entities(text)
        self.assertEqual(entities["name"], "Jane Doe")
        self.assertEqual(entities["age"], "23 years old")
        self.assertEqual(entities["gender"], "female")
        self.assertEqual(entities["condition"], "diabetes")

        text = "Schedule a follow-up for John Smith on the 23rd of March."
        entities = extract_entities(text)
        self.assertEqual(entities["name"], "John Smith")
        self.assertEqual(entities["date"], "23rd of March")

    def test_validation(self):
        """Test entity validation functionality."""
        intent = "assign medication"
        entities = {"name": "John Doe", "medication": "Paracetamol"}  # Missing dosage and frequency
        missing, errors = validate_entities(intent, entities)
        self.assertIn("dosage", missing)
        self.assertIn("frequency", missing)

        intent = "add patient"
        entities = {"name": "Jane Doe", "age": "45 years old"}  # Missing condition and gender
        missing, errors = validate_entities(intent, entities)
        self.assertIn("condition", missing)
        self.assertIn("gender", missing)

        intent = "assign medication"
        entities = {"name": "John Doe", "medication": "InvalidMed"}  # Invalid medication
        missing, errors = validate_entities(intent, entities)
        self.assertEqual(errors, ["Invalid medication: InvalidMed. Please provide a valid medication."])

    def test_edge_cases(self):
        """Test edge cases and unexpected inputs."""
        text = " its just random text without any intent."
        result = detect_intent(text)
        self.assertIsNone(result["intent"])

        text = ""
        entities = extract_entities(text)
        self.assertEqual(entities, {})

        intent = "assign medication"
        entities = {}
        missing, errors = validate_entities(intent, entities)
        self.assertIn("name", missing)
        self.assertIn("medication", missing)
        self.assertIn("dosage", missing)
        self.assertIn("frequency", missing)


# run: python -m unittest discover
if __name__ == "__main__":
    unittest.main()
