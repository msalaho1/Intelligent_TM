from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uuid
import json
import os
from threading import Lock
from typing import Dict, List, Tuple, Any

from nlp.intent_extraction import detect_intent
from nlp.entity_extraction import extract_entities, validate_entities
from nlp.segment_commands import segment_input

app = FastAPI()

# Configure logging
logging.basicConfig(
    filename="logs/bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Path to the JSON file storing conversations
CONVERSATIONS_FILE = "./data/conversations.json"

# Lock for thread-safe file operations
file_lock = Lock()

# Input model
class TaskInput(BaseModel):
    conversation_id: str = None
    text: str

def log_request_response(input_text: str, response: Dict[str, Any]) -> None:
    """
    Save the input text and response to a log file.

    Args:
        input_text (str): The input text provided by the user.
        response (Dict[str, Any]): The response generated for the input.
    """
    logging.info(f"Input: {input_text} | Response: {response}")

def read_conversations() -> Dict[str, Any]:
    """
    Read the conversations from the JSON file.

    Returns:
        Dict[str, Any]: The conversations data read from the file.
    """
    if not os.path.exists(CONVERSATIONS_FILE):
        return {}
    with open(CONVERSATIONS_FILE, "r") as file:
        return json.load(file)

def write_conversations(conversations: Dict[str, Any]) -> None:
    """
    Write the updated conversations to the JSON file.

    Args:
        conversations (Dict[str, Any]): The conversations data to write to the file.
    """
    with open(CONVERSATIONS_FILE, "w") as file:
        json.dump(conversations, file, indent=4)

@app.post("/parse-task/")
def parse_task(input: TaskInput) -> Dict[str, Any]:
    """
    Process input, detect intent, extract entities, and validate completeness.
    Save each conversation with a unique ID into a JSON file.

    Args:
        input (TaskInput): The input data containing the text and optional conversation ID.

    Returns:
        Dict[str, Any]: The response for the input text.
    """
    if not input.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )

    conversation_id = input.conversation_id or str(uuid.uuid4())

    # Initialize conversation state
    state = {
        "intent": None,
        "entities": {},
        "missing_fields": [],
        "state": "start"
    }

    # Segment commands to detect multiple command entries
    commands = segment_input(input.text)
    response = {}
    final_entities = {}

    for i, command in enumerate(commands):
        # Detect intent
        intent_result = detect_intent(command)
        if intent_result.get("error"):
            response[f"Request number {i+1}"] = {"status": "error", "message": intent_result["error"]}
            log_request_response(command, response)
            continue
        intent = intent_result["intent"]
        state["intent"] = intent

        # Extract entities
        entities = extract_entities(command)
        final_entities = entities

        # Validate required fields and missing fields
        missing_fields, errors = validate_entities(state["intent"], final_entities)
        if missing_fields or errors:
            state["missing_fields"] = missing_fields
            response[f"Request number {i+1}"] = {
                "status": "incomplete",
                "message": f"Please provide: {', '.join(errors)} {', '.join(missing_fields)} to complete the task.",
                "entities": final_entities
            }
            log_request_response(command, response)
            continue

        # Final response if all required information is collected
        response[f"Request number {i+1}"] = {
            "status": "success",
            "intent": state["intent"],
            "entities": final_entities,
        }

    # Save the conversation to the JSON file
    with file_lock:
        conversations = read_conversations()
        conversations[conversation_id] = {
            "input": input.text,
            "response": response
        }
        write_conversations(conversations)

    log_request_response(input.text, response)
    return response
