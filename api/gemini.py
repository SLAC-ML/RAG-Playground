import os
import vertexai
from vertexai.generative_models import GenerativeModel


# Gemini API
def chat(model, messages):
    PROJECT_ID = os.environ.get("GEMINI_API_PROJECT_ID")
    REGION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=REGION)

    # Extract the system instruction from the messages
    system_instruction = None
    for message in messages:
        if message["role"] == "system":
            system_instruction = message["content"]
            break

    client = GenerativeModel(
        model,
        system_instruction=system_instruction,
    )

    # Extract the user message from the messages
    user_message = None
    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
            break
    assert user_message is not None, "User message not found in the input messages"

    # Call the Gemini API
    completion = client.generate_content(user_message)

    # Extract the assistant's reply
    reply = completion.text

    return reply


def models():
    # Extract relevant model details
    model_details = [
        {
            "created": None,
            "id": "gemini-1.5-pro-001",
            "owned_by": "google",
        },
        {
            "created": None,
            "id": "gemini-1.5-pro-002",
            "owned_by": "google",
        },
        {
            "created": None,
            "id": "gemini-1.5-flash-001",
            "owned_by": "google",
        },
        {
            "created": None,
            "id": "gemini-1.5-flash-002",
            "owned_by": "google",
        },
    ]

    return model_details
