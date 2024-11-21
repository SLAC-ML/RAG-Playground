import os
from anthropic import Anthropic


MODEL_MAP = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3.5-sonnet-v2": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-2.1": "claude-2.1",
    "claude-2.0": "claude-2.0",
}


# Anthropic API
def chat(model, messages):
    # Initialize the Anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Extract the system instruction from the messages
    system_instruction = None
    for message in messages:
        if message["role"] == "system":
            system_instruction = message["content"]
            break

    # Format the user message for Anthropic API
    user_messages = []
    for message in messages:
        if message["role"] == "user":
            user_messages.append(message)
            break
    assert user_messages, "User message not found in the input messages"

    # Call the Anthropic API
    completion = client.messages.create(
        max_tokens=2048,
        model=MODEL_MAP[model],
        system=system_instruction,
        messages=user_messages
    )

    # Extract the assistant's reply
    reply = completion.content[0].text

    return reply


def models():
    model_details = [
        {
            "created": None,
            "id": model_name,
            "owned_by": "anthropic",
        }
        for model_name in MODEL_MAP.keys()
    ]

    return model_details
