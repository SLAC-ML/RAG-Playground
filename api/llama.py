import os
import openai
from google.auth import default
from google.auth.transport import requests


MODEL_MAP = {
    "llama-3.1-405b": "meta/llama-3.1-405b-instruct-maas",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct-maas",
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct-maas",
    "llama-3.2": "meta/llama-3.2-90b-vision-instruct-maas",
}


# Gemini API
def chat(model, messages):
    PROJECT_ID = os.environ.get("GEMINI_API_PROJECT_ID")
    REGION = "us-central1"
    MAAS_ENDPOINT = "us-central1-aiplatform.googleapis.com"

    # Get credentials
    credentials, _ = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    auth_request = requests.Request()
    credentials.refresh(auth_request)

    # Initialize the OpenAI client
    client = openai.OpenAI(
        base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi",
        api_key=credentials.token
    )

    # Call the OpenAI API
    completion = client.chat.completions.create(
        model=MODEL_MAP[model],
        messages=messages
    )

    # Extract the assistant's reply
    reply = completion.choices[0].message.content

    return reply


def models():
    # Extract relevant model details
    model_details = [
        {
            "created": None,
            "id": "llama-3.1-405b",
            "owned_by": "meta",
        },
        {
            "created": None,
            "id": "llama-3.1-70b",
            "owned_by": "meta",
        },
        {
            "created": None,
            "id": "llama-3.1-8b",
            "owned_by": "meta",
        },
        {
            "created": None,
            "id": "llama-3.2",
            "owned_by": "meta",
        },
    ]

    return model_details
