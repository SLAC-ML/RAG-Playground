import os
from openai import OpenAI


# Qwen API
def chat(model, messages):
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )

    # Call the OpenAI API
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Extract the assistant's reply
    reply = completion.choices[0].message.content

    return reply


def models():
    # Retrieve the list of available models
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    models = client.models.list()

    # Extract relevant model details
    model_details = []
    for model in models.data:
        if model.id.startswith('grok-'):
            model_info = {
                'id': model.id,
                'created': model.created,
                'owned_by': model.owned_by,
            }
            model_details.append(model_info)

    return model_details
