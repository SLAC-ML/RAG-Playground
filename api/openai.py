import os
from openai import OpenAI


# OpenAI API
def chat(model, messages):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
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
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    models = client.models.list()

    # Extract relevant model details
    model_details = []
    for model in models.data:
        if model.id.startswith('gpt-'):
            model_info = {
                'id': model.id,
                'created': model.created,
                'owned_by': model.owned_by,
            }
            model_details.append(model_info)

    return model_details
