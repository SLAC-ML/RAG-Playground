from flask import Flask, request, jsonify
import time
import api.openai
import api.qwen
import api.gemini
import api.llama
import api.claude
import api.grok
from rag import base


# Initialize the knowledge base
base.init_knowledge_base()

# Initialize the Flask app
app = Flask(__name__)

# Cache variables for models
cached_models = None
cache_timestamp = 0
cache_duration = 3600  # Cache duration in seconds (e.g., 1 hour)


# Routes
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    messages = data.get('messages')
    model = data.get('model', 'gpt-4o-mini')  # Default model if none provided

    if not messages:
        return jsonify({'error': 'No messages provided'}), 400

    try:
        if model.startswith('gpt-'):
            reply = api.openai.chat(model, messages)
        elif model.startswith('qwen-'):
            reply = api.qwen.chat(model, messages)
        elif model.startswith('gemini-'):
            reply = api.gemini.chat(model, messages)
        elif model.startswith('llama-'):
            reply = api.llama.chat(model, messages)
        elif model.startswith('claude-'):
            reply = api.claude.chat(model, messages)
        elif model.startswith('grok-'):
            reply = api.grok.chat(model, messages)
        else:
            # TODO: Use custom InvalidModelError
            # since model can be non-GPT model
            raise NotImplementedError("Invalid model specified.")

        # Return the reply to the client
        return jsonify({'reply': reply}), 200

    except NotImplementedError as e:
        app.logger.error(f"Invalid request: {e}", exc_info=True)
        return jsonify({'error': 'Invalid model specified.'}), 400
    except Exception as e:
        app.logger.error(f"Exception occurred: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    global cached_models, cache_timestamp

    try:
        current_time = time.time()

        # Check if cache is expired or doesn't exist
        if not cached_models or (current_time - cache_timestamp) > cache_duration:
            model_details = []
            model_details += api.openai.models()
            model_details += api.qwen.models()
            model_details += api.gemini.models()
            model_details += api.llama.models()
            model_details += api.claude.models()
            model_details += api.grok.models()

            # Update cache
            cached_models = model_details
            cache_timestamp = current_time
        else:
            model_details = cached_models

        # Return the list of models to the client
        return jsonify({'models': model_details}), 200

    except Exception as e:
        app.logger.error(f"Exception occurred: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/list_entries', methods=['GET'])
def list_entries():
    n = int(request.args.get('n', 0))  # if n == 0, return all entries

    entries = base.list_entries(n)
    return jsonify({'entries': entries}), 200


@app.route('/add_entries', methods=['POST'])
def add_entries():
    data = request.get_json()
    entry_list = data.get('entries')
    if not entry_list or not isinstance(entry_list, list):
        return jsonify({'error': 'A list of entries is required.'}), 400

    base.add_entries(entry_list)
    return jsonify({'message': 'Entries added successfully.'})


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    top_k = int(request.args.get('top_k', 5))
    if not query:
        return jsonify({'error': 'Query parameter is required.'}), 400

    results = base.search(query, top_k)
    return jsonify({'results': results}), 200
