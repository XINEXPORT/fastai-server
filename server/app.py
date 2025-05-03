import json
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_embedding, cosine_search

app = Flask(__name__)
CORS(app)  # Enable CORS so React or external clients can talk to Flask

# Load the text + embeddings into memory
with open(os.path.join(os.path.dirname(__file__), "embeddings.json"), "r") as f:
    documents = json.load(f)

@app.route('/')
def index():
    return 'FAST AI Landing Page'

@app.route('/api/greet', methods=['GET'])
def greet():
    return jsonify({'message': 'Hello from Flask!'})

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    query_embedding = get_embedding(user_query)
    top_texts = cosine_search(query_embedding, documents)

    return jsonify({
        "query": user_query,
        "results": top_texts
    })

if __name__ == '__main__':
    app.run(debug=True)
