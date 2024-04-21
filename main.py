"""
Part B  
Deploy the Algorithm/Model built-in Part A in any cloud service provider. Your final algorithm should be 
exposed as a Server API Endpoint. In order to test this API make sure you hit a request to the server to 
get the result as a response to the API. The request-response body should be in the following format:  
Request body: {“text1”: ”nuclear body seeks new tech .......”, ”text2”: ”terror suspects face arrest ......”} 
Response body: {“similarity score”: 0.2 } 
Note: “text1”, “text2”, and “similarity score” keys should be kept as it is, without any change. 

"""

# Importing Libraries 
from flask import Flask, request, jsonify
import tensorflow_hub as hub
import numpy as np
import re

# Loading Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

app = Flask(__name__)

# Function to preprocess text
def preprocess_text(text):
    # Convert contractions
    text = decontracted(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to convert contractions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

@app.route('/calculate_similarity', methods=['POST'])
def get_similarity():
    data = request.json
    if 'text1' not in data or 'text2' not in data:
        return jsonify({"error": "Missing 'text1' or 'text2' in request body"}), 400
    
    text1 = preprocess_text(data['text1'])
    text2 = preprocess_text(data['text2'])
    
    # Calculate similarity score using USE
    similarity_score = calculate_similarity(text1, text2)
    
    return jsonify({"similarity_score": float(similarity_score)})

def calculate_similarity(text1, text2):
    embeddings = model([text1, text2])
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return cosine_similarity

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
  # Running the app on 0.0.0.0 to allow external access
    