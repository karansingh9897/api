import os
import cv2
import numpy as np
import pickle
import requests
from flask import Flask, request, jsonify
from pyngrok import ngrok
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Prepare face model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# Load saved embeddings
with open('/content/embeddings.pkl', 'rb') as f:
    embeddings, labels = pickle.load(f)

# Set up Flask app
app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()

    if not data or 'image_url' not in data:
        return jsonify({'error': 'image_url missing in request'}), 400

    image_url = data['image_url']

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        npimg = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch or decode image: {str(e)}'}), 400

    if img is None:
        return jsonify({'error': 'Invalid image data'}), 400

    faces = face_app.get(img)
    if not faces:
        return jsonify({'error': 'No face detected'}), 200

    query_embedding = faces[0].embedding
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    best_idx = np.argmax(similarities)

    return jsonify({
        'matched_label': labels[best_idx],
        'similarity_score': float(similarities[best_idx])
    })

# Start ngrok and Flask server
from pyngrok import conf
conf.get_default().auth_token = "2yRylhUTF26AhTsIQfmytyeMnNN_7DrHZR8A37fVXoMKSS4Gy"
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

app.run(port=5000)
