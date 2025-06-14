import os
import cv2
import numpy as np
import pickle
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ---------- SETUP ----------

# Path to dataset of known faces
dataset_path = '/workspaces/api/face_images'

# Initialize face recognition model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# Load embeddings if already saved
embeddings_path = "embeddings.pkl"
if os.path.exists(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        embeddings, labels = pickle.load(f)
else:
    embeddings, labels = [], []
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = face_app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)
            labels.append(img_name)
    with open(embeddings_path, 'wb') as f:
        pickle.dump((embeddings, labels), f)

# ---------- FASTAPI APP ----------

app = FastAPI()

class RecognizeRequest(BaseModel):
    image_url: str

@app.post("/recognize")
def recognize_face(data: RecognizeRequest):
    if not data.image_url:
        raise HTTPException(status_code=400, detail="Missing image_url in request")

    # Download and decode image
    try:
        response = requests.get(data.image_url)
        response.raise_for_status()
        npimg = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching or decoding image: {e}")

    # Run face detection
    faces = face_app.get(img)
    if not faces:
        return {"error": "No face detected"}

    query_embedding = faces[0].embedding
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    best_idx = np.argmax(similarities)

    return {
        "matched_label": labels[best_idx],
        "similarity_score": float(similarities[best_idx])
    }
