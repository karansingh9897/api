#!/bin/bash
echo "Starting Face Recognition API..."
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Flask server..."
python app.py
