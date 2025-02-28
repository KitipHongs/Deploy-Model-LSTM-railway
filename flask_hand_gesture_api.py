from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import tensorflow as tf
import mediapipe as mp
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load gesture classes
with open("gesture_classes.json", 'r') as f:
    gesture_names = json.load(f)

# Load trained model
model = tf.keras.models.load_model("flip_hand_gesture_model.h5")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

SEQUENCE_LENGTH = 30
sequence = []


def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    landmarks = []

    # Extract hand landmarks
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))
    
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))
    
    return landmarks


@app.route('/predict', methods=['POST'])
def predict():
    global sequence
    
    file = request.files['frame']
    image = Image.open(BytesIO(file.read()))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    landmarks = extract_landmarks(frame)
    sequence.append(landmarks)
    
    if len(sequence) == SEQUENCE_LENGTH:
        sequence_array = np.array([sequence])
        prediction = model.predict(sequence_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        sequence = []  # Reset sequence
        
        return jsonify({"gesture": gesture_names[class_idx], "confidence": float(confidence)})
    
    return jsonify({"message": "Frame received. Waiting for more frames."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
