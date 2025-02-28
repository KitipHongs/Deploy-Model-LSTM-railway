import cv2
import numpy as np
import os
import json
import tensorflow as tf
import mediapipe as mp

# Load gesture classes
with open("gesture_classes.json", 'r') as f:
    gesture_names = json.load(f)

# Load the trained model
model = tf.keras.models.load_model("flip_hand_gesture_model.h5")


# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def extract_landmarks(frame):
    # Define upper body indices outside the conditional block
    upper_body_indices = [
        0,  # nose
        11, 12,  # shoulders
        13, 14, 15, 16,  # arms
        23, 24,  # hips
        19, 20,  # elbows
        21, 22,  # wrists
    ]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    landmarks = []
    
    # 1. Extract Hand landmarks (21 points per hand)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))  # Pad with zeros if no left hand detected
        
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))  # Pad with zeros if no right hand detected
    
    # 2. Extract Face mesh landmarks (468 points)
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (468 * 3))  # Pad with zeros if no face detected
    
    # 3. Extract Upper body landmarks
    if results.pose_landmarks:
        # Extract upper body points (shoulders, arms, torso)
        for idx in upper_body_indices:
            landmark = results.pose_landmarks.landmark[idx]
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (len(upper_body_indices) * 3))
    
    return landmarks

def draw_landmarks(frame, results):
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    # Draw face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())

def get_gesture_name(class_idx):
    return gesture_names[class_idx] if 0 <= class_idx < len(gesture_names) else "Unknown"

def real_time_prediction():
    cap = cv2.VideoCapture(1)
    sequence = []
    is_recording = False
    recording_frames = 0
    SEQUENCE_LENGTH = 30
    PREDICTION_THRESHOLD = 0.8
    final_prediction = None
    waiting_for_hand = False
    
    print("\nStarting real-time prediction... Press 'r' to start recording, 'c' to clear, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Extract landmarks only if we're recording
        if is_recording or waiting_for_hand:
            landmarks = extract_landmarks(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            is_recording = False
            waiting_for_hand = True
            sequence = []
            recording_frames = 0
            final_prediction = None
            print("Waiting for hand detection...")
        elif key == ord('c'):
            is_recording = False
            waiting_for_hand = False
            sequence = []
            recording_frames = 0
            final_prediction = None
            print("Predictions cleared.")
        elif key == ord('q'):
            break
        
        # Check for hand detection when waiting
        if waiting_for_hand:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                is_recording = True
                waiting_for_hand = False
                print("Hand detected! Recording started...")
            else:
                cv2.putText(display_frame, "Waiting for hand detection...",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Only record frames if we're recording and have detected hands
        if is_recording and recording_frames < SEQUENCE_LENGTH:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                sequence.append(landmarks)
                recording_frames += 1
                
                cv2.putText(display_frame, f"Recording: {recording_frames}/{SEQUENCE_LENGTH}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If we lose hand detection during recording
                cv2.putText(display_frame, "Hand lost! Please show your hand",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if recording_frames == SEQUENCE_LENGTH:
                is_recording = False
                sequence_array = np.array([sequence])
                prediction = model.predict(sequence_array, verbose=0)
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                if confidence > PREDICTION_THRESHOLD:
                    final_prediction = get_gesture_name(class_idx)
                else:
                    final_prediction = "Uncertain"
                
                print(f"Prediction: {final_prediction} (confidence: {confidence:.2f})")
        
        cv2.putText(display_frame, f"Press 'r' to record, 'c' to clear, 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if final_prediction:
            cv2.putText(display_frame, f"Prediction: {final_prediction}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Real-Time Gesture Recognition', display_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


real_time_prediction()