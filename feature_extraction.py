# NOTE: Requires 'opencv-python' (install with `pip install opencv-python`)
import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path, static_image_mode=False, min_detection_confidence=0.5):
    """
    Extrait les landmarks Mediapipe d'une vidéo complète.
    Retourne un tableau NumPy de forme (n_frames, 1662)
    """
    holistic = mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence,
        model_complexity=2
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Impossible d’ouvrir la vidéo : {video_path}")
        return None

    landmarks_all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks and results.left_hand_landmarks and results.right_hand_landmarks and results.face_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
            landmarks = np.concatenate([pose, left_hand, right_hand, face])
        else:
            landmarks = np.zeros(1662)  # vecteur nul si détection manquée

        landmarks_all_frames.append(landmarks)

    cap.release()
    holistic.close()

    return np.array(landmarks_all_frames)
