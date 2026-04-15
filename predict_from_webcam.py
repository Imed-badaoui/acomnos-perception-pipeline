# NOTE: Requires 'opencv-python' (install with `pip install opencv-python`)
import cv2
import numpy as np
import time
import pickle
from tensorflow.keras.models import load_model

# --- Paramètres ---
MODEL_PATH = "model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
IMG_HEIGHT, IMG_WIDTH = 96, 96
MAX_SEQ_LENGTH = 30
DURATION = 3  # secondes
FPS = 10

# --- 1. Charger le modèle et les classes ---
model = load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
class_names = le.classes_

# --- 2. Capture webcam ---
cap = cv2.VideoCapture(0)
frames = []
print("🎥 Préparation...")

time.sleep(1)
print("🎬 Capture en cours...")

start_time = time.time()
while len(frames) < MAX_SEQ_LENGTH and (time.time() - start_time) < DURATION:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frames.append(frame)
    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1 / FPS)

cap.release()
cv2.destroyAllWindows()

# --- 3. Prétraitement ---
if len(frames) < MAX_SEQ_LENGTH:
    frames += [frames[-1]] * (MAX_SEQ_LENGTH - len(frames))
elif len(frames) > MAX_SEQ_LENGTH:
    idx = np.linspace(0, len(frames)-1, MAX_SEQ_LENGTH).astype(int)
    frames = [frames[i] for i in idx]

video_array = np.array(frames).astype("float32") / 255.0
video_array = np.expand_dims(video_array, axis=0)  # (1, 30, 96, 96, 3)

# --- 4. Prédiction ---
pred = model.predict(video_array)
pred_index = np.argmax(pred[0])
pred_label = class_names[pred_index]
confidence = np.max(pred[0])

print(f"✅ Geste prédit : {pred_label} (confiance : {confidence:.2f})")
