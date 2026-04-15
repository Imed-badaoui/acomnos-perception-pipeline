# NOTE: Requires 'opencv-python' (install with `pip install opencv-python`)
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

def load_data(dataset_path, image_size=(224, 224), max_frames=30):
    data = []
    labels = []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    if not classes:
        print("❌ Aucun dossier de classe trouvé dans le dataset.")
        return None, None

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        for filename in tqdm(os.listdir(class_dir), desc=f"Classe {class_name}"):
            filepath = os.path.join(class_dir, filename)
            if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            frames = extract_frames(filepath, image_size, max_frames)
            if frames is not None:
                data.append(frames)
                labels.append(class_name)

    return np.array(data), np.array(labels)

def extract_frames(video_path, image_size, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Impossible d’ouvrir la vidéo : {video_path}")
        return None

    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Optionnel
        frames.append(frame)
        count += 1
    cap.release()

    if len(frames) < max_frames:
        return None  # vidéo trop courte
    return np.array(frames)

def main():
    if len(sys.argv) != 2:
        print("Usage : python data_preprocessing.py chemin_du_dataset")
        sys.exit(1)

    dataset_path = sys.argv[1]
    print(f"📂 Chargement des vidéos depuis : {dataset_path}")

    data, labels = load_data(dataset_path)

    if data is None or labels is None or len(data) == 0:
        print("❌ Aucun échantillon valide n’a été chargé.")
        sys.exit(1)

    print(f"\n✅ Chargé {len(data)} vidéos valides.")
    print(f"🧾 Classes trouvées : {set(labels)}")
    print("💾 Sauvegarde dans features_labels.npz ...")

    np.savez_compressed("features_labels.npz", features=data, labels=labels)
    print("✅ Données sauvegardées dans : features_labels.npz")

if __name__ == "__main__":
    main()
