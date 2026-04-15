# NOTE: Requires 'opencv-python' (install with `pip install opencv-python`)
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation simple
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

def load_data(data_path, max_seq_length, img_height, img_width):
    print(f"📦 Chargement de : {data_path}")
    data = np.load(data_path, allow_pickle=True)
    features = data['features']
    labels = data['labels']

    num_samples = len(features)
    padded = np.zeros((num_samples, max_seq_length, img_height, img_width, 3), dtype='float32')

    for i, seq in enumerate(features):
        seq_len = min(len(seq), max_seq_length)
        for j in range(seq_len):
            resized = cv2.resize(seq[j], (img_width, img_height))
            img = resized / 255.0
            # Appliquer augmentation image par image
            img = datagen.random_transform(img)
            padded[i, j] = img

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    print(f"Nombre de classes : {len(le.classes_)}")

    return padded, labels_enc, le
