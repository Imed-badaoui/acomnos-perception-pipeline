import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, TimeDistributed, LSTM, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from data_generator import load_data
import numpy as np
import os
import pickle

DATA_PATH = r"/app/data/features_labels.npz"
MAX_SEQ_LENGTH = 30
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32
EPOCHS = 10

def build_model(seq_length, img_height, img_width, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # ou True partiellement si fine tuning

    inputs = Input(shape=(seq_length, img_height, img_width, 3))

    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # LSTM avec dropout pour régularisation
    x = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(x)
    x = Dropout(0.5)(x)  # Dropout après LSTM

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("📦 Chargement des données ...")
    features, labels, le = load_data(DATA_PATH, MAX_SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH)

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    model = build_model(MAX_SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, len(le.classes_))
    print(model.summary())

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',       # ou 'val_accuracy' si tu préfères
    patience=10,
    restore_best_weights=True
)
    print("🛠️ Entraînement du modèle ...")

    model.fit(features, labels,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2,
              callbacks=[early_stopping])

    os.makedirs("saved_models", exist_ok=True)

    with open(os.path.join("saved_models", "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("Label encoder sauvegardé.")

    model_path = os.path.join("saved_models", "model.keras")
    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")

if __name__ == "__main__":
    main()
