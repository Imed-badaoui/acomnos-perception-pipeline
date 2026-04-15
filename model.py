import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def build_model(seq_length=50, img_height=224, img_width=224, num_classes=4):
    input_shape = (seq_length, img_height, img_width, 3)
    inputs = Input(shape=input_shape)

    # Backbone ResNet50 avec poids ImageNet, sans la couche FC finale
    base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_cnn.trainable = False  # Geler le CNN

    # Appliquer ResNet frame par frame
    x = TimeDistributed(base_cnn)(inputs)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
