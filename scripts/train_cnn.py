"""Train a simple CNN classifier on mel-spectrogram inputs."""
import argparse
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_simple_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape(input_shape + (1,))(inputs)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=False)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    # Minimal demo: create random data
    num_classes = 2
    input_shape = (128, 256)
    X = np.random.randn(16, *input_shape).astype('float32')
    y = np.random.randint(0, num_classes, size=(16,))

    model = build_simple_cnn(input_shape, num_classes)
    model.fit(X, y, epochs=args.epochs, batch_size=4)
    os.makedirs('../models', exist_ok=True)
    model.save('../models/cnn_model.h5')
