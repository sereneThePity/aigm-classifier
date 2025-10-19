"""Evaluate a saved model on a small test set or manifest."""
import argparse
import numpy as np
from tensorflow import keras

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False)
    args = parser.parse_args()
    # tiny demo
    model_path = args.model or '../models/cnn_model.h5'
    model = keras.models.load_model(model_path)
    X = np.random.randn(8, 128, 256).astype('float32')
    y = np.random.randint(0, 2, size=(8,))
    loss, acc = model.evaluate(X, y, verbose=0)
    print('loss', loss, 'acc', acc)
