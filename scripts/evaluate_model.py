import numpy as np
import tensorflow as tf
from preprocess import load_dataset

def evaluate(model_path, manifest_path):
    X, y = load_dataset(manifest_path)
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X)
    preds_bin = (preds > 0.5).astype(int)
    acc = np.mean(preds_bin.flatten() == y)
    print(f"✅ Accuracy: {acc*100:.2f}% on testset")

if __name__ == "__main__":
    evaluate("audio_classifier_model.keras", "data/testset/manifest.csv")
