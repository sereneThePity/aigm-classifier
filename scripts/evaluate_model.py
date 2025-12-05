import numpy as np
import tensorflow as tf
from preprocess import load_dataset, load_dataset_with_transforms

def evaluate(model_path, manifest_path):
    model = tf.keras.models.load_model(model_path)
    X, y = load_dataset(manifest_path)
    preds = model.predict(X)
    preds_bin = (preds > 0.5).astype(int)
    acc = np.mean(preds_bin.flatten() == y)
    print(f"✅ Accuracy: {acc*100:.2f}% on testset")

def evaluate_with_transform(model_path, manifest_path, n_mels=128, target_shape=(128, 128), transform="random"):
    """
    Evaluate model on audio files with random transforms applied.
    
    Args:
        model_path: Path to trained model
        manifest_path: CSV with 'filepath' and 'label' columns
        n_mels: Number of mel bins
        target_shape: Target shape for mel spectrogram (freq, time)
    """
    model = tf.keras.models.load_model(model_path)
    X, y = load_dataset_with_transforms(manifest_path, target_shape=target_shape, n_mels=n_mels, transform=transform)
    
    if len(X) == 0:
        print("❌ No valid samples processed.")
        return
    
    # Run predictions
    preds = model.predict(X, verbose=0)
    preds_bin = (preds > 0.5).astype(int)
    
    # Compute accuracy
    acc = np.mean(preds_bin.flatten() == y)
    print(f"\n✅ Accuracy with transforms: {acc*100:.2f}% ({len(X)} samples)")
    print(f"   Mean confidence: {np.mean(preds):.3f}")
    print(f"   Std confidence: {np.std(preds):.3f}")

def extract_intermediate_activations(model_path, manifest_path, layer_name=None, save_path="intermediate_features.npy"):
    # Load trained classifier
    model = tf.keras.models.load_model(model_path)
    model.summary()  # useful to see layer names if you don't know them yet

    # Load your dataset
    X, y = load_dataset(manifest_path)

    # Pick layer by name or default to penultimate
    if layer_name is None:
        # automatically pick the 2nd to last layer
        layer_name = model.layers[-3].name
        print(f"ℹ️  No layer_name provided. Using penultimate layer: '{layer_name}'")

    # Build a new model up to that layer
    feature_extractor = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=model.get_layer(layer_name).output
    )

    # Compute activations
    features = feature_extractor.predict(X, batch_size=32, verbose=1)
    print(f"✅ Extracted features from layer '{layer_name}', shape: {features.shape}")

    # Optional: flatten features if needed
    if len(features.shape) > 2:
        features = features.reshape((features.shape[0], -1))
        print(f"Flattened feature shape: {features.shape}")

    # Save to disk for later use
    np.save(save_path, features)
    print(f"💾 Features saved to {save_path}")


if __name__ == "__main__":
    # Evaluate without transforms
    # evaluate("models/audio_classifier_model.keras", "data/testset/manifest.csv")
    
    # Evaluate with random transforms (for robustness testing)
    evaluate_with_transform(
        "models/audio_classifier_model.keras", 
        "data/testset/manifest.csv",
        n_mels=128,
        target_shape=(128, 128),
        transform="add_noise" 
    )
    
    # extract_intermediate_activations("models/audio_classifier_model.keras", "data/testset/manifest.csv", layer_name=None, save_path="intermediate_features.npy")


