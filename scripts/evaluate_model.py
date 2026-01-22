import numpy as np
import tensorflow as tf
import argparse
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

    return acc

def extract_intermediate_activations(model_path, manifest_path, layer_name=None, save_path="intermediate_activations.npy"):
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

    save_path = save_path.replace(".npy", f"_{layer_name}.npy")

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
    np.save("y_test", y)
    print(f"✅ Saved labels to y_test.npy")
    print(f"💾 Features saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio classifier model")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model without transforms')
    eval_parser.add_argument('--model_path', required=True, help='Path to the model')
    eval_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')

    # Subparser for evaluate_with_transform
    eval_trans_parser = subparsers.add_parser('evaluate_transform', help='Evaluate model with transforms')
    eval_trans_parser.add_argument('--model_path', required=True, help='Path to the model')
    eval_trans_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')
    eval_trans_parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bins')
    eval_trans_parser.add_argument('--freq', type=int, default=128, help='Frequency dimension of target shape')
    eval_trans_parser.add_argument('--time', type=int, default=128, help='Time dimension of target shape')
    eval_trans_parser.add_argument('--transform', default='random', help='Transform type')

    # Subparser for extract_intermediate_activations
    extract_parser = subparsers.add_parser('extract', help='Extract intermediate activations')
    extract_parser.add_argument('--model_path', required=True, help='Path to the model')
    extract_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')
    extract_parser.add_argument('--layer_name', help='Layer name to extract from')
    extract_parser.add_argument('--save_path', default='intermediate_activations.npy', help='Path to save features')

    args = parser.parse_args()

    if args.command == 'evaluate':
        evaluate(args.model_path, args.manifest_path)
    elif args.command == 'evaluate_transform':
        target_shape = (args.freq, args.time)
        evaluate_with_transform(args.model_path, args.manifest_path, args.n_mels, target_shape, args.transform)
    elif args.command == 'extract':
        extract_intermediate_activations(args.model_path, args.manifest_path, args.layer_name, args.save_path)
    else:
        parser.print_help()


