import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import numpy as np
import torch
import argparse
from pathlib import Path
from preprocess import load_dataset, load_dataset_with_transforms, load_dataset_comprehensive
from utils import ROOT_DIR, DATA_DIR
import tensorflow as tf
from train_cnn import SimpleCNN

def load_model_auto(model_path):
    """
    Automatically load either PyTorch or Keras model based on file extension.
    
    Args:
        model_path: Path to model file (.pt for PyTorch, .keras for Keras)
    
    Returns:
        Tuple of (model, model_type) where model_type is 'pytorch' or 'keras'
    """
    model_path = str(model_path)
    
    if model_path.endswith('.pt'):
        # Load PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded_obj = torch.load(model_path, map_location=device)
        
        # Check if it's a state_dict (OrderedDict) or full model
        if isinstance(loaded_obj, dict):
            # It's a state_dict - need to reconstruct the model
            # Load model metadata from training_info.json if available
            info_path = model_path.replace('cnn_model.pt', 'training_info.json').replace('.pt', '_info.json')
            if not os.path.exists(info_path):
                # Try alternate path
                base_dir = os.path.dirname(model_path)
                info_path = os.path.join(base_dir, 'training_info.json')
            
            if os.path.exists(info_path):
                import json
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    input_shape = info.get('input_shape', [128, 128])
                    num_classes = info.get('num_classes', 2)
            else:
                # Default values
                input_shape = [128, 128]
                num_classes = 2
            
            model = SimpleCNN(input_shape, num_classes)
            model.load_state_dict(loaded_obj)
        else:
            # It's already a full model object
            model = loaded_obj
        
        model.eval()
        return model, 'pytorch'
    elif model_path.endswith('.keras'):
        # Load Keras model
        model = tf.keras.models.load_model(model_path)
        return model, 'keras'
    else:
        raise ValueError(f"Unsupported model format: {model_path}. Use .pt (PyTorch) or .keras (Keras)")

def evaluate(model_path, manifest_path, codec_name=None):
    model, model_type = load_model_auto(model_path)
    X, y = load_dataset_comprehensive(manifest_path, codec_name=codec_name)
    
    if model_type == 'pytorch':
        # Convert to torch tensor and move to same device as model
        X_tensor = torch.from_numpy(X).float()
        device = next(model.parameters()).device
        X_tensor = X_tensor.to(device)
        
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
    else:
        # Keras model
        preds = model.predict(X)
    
    # Handle multi-class output: take argmax if preds has shape (N, num_classes), else use threshold
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds_bin = np.argmax(preds, axis=1)
    else:
        preds_bin = (preds.flatten() > 0.5).astype(int)
    
    acc = np.mean(preds_bin == y)
    print(f"✅ Accuracy: {acc*100:.2f}% on testset")

def evaluate_with_transform(model_path, manifest_path, n_mels=128, target_shape=(128, 128), transform="random"):
    """
    Evaluate model on audio files with random transforms applied.
    
    Args:
        model_path: Path to trained model (.pt for PyTorch, .keras for Keras)
        manifest_path: CSV with 'filepath' and 'label' columns
        n_mels: Number of mel bins
        target_shape: Target shape for mel spectrogram (freq, time)
    """
    model, model_type = load_model_auto(model_path)
    X, y = load_dataset_with_transforms(manifest_path, target_shape=target_shape, n_mels=n_mels, transform=transform)
    
    if len(X) == 0:
        print("❌ No valid samples processed.")
        return
    
    # Run predictions
    if model_type == 'pytorch':
        X_tensor = torch.from_numpy(X).float()
        device = next(model.parameters()).device
        X_tensor = X_tensor.to(device)
        
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
    else:
        # Keras model
        preds = model.predict(X, verbose=0)
    
    # Handle multi-class output: take argmax if preds has shape (N, num_classes), else use threshold
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds_bin = np.argmax(preds, axis=1)
    else:
        preds_bin = (preds.flatten() > 0.5).astype(int)
    
    # Compute accuracy
    acc = np.mean(preds_bin == y)
    print(f"\n✅ Accuracy with transforms: {acc*100:.2f}% ({len(X)} samples)")
    print(f"   Mean confidence: {np.mean(preds):.3f}")
    print(f"   Std confidence: {np.std(preds):.3f}")

    return acc

def extract_intermediate_activations(model_path, manifest_path, layer_name=None, save_path="intermediate_activations.npy"):
    model, model_type = load_model_auto(model_path)
    
    

    if model_type == 'keras':
        X, y = load_dataset(manifest_path)
        model.summary()  # useful to see layer names if you don't know them yet

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
    
    else:  # PyTorch model
        # Use forward hook to capture intermediate activations
        X, y = load_dataset_comprehensive(manifest_path)

        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        if layer_name is None:
            # Find last conv or dense layer
            layer_name = None
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    layer_name = name
            print(f"ℹ️  No layer_name provided. Using layer: '{layer_name}'")
        
        save_path = save_path.replace(".npy", f"_{layer_name}.npy")
        
        # Register hook
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(get_activation(layer_name))
                break
        
        # Forward pass
        X_tensor = torch.from_numpy(X).float()
        device = next(model.parameters()).device
        X_tensor = X_tensor.to(device)
        
        with torch.no_grad():
            model(X_tensor)
        
        # Get features from activation
        features = activation[layer_name].cpu().numpy()
        print(f"✅ Extracted features from layer '{layer_name}', shape: {features.shape}")
        

    # Optional: flatten features if needed
    if len(features.shape) > 2:
        features = features.reshape((features.shape[0], -1))
        print(f"Flattened feature shape: {features.shape}")

    # Save to disk for later use
    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(save_path, features)
    np.save(os.path.join(processed_dir, "y_labels"), y)
    np.save(os.path.join(processed_dir, "X_spectograms"), X)
    print(f"✅ Saved labels to y_labels.npy")
    print(f"✅ Saved spectrograms to X_spectograms.npy")
    print(f"💾 Features saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio classifier model")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model without transforms')
    eval_parser.add_argument('--model_path', required=True, help='Path to the model')
    eval_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')
    eval_parser.add_argument('--codec_name', default=None, help='Codec to apply (random, encodec_meta, dac, griffinmel, audiolm, valle)')

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
    extract_parser.add_argument('--save_path', default=os.path.join(DATA_DIR, 'processed/intermediate_activations.npy'), help='Path to save features')

    args = parser.parse_args()

    if args.command == 'evaluate':
        evaluate(args.model_path, args.manifest_path, codec_name=args.codec_name)
    elif args.command == 'evaluate_transform':
        target_shape = (args.freq, args.time)
        evaluate_with_transform(args.model_path, args.manifest_path, args.n_mels, target_shape, args.transform)
    elif args.command == 'extract':
        extract_intermediate_activations(args.model_path, args.manifest_path, args.layer_name, args.save_path)
    else:
        parser.print_help()


