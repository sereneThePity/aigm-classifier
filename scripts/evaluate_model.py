import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from preprocess import load_dataset, load_dataset_comprehensive
from utils import ROOT_DIR, DATA_DIR
import tensorflow as tf
from train_cnn import SimpleCNN
from train_cnn_2d import CNN2D, CNN2D_Legacy
from scipy.ndimage import zoom

def is_model_2d_cnn(model):
    """Check if model is a 2D CNN."""
    return isinstance(model, (CNN2D, CNN2D_Legacy))

def preprocess_spectrograms_for_2d_cnn(X):
    """
    Preprocess spectrograms for 2D CNN: resize to (128, 128) and add channel dimension.
    
    Args:
        X: Array of spectrograms with shape (N, 128, T) where T is variable time steps
    
    Returns:
        Array with shape (N, 1, 128, 128)
    """
    print(f"Preprocessing data for 2D CNN...")
    X_reshaped = np.zeros((len(X), 128, 128), dtype=np.float32)
    for i, spec in enumerate(tqdm(X, desc="Resizing spectrograms", leave=False)):
        # spec has shape (128, T) where T is variable time steps
        # Resize to (128, 128)
        if spec.ndim == 2:
            # Use zoom to resize to (128, 128)
            zoom_factors = (128 / spec.shape[0], 128 / spec.shape[1])
            X_reshaped[i] = zoom(spec, zoom_factors, order=1)  # Linear interpolation
        else:
            # Already 2D, just ensure it's 128x128
            X_reshaped[i] = spec
    
    # Add channel dimension: (N, 128, 128) -> (N, 1, 128, 128)
    return np.expand_dims(X_reshaped, axis=1)

def preprocess_data(X, model):
    if is_model_2d_cnn(model):
        return preprocess_spectrograms_for_2d_cnn(X)
    return X

def load_model_auto(model_path):
    """
    Automatically load either PyTorch or Keras model based on file extension.
    Auto-detects whether it's a 1D CNN or 2D CNN model based on state_dict shape.
    
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
            
            input_shape = [128, 128]
            num_classes = 2
            
            if os.path.exists(info_path):
                import json
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    input_shape = info.get('input_shape', [128, 128])
                    num_classes = info.get('num_classes', 2)
            
            # Detect model type from state_dict shape (most reliable method)
            first_conv_weight = loaded_obj.get('conv1.weight')
            if first_conv_weight is not None:
                if first_conv_weight.dim() == 4:  # Conv2d: [out, in, h, w]
                    print(f"Detected 2D CNN model (Conv2d with shape {first_conv_weight.shape})")
                    # Try loading with current CNN2D, fall back to CNN2D_Legacy if it fails
                    try:
                        model = CNN2D(input_shape, num_classes)
                        model.load_state_dict(loaded_obj)
                    except RuntimeError as e:
                        print(f"⚠️  CNN2D load failed, trying legacy version: {str(e)[:100]}...")
                        model = CNN2D_Legacy(input_shape, num_classes)
                        model.load_state_dict(loaded_obj)
                        print(f"✅ Loaded legacy CNN2D model")
                elif first_conv_weight.dim() == 3:  # Conv1d: [out, in, k]
                    print(f"Detected 1D CNN model (Conv1d with shape {first_conv_weight.shape})")
                    model = SimpleCNN(input_shape, num_classes)
                    model.load_state_dict(loaded_obj)
                else:
                    raise ValueError(f"Unknown conv layer dimension: {first_conv_weight.dim()}")
            else:
                raise ValueError("Could not find conv1.weight in checkpoint")
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

def evaluate(model_path, manifest_path):
    model, model_type = load_model_auto(model_path)
    if model_type == 'keras':
        X, y = load_dataset(manifest_path)
    else:
        X, y = load_dataset_comprehensive(manifest_path, num_samples=1000, num_workers=10)
        X = preprocess_data(X, model)
    
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
        # Load manifest and process samples one-by-one
        import csv
        import librosa
        
        print("Loading manifest...")
        manifest_data = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_data.append(row)
        
        print(f"Found {len(manifest_data)} samples in manifest")
        
        activation = {} 
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cpu().numpy()
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
        
        device = next(model.parameters()).device
        
        # Pre-allocate memmap files (we'll determine shape from first sample)
        first_pass = True
        memmap_features = None
        memmap_labels = None
        memmap_specs = None
        processed_count = 0
        
        print(f"Processing {len(manifest_data)} samples...")
        
        # Process samples one-by-one, writing to disk incrementally
        for sample_idx, sample_info in enumerate(tqdm(manifest_data, desc="Extracting features")):
            filepath = sample_info.get('filepath')
            label = int(sample_info.get('label', 0))
            
            if not os.path.exists(filepath):
                continue
            
            try:
                # Load and process single audio file
                audio, sr = librosa.load(filepath, sr=44100, mono=True)
                spec = librosa.feature.melspectrogram(y=audio, sr=44100, n_mels=128)
                
                # Resize to (128, 128)
                zoom_factors = (128 / spec.shape[0], 128 / spec.shape[1])
                spec_resized = zoom(spec, zoom_factors, order=1).astype(np.float32)
                
                # Add batch and channel dims: (1, 1, 128, 128)
                spec_tensor = torch.from_numpy(spec_resized[np.newaxis, np.newaxis, :, :]).float().to(device)
                
                # Forward pass and extract features
                with torch.no_grad():
                    model(spec_tensor)
                
                # Get features from activation
                feat = activation[layer_name][0]  # Remove batch dim
                
                # On first pass, allocate memmaps
                if first_pass:
                    feat_shape = (len(manifest_data),) + feat.shape
                    memmap_features = np.memmap(save_path, dtype=np.float32, mode='w+', shape=feat_shape)
                    memmap_labels = np.memmap(save_path.replace('.npy', '_labels.npy'), dtype=np.int32, mode='w+', shape=(len(manifest_data),))
                    memmap_specs = np.memmap(save_path.replace('.npy', '_specs.npy'), dtype=np.float32, mode='w+', shape=(len(manifest_data), 128, 128))
                    first_pass = False
                
                # Write to memmap
                memmap_features[processed_count] = feat
                memmap_labels[processed_count] = label
                memmap_specs[processed_count] = spec_resized
                
                processed_count += 1
                
                # Clear GPU memory
                del spec_tensor, feat
                torch.cuda.empty_cache()
                
                # Flush to disk every 50 samples
                if processed_count % 50 == 0:
                    memmap_features.flush()
                    memmap_labels.flush()
                    memmap_specs.flush()
                    
            except Exception as e:
                continue
        
        # Final flush and proper save
        if memmap_features is not None:
            memmap_features.flush()
            memmap_labels.flush()
            memmap_specs.flush()
            
            # Trim to actual processed count and save as proper numpy files
            processed_dir = os.path.join(DATA_DIR, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Load the memmap data and save as proper .npy files
            features_trimmed = np.array(memmap_features[:processed_count])
            labels_trimmed = np.array(memmap_labels[:processed_count])
            specs_trimmed = np.array(memmap_specs[:processed_count])
            
            # Save as proper numpy files
            np.save(save_path, features_trimmed)
            np.save(save_path.replace('.npy', '_labels.npy'), labels_trimmed)
            np.save(save_path.replace('.npy', '_specs.npy'), specs_trimmed)
            
            del memmap_features, memmap_labels, memmap_specs
        
        print(f"✅ Extracted features from layer '{layer_name}' for {processed_count} samples")
        print(f"💾 Features saved to {save_path}")
        print(f"💾 Labels saved to {save_path.replace('.npy', '_labels.npy')}")
        print(f"💾 Specs saved to {save_path.replace('.npy', '_specs.npy')}")


    # Only apply flattening for Keras models (PyTorch model already saved via memmap above)
    if model_type == 'keras':
        # Optional: flatten features if needed
        if len(features.shape) > 2:
            # Save spatial features before flattening (useful for SAE training)
            spatial_path = save_path.replace(".npy", "_spatial.npy")
            np.save(spatial_path, features)
            print(f"✅ Saved spatial features to {spatial_path}")

        # Save to disk for later use
        processed_dir = os.path.join(DATA_DIR, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        np.save(save_path, features)
        np.save(os.path.join(processed_dir, "y_labels_legacy"), y)
        np.save(os.path.join(processed_dir, "X_spectograms_legacy"), X)
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
        evaluate(args.model_path, args.manifest_path)
    elif args.command == 'evaluate_transform':
        target_shape = (args.freq, args.time)
        evaluate_with_transform(args.model_path, args.manifest_path, args.n_mels, target_shape, args.transform)
    elif args.command == 'extract':
        extract_intermediate_activations(args.model_path, args.manifest_path, args.layer_name, args.save_path)
    else:
        parser.print_help()


