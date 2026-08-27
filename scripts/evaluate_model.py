import os
import numpy as np
import torch
import argparse
import csv
from pathlib import Path
from tqdm import tqdm
from preprocess import load_and_preprocess_audio, audio_to_mel_spectrogram, make_transform_fn
from utils import ROOT_DIR, DATA_DIR
from train_cnn import SimpleCNN
from train_cnn_2d import CNN2D, CNN2D_Legacy
from scipy.ndimage import zoom
from encode_latents import CODECS
from decode_latents_to_audio import extract_mel_spectrogram, pad_or_crop_spectrogram


def is_model_2d_cnn(model):
    """Check if model is a 2D CNN."""
    return isinstance(model, (CNN2D, CNN2D_Legacy))

def preprocess_spectrograms_for_2d_cnn(X):
    """
    Preprocess spectrograms for 2D CNN: resize to (128, 128) and add channel dimension.
    
    Args:
        X: Array of spectrograms with shape (N, 1, 128, T) or (N, 128, T)
    
    Returns:
        Array with shape (N, 1, 128, 128)
    """
    print(f"Preprocessing data for 2D CNN...")
    
    # Remove channel dimension if present
    if X.ndim == 4 and X.shape[1] == 1:
        X = np.squeeze(X, axis=1)  # (N, 1, 128, T) -> (N, 128, T)
    
    X_reshaped = np.zeros((len(X), 128, 128), dtype=np.float32)
    for i, spec in enumerate(tqdm(X, desc="Resizing spectrograms", leave=False)):
        # spec has shape (128, T) where T is variable time steps
        # Resize to (128, 128)
        if spec.ndim == 2:
            # Use zoom to resize to (128, 128)
            zoom_factors = (128 / spec.shape[0], 128 / spec.shape[1])
            spec_zoomed = zoom(spec, zoom_factors, order=1)  # Linear interpolation
            
            # Ensure exact shape after zoom
            if spec_zoomed.shape[0] < 128:
                spec_zoomed = np.pad(spec_zoomed, ((0, 128 - spec_zoomed.shape[0]), (0, 0)), mode='constant')
            elif spec_zoomed.shape[0] > 128:
                spec_zoomed = spec_zoomed[:128, :]
            
            if spec_zoomed.shape[1] < 128:
                spec_zoomed = np.pad(spec_zoomed, ((0, 0), (0, 128 - spec_zoomed.shape[1])), mode='constant')
            elif spec_zoomed.shape[1] > 128:
                spec_zoomed = spec_zoomed[:, :128]
            
            X_reshaped[i] = spec_zoomed
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
    Load PyTorch model from checkpoint file.
    Auto-detects whether it's a 1D CNN or 2D CNN model based on state_dict shape.
    
    Args:
        model_path: Path to model file (.pt for PyTorch)
    
    Returns:
        model: Loaded PyTorch model
    """
    model_path = str(model_path)
    
    if not model_path.endswith('.pt'):
        raise ValueError(f"Unsupported model format: {model_path}. Only .pt (PyTorch) is supported.")
    
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
    return model

def _predict_and_score(model, X, y, device):
    """Run the model over X and return (accuracy, raw predictions)."""
    X_tensor = torch.from_numpy(X).float().to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    # Handle multi-class output: take argmax if preds has shape (N, num_classes), else use threshold
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds_bin = np.argmax(preds, axis=1)
    else:
        preds_bin = (preds.flatten() > 0.5).astype(int)

    acc = np.mean(preds_bin == y)
    return acc, preds

def evaluate(model_path, manifest_path, n_samples=None, sample_rate=16000,
             transform=None, codec_name=None, n_mels=128, target_shape=(128, 128)):
    """
    Evaluate model accuracy on a manifest of audio files.

    Optional flags select what happens to the audio between loading and the mel-spectrogram,
    replacing what used to be three separate functions:
        transform: augmentation to apply before scoring ('pitch_shift', 'time_stretch', 'random')
        codec_name: neural codec (from CODECS) to round-trip the audio through first, following
            the same preprocess -> codec -> mel-spectrogram pipeline used to build the codec-latent
            training set (see encode_latents.py / decode_latents_to_audio.py)
    """
    model = load_model_auto(model_path)
    device = next(model.parameters()).device

    codec = None
    if codec_name is not None:
        codec = CODECS[codec_name](sr=sample_rate)
        if hasattr(codec, 'model'):
            codec.model.eval()

    transform_fn = make_transform_fn(transform)

    manifest_data = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_data.append(row)

    if n_samples is not None:
        manifest_data = manifest_data[:n_samples]

    print(f"Found {len(manifest_data)} samples in manifest")
    print(f"ℹ️  Using sample_rate={sample_rate}Hz for audio loading")

    desc = "Evaluating"
    if codec_name:
        desc += f" via {codec_name}"
    if transform:
        desc += f" with {transform} transform"

    X_list, y_list = [], []
    skipped_count = 0

    for sample_info in tqdm(manifest_data, desc=desc):
        filepath = sample_info.get('filepath')
        label = int(sample_info.get('label', 0))

        if not os.path.exists(filepath):
            skipped_count += 1
            continue

        try:
            # Same pipeline used to build training data; center crop for determinism
            audio = load_and_preprocess_audio(filepath, sr=sample_rate, crop="center", transform_fn=transform_fn)
            if audio is None:
                skipped_count += 1
                continue

            if codec is not None:
                # Mirrors the codec-latent training pipeline's spectrogram computation exactly
                decoded_audio = codec.process_audio(audio)
                spec = extract_mel_spectrogram(decoded_audio, sr=sample_rate, n_mels=n_mels)
                spec = pad_or_crop_spectrogram(spec, target_shape)
            else:
                spec = audio_to_mel_spectrogram(audio, sr=sample_rate, n_mels=n_mels, resize_to=target_shape)[0]

            X_list.append(spec.astype(np.float32))
            y_list.append(label)

        except Exception:
            skipped_count += 1
            continue

    if len(X_list) == 0:
        print("❌ No valid samples processed.")
        return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    X = preprocess_data(X, model)

    acc, preds = _predict_and_score(model, X, y, device)
    print(f"\n✅ Accuracy: {acc*100:.2f}% ({len(X)} samples, skipped {skipped_count}/{len(manifest_data)})")
    print(f"   Mean confidence: {np.mean(preds):.3f}")
    print(f"   Std confidence: {np.std(preds):.3f}")
    return acc

def extract_intermediate_activations(model_path, manifest_path, layer_name=None, save_path=None, sample_rate=16000, device=None):
    """
    Extract intermediate activations from a model layer.
    
    Args:
        model_path: Path to trained model
        manifest_path: CSV manifest with file paths and labels
        layer_name: Name of layer to extract from (auto-detect if None)
        save_path: Path to save features (if None, auto-generate from model name)
        sample_rate: Sample rate for audio loading (IMPORTANT: must match model training)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    model = load_model_auto(model_path)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Extract model name from path if save_path not provided
    if save_path is None:
        model_name = Path(model_path).stem  # Get filename without extension
        model_dir = os.path.join(DATA_DIR, "processed", model_name)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"{model_name}.npy")
    else:
        model_name = Path(save_path).stem
    
    # PyTorch model
    # Load manifest and process samples one-by-one
    
    print("Loading manifest...")
    print(f"ℹ️  Using sample_rate={sample_rate}Hz for audio loading")
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
    
    # Create model directory
    model_dir = os.path.dirname(save_path)
    os.makedirs(model_dir, exist_ok=True)
        
    model_name_from_path = Path(save_path).stem
    labels_path = os.path.join(model_dir, f"{model_name_from_path}_labels.npy")
    specs_path = os.path.join(model_dir, f"{model_name_from_path}_specs.npy")
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(layer_name))
            break
    
    # Pre-allocate memmap files (we'll determine shape from first sample)
    first_pass = True
    memmap_features = None
    memmap_labels = None
    memmap_specs = None
    processed_count = 0
    skipped_count = 0
    
    
    # Process samples one-by-one, writing to disk incrementally
    for sample_idx, sample_info in enumerate(tqdm(manifest_data, desc="Extracting features")):
        filepath = sample_info.get('filepath')
        label = int(sample_info.get('label', 0))
        
        if not os.path.exists(filepath):
            skipped_count += 1
            continue
        
        try:
            # Same pipeline used to build training data; center crop for determinism
            audio = load_and_preprocess_audio(filepath, sr=sample_rate, crop="center")
            if audio is None:
                skipped_count += 1
                continue
            
            spec_resized = audio_to_mel_spectrogram(audio, sr=sample_rate, resize_to=(128, 128))[0]
            
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
                memmap_labels = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=(len(manifest_data),))
                memmap_specs = np.memmap(specs_path, dtype=np.float32, mode='w+', shape=(len(manifest_data), 128, 128))
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
            # Skip files with codec errors or loading issues
            skipped_count += 1
            continue
    
    # Final flush and proper save
    if memmap_features is not None:
        memmap_features.flush()
        memmap_labels.flush()
        memmap_specs.flush()
        
        # Load the memmap data and save as proper .npy files
        features_trimmed = np.array(memmap_features[:processed_count])
        labels_trimmed = np.array(memmap_labels[:processed_count])
        specs_trimmed = np.array(memmap_specs[:processed_count])
        
        # Save as proper numpy files
        np.save(save_path, features_trimmed)
        np.save(labels_path, labels_trimmed)
        np.save(specs_path, specs_trimmed)
        
        del memmap_features, memmap_labels, memmap_specs
        
        # Clean up memmap files (they're temporary and now saved as .npy)
        if os.path.exists(save_path.replace('.npy', '.memmap')):
            os.remove(save_path.replace('.npy', '.memmap'))
        if os.path.exists(labels_path.replace('.npy', '.memmap')):
            os.remove(labels_path.replace('.npy', '.memmap'))
        if os.path.exists(specs_path.replace('.npy', '.memmap')):
            os.remove(specs_path.replace('.npy', '.memmap'))
    
    print(f"✅ Extracted features from layer '{layer_name}' for {processed_count} samples (skipped {skipped_count}/{len(manifest_data)} problematic files)")
    print(f"💾 Features saved to {save_path}")
    print(f"💾 Labels saved to {labels_path}")
    print(f"💾 Specs saved to {specs_path}")


def extract_all_layer_activations(model_path, manifest_path, save_path=None, sample_rate=16000, layer_types=None, device=None):
    """
    Extract intermediate activations from ALL layers in a model.
    
    Args:
        model_path: Path to trained model
        manifest_path: CSV manifest with file paths and labels
        save_path: Directory to save features (if None, auto-generates as data/processed/{model_name})
        sample_rate: Sample rate for audio loading (IMPORTANT: must match model training)
        layer_types: Tuple of layer types to extract from (default: Conv2d and Linear for PyTorch)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Dict mapping layer names to their activation shapes
    """
    model = load_model_auto(model_path)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Extract model name from path if save_path not provided
    if save_path is None:
        model_name = Path(model_path).stem
        save_path = os.path.join(DATA_DIR, "processed", model_name)
    
    os.makedirs(save_path, exist_ok=True)
    model_name = Path(model_path).stem
    
    # PyTorch model
    # Load manifest
    print("Loading manifest...")
    print(f"ℹ️  Using sample_rate={sample_rate}Hz for audio loading")
    manifest_data = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_data.append(row)
    
    print(f"Found {len(manifest_data)} samples in manifest")
    
    # Set default layer types if not provided
    if layer_types is None:
        layer_types = (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)
    
    # Find all layers matching the types
    target_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            target_layers[name] = module
    
    print(f"Found {len(target_layers)} layers to extract from")
    
    # Store activations from all layers
    activations = {}
    
    def get_activation_hook(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks on all target layers
    hooks = []
    for layer_name, module in target_layers.items():
        hook = module.register_forward_hook(get_activation_hook(layer_name))
        hooks.append(hook)
    
    # Pre-allocate memmaps for each layer
    layer_memmaps = {}
    layer_info = {}
    processed_count = 0
    skipped_count = 0
    
    print(f"Processing {len(manifest_data)} samples...")
    
    # First pass: process first sample to determine shapes
    first_pass = True
    
    for sample_idx, sample_info in enumerate(tqdm(manifest_data, desc="Extracting features")):
        filepath = sample_info.get('filepath')
        label = int(sample_info.get('label', 0))
        
        if not os.path.exists(filepath):
            skipped_count += 1
            continue
        
        try:
            # Same pipeline used to build training data; center crop for determinism
            audio = load_and_preprocess_audio(filepath, sr=sample_rate, crop="center")
            if audio is None:
                skipped_count += 1
                continue
            
            spec_resized = audio_to_mel_spectrogram(audio, sr=sample_rate, resize_to=(128, 128))[0]
            
            spec_tensor = torch.from_numpy(spec_resized[np.newaxis, np.newaxis, :, :]).float().to(device)
            
            # Forward pass
            with torch.no_grad():
                model(spec_tensor)
            
            # On first pass, allocate memmaps for all layers
            if first_pass:
                for layer_name, feat in activations.items():
                    feat_shape = (len(manifest_data),) + feat.shape[1:]  # Remove batch dim
                    memmap_path = os.path.join(save_path, f"{model_name}_{layer_name}.memmap")
                    layer_memmaps[layer_name] = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=feat_shape)
                    layer_info[layer_name] = {
                        'shape': feat_shape,
                        'memmap_path': memmap_path,
                        'file_path': os.path.join(save_path, f"{model_name}_{layer_name}.npy")
                    }
                
                # Allocate for labels and specs
                labels_memmap_path = os.path.join(save_path, f"{model_name}_labels.memmap")
                specs_memmap_path = os.path.join(save_path, f"{model_name}_specs.memmap")
                layer_memmaps['labels'] = np.memmap(labels_memmap_path, dtype=np.int32, mode='w+', shape=(len(manifest_data),))
                layer_memmaps['specs'] = np.memmap(specs_memmap_path, dtype=np.float32, mode='w+', shape=(len(manifest_data), 128, 128))
                
                first_pass = False
            
            # Write activations to memmaps
            for layer_name, feat in activations.items():
                layer_memmaps[layer_name][processed_count] = feat[0]  # Remove batch dim
            
            layer_memmaps['labels'][processed_count] = label
            layer_memmaps['specs'][processed_count] = spec_resized
            
            processed_count += 1
            
            # Flush every 50 samples
            if processed_count % 50 == 0:
                for memmap in layer_memmaps.values():
                    memmap.flush()
            
            # Clear GPU memory
            del spec_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            skipped_count += 1
            continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Final flush and save all memmaps as proper numpy files
    for memmap in layer_memmaps.values():
        memmap.flush()
    
    # Convert memmaps to numpy files
    print(f"Saving {len(layer_info)} layers as numpy files...")
    for layer_name, info in layer_info.items():
        memmap = layer_memmaps[layer_name]
        data_trimmed = np.array(memmap[:processed_count])
        np.save(info['file_path'], data_trimmed)
    
    # Save labels and specs
    labels_trimmed = np.array(layer_memmaps['labels'][:processed_count])
    specs_trimmed = np.array(layer_memmaps['specs'][:processed_count])
    np.save(os.path.join(save_path, f"{model_name}_labels.npy"), labels_trimmed)
    np.save(os.path.join(save_path, f"{model_name}_specs.npy"), specs_trimmed)
    
    # Clean up temporary memmap files
    print(f"Cleaning up temporary memmap files...")
    for layer_name, info in layer_info.items():
        memmap_path = info['memmap_path']
        if os.path.exists(memmap_path):
            os.remove(memmap_path)
    
    # Remove labels and specs memmap files
    labels_memmap_path = os.path.join(save_path, f"{model_name}_labels.memmap")
    specs_memmap_path = os.path.join(save_path, f"{model_name}_specs.memmap")
    if os.path.exists(labels_memmap_path):
        os.remove(labels_memmap_path)
    if os.path.exists(specs_memmap_path):
        os.remove(specs_memmap_path)
    
    # Clean up memmaps
    del layer_memmaps
    
    print(f"\n✅ Extracted {len(layer_info)} layers for {processed_count} samples (skipped {skipped_count}/{len(manifest_data)} files)")
    print(f"💾 All features saved to {save_path}/")
    print(f"   Layer files: {model_name}_<layer_name>.npy")
    print(f"   Labels: {model_name}_labels.npy")
    print(f"   Specs: {model_name}_specs.npy")
    
    return layer_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio classifier model")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for evaluate (accuracy scoring, optionally through a transform and/or a codec)
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model accuracy, optionally with a transform and/or a codec')
    eval_parser.add_argument('--model_path', required=True, help='Path to the model')
    eval_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')
    eval_parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to evaluate (default: all)')
    eval_parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio loading')
    eval_parser.add_argument('--transform', default=None, choices=['pitch_shift', 'time_stretch', 'random'], help='Optional augmentation to apply before scoring')
    eval_parser.add_argument('--codec_name', default=None, choices=list(CODECS.keys()), help='Optional neural codec to round-trip audio through before scoring')
    eval_parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bins')
    eval_parser.add_argument('--freq', type=int, default=128, help='Frequency dimension of target shape')
    eval_parser.add_argument('--time', type=int, default=128, help='Time dimension of target shape')

    # Subparser for extract_intermediate_activations
    extract_parser = subparsers.add_parser('extract', help='Extract intermediate activations')
    extract_parser.add_argument('--model_path', required=True, help='Path to the model')
    extract_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')
    extract_parser.add_argument('--layer_name', help='Layer name to extract from')
    extract_parser.add_argument('--save_path', default=None, help='Path to save features (if None, auto-generates as data/processed/{model_name}/{model_name}.npy)')
    extract_parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio loading (IMPORTANT: must match model training)')
    extract_parser.add_argument('--device', default=None, help='Device to use (cuda/cpu, default: auto-detect)')

    # Subparser for extract_all_layer_activations
    extract_all_parser = subparsers.add_parser('extract_all', help='Extract intermediate activations from ALL layers')
    extract_all_parser.add_argument('--model_path', required=True, help='Path to the model')
    extract_all_parser.add_argument('--manifest_path', required=True, help='Path to the manifest CSV')
    extract_all_parser.add_argument('--save_path', default=None, help='Path to save features (if None, auto-generates as data/processed/{model_name})')
    extract_all_parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio loading (IMPORTANT: must match model training)')
    extract_all_parser.add_argument('--device', default=None, help='Device to use (cuda/cpu, default: auto-detect)')

    args = parser.parse_args()

    if args.command == 'evaluate':
        target_shape = (args.freq, args.time)
        evaluate(args.model_path, args.manifest_path, args.n_samples, args.sample_rate,
                 transform=args.transform, codec_name=args.codec_name, n_mels=args.n_mels, target_shape=target_shape)
    elif args.command == 'extract':
        extract_intermediate_activations(args.model_path, args.manifest_path, args.layer_name, args.save_path, args.sample_rate, args.device)
    elif args.command == 'extract_all':
        extract_all_layer_activations(args.model_path, args.manifest_path, args.save_path, args.sample_rate, device=args.device)
    else:
        parser.print_help()


