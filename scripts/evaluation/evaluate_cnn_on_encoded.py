"""
Evaluate CNN2D model on encoded_trainset data.
Converts raw audio to mel-spectrograms and computes accuracy metrics.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
from ..training.train_cnn_2d import CNN2D, CNN2D_Legacy




def load_and_extract_spectrograms(encoded_dir, max_per_class=None):
    """
    Load pre-computed mel-spectrograms from encoded_trainset directory.
    Uses the same spectrograms that were used for training.
    
    Args:
        encoded_dir: Path to encoded_trainset directory (contains pre-computed spectrograms)
        max_per_class: Max samples per class per codec (None for all)
    
    Returns:
        X: Array of shape (N, 1, 128, 128) mel-spectrograms
        y: Array of shape (N,) labels
    """
    all_specs = []
    all_labels = []
    
    # Find all codec subdirectories
    codecs = sorted([d for d in os.listdir(encoded_dir) 
                     if os.path.isdir(os.path.join(encoded_dir, d))])
    
    if not codecs:
        raise ValueError(f"No codec subdirectories found in {encoded_dir}")
    
    print(f"Found codecs: {codecs}\n")
    
    # Load pre-computed spectrograms from each codec
    for codec_name in codecs:
        codec_path = os.path.join(encoded_dir, codec_name)
        print(f"Processing {codec_name}...")
        
        for class_label in range(2):
            class_dir = os.path.join(codec_path, str(class_label))
            if not os.path.exists(class_dir):
                continue
            
            npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            if max_per_class is not None:
                npy_files = npy_files[:max_per_class]
            
            for npy_file in tqdm(npy_files, desc=f"  {codec_name} class {class_label}", leave=False):
                try:
                    spec_path = os.path.join(class_dir, npy_file)
                    # Load pre-computed spectrogram (already normalized and sized to 128x128)
                    mel_spec_db = np.load(spec_path)
                    
                    # Verify shape
                    if mel_spec_db.shape != (128, 128):
                        print(f"Warning: {npy_file} has shape {mel_spec_db.shape}, expected (128, 128)")
                        continue
                    
                    all_specs.append(mel_spec_db)
                    all_labels.append(class_label)
                except Exception as e:
                    print(f"Error processing {npy_file}: {e}")
                    continue
    
    if not all_specs:
        raise ValueError("No spectrograms loaded")
    
    # Stack and add channel dimension: (N, 128, 128) -> (N, 1, 128, 128)
    X = np.array(all_specs, dtype=np.float32)
    X = np.expand_dims(X, axis=1)
    
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\n✅ Loaded {len(X)} pre-computed spectrograms")
    print(f"   X shape: {X.shape} (should be (N, 1, 128, 128))")
    print(f"   y shape: {y.shape}\n")
    
    # Ensure correct shape for CNN2D: (N, 1, 128, 128)
    assert X.shape[1:] == (1, 128, 128), f"Expected shape (N, 1, 128, 128), got {X.shape}"
    
    return X, y


def evaluate_model(model, X, y, device, batch_size=32):
    """
    Evaluate model on data and compute metrics.
    
    Args:
        model: CNN2D model
        X: Input spectrograms (N, 1, 128, 128)
        y: Labels (N,)
        device: torch device
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n📋 Input verification:")
    print(f"   X dtype: {X.dtype}, shape: {X.shape}")
    print(f"   Expected: float32, shape: (N, 1, 128, 128)")
    
    model.eval()
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    print(f"   X_tensor shape after conversion: {X_tensor.shape}")
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating", leave=False):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels
    }


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*70)
    print("📊 CNN2D EVALUATION RESULTS ON ENCODED_TRAINSET")
    print("="*70)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  TN: {cm[0,0]:>6}  FP: {cm[0,1]:>6}")
    print(f"  FN: {cm[1,0]:>6}  TP: {cm[1,1]:>6}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN2D on encoded_trainset")
    parser.add_argument("--encoded_dir", type=str, required=True,
                        help="Path to encoded_trainset directory")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained CNN2D model (.pt)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max", type=int, default=None,
                        help="Max samples per class per codec (None for all)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load pre-computed spectrograms from encoded_trainset
    print(f"📁 Loading pre-computed spectrograms from {args.encoded_dir}")
    X, y = load_and_extract_spectrograms(
        args.encoded_dir,
        max_per_class=args.max
    )
    
    # Load model
    print(f"📂 Loading model from {args.model}")
    model = CNN2D(input_shape=(1, 128, 128), num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print("✅ Model loaded\n")
    
    # Evaluate
    print("🧪 Evaluating model...")
    results = evaluate_model(model, X, y, device, batch_size=args.batch_size)
    
    # Print results
    print_results(results)
    

