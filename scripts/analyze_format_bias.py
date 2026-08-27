"""
Analyze if model is format-biased (MP3 detector) rather than AIGM detector.

Tests:
1. Performance on real MP3s vs real WAVs (should be similar)
2. Performance on fake MP3s vs fake WAVs (should be similar)
3. Cross-format evaluation (train on MP3, test on WAV, etc.)
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from preprocess import load_all_from_manifest, load_and_preprocess_audio, audio_to_mel_spectrogram
from utils import ROOT_DIR, DATA_DIR
from evaluate_model import load_model_auto, preprocess_data
from train_cnn import SimpleCNN
from train_cnn_2d import CNN2D


def evaluate_subset(model, dataloader, device):
    """Evaluate model on a subset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def load_manifest_with_format(manifest_path):
    """Load manifest and filter by format."""
    df = pd.read_csv(manifest_path)
    # Extract format from file extension
    df['file_format'] = df['filepath'].str.split('.').str[-1].str.lower()
    return df


def get_metrics(preds, labels):
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'cm': confusion_matrix(labels, preds).tolist()
    }


def analyze_format_bias(model_path, testset_manifest, output_path=None, sample_rate=16000):
    """
    Analyze if model is format-biased.
    
    Args:
        model_path: Path to trained model
        testset_manifest: Path to testset manifest CSV
        output_path: Where to save results (default: alongside model)
        sample_rate: Sample rate for audio loading
    """
    if output_path is None:
        output_path = str(Path(model_path).parent / "format_bias_analysis.json")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}")
    model = load_model_auto(model_path)
    model.to(device)
    model.eval()
    
    # Load test manifest with format info
    print(f"Loading manifest from {testset_manifest}")
    df = load_manifest_with_format(testset_manifest)
    
    print(f"\nDataset breakdown:")
    print(f"Total samples: {len(df)}")
    print(f"\nBy format:")
    print(df['file_format'].value_counts())
    print(f"\nBy label:")
    print(df['label'].value_counts())
    print(f"\nBy format and label:")
    print(pd.crosstab(df['label'], df['file_format']))
    
    results = {
        'dataset_info': {
            'total_samples': len(df),
            'label_counts': df['label'].value_counts().to_dict(),
            'format_counts': df['file_format'].value_counts().to_dict()
        },
        'subset_evaluations': {}
    }
    
    # Process ALL data from raw audio (same preprocessing as evaluate_model.py)
    print(f"\n{'='*60}")
    print(f"Processing ALL data from raw audio (sample_rate={sample_rate}Hz)...")
    print(f"{'='*60}")
    
    X_list = []
    y_list = []
    filepath_list = []
    format_list = []
    skipped_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading and preprocessing"):
        filepath = row['filepath']
        label = int(row['label'])
        file_format = row['file_format']
        
        if not os.path.exists(filepath):
            skipped_count += 1
            continue
        
        try:
            # Same pipeline used to build training data; center crop for determinism
            audio = load_and_preprocess_audio(filepath, sr=sample_rate, crop="center")
            if audio is None:
                skipped_count += 1
                continue
            
            spec_resized = audio_to_mel_spectrogram(audio, sr=sample_rate, resize_to=(128, 128))
            
            X_list.append(spec_resized[0])
            y_list.append(label)
            filepath_list.append(filepath)
            format_list.append(file_format)
            
        except Exception:
            skipped_count += 1
            continue
    
    if len(X_list) == 0:
        print("❌ No valid samples processed")
        return results
    
    print(f"Loaded {len(X_list)} samples (skipped {skipped_count}/{len(df)})")
    
    # Stack data and add channel dimension: (N, 128, 128) -> (N, 1, 128, 128)
    X = np.array(X_list, dtype=np.float32)
    X = np.expand_dims(X, axis=1)
    y = np.array(y_list, dtype=np.int64)
    
    # Convert to tensors once
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    # Create updated dataframe with only successfully loaded samples
    df_loaded = pd.DataFrame({
        'filepath': filepath_list,
        'label': y_list,
        'file_format': format_list
    })
    
    # Define subsets
    subsets = [
        ('all', None, None),
        ('real_only', 0, None),
        ('fake_only', 1, None),
        ('mp3_only', None, 'mp3'),
        ('wav_only', None, 'wav'),
        ('real_mp3', 0, 'mp3'),
        ('real_wav', 0, 'wav'),
        ('fake_mp3', 1, 'mp3'),
        ('fake_wav', 1, 'wav'),
    ]
    
    # Extract indices for each subset from loaded dataframe
    print(f"\n{'='*60}")
    print("Evaluating subsets (using preloaded data)...")
    print(f"{'='*60}")
    
    for subset_name, label_filter, format_filter in subsets:
        # Get indices for this subset
        subset_mask = np.ones(len(df_loaded), dtype=bool)
        
        if label_filter is not None:
            subset_mask &= df_loaded['label'].values == label_filter
        if format_filter is not None:
            subset_mask &= df_loaded['file_format'].values == format_filter
        
        subset_indices = np.where(subset_mask)[0]
        
        if len(subset_indices) == 0:
            print(f"\n⚠️  Subset '{subset_name}': 0 samples, skipping")
            results['subset_evaluations'][subset_name] = None
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating on: {subset_name}")
        subset_df = df_loaded.iloc[subset_indices]
        print(f"  Samples: {len(subset_df)}")
        print(f"  Label distribution: {subset_df['label'].value_counts().to_dict()}")
        print(f"  Format distribution: {subset_df['file_format'].value_counts().to_dict()}")
        
        # Extract subset tensors (already preprocessed)
        X_subset = X_tensor[subset_indices]
        y_subset = y_tensor[subset_indices]
        
        # Create dataloader
        batch_size = 32
        dataloader = DataLoader(
            list(zip(X_subset, y_subset)),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Evaluate
        preds, labels = evaluate_subset(model, dataloader, device)
        metrics = get_metrics(preds, labels)
        
        # Convert metrics to JSON-serializable format
        metrics_json = {}
        for k, v in metrics.items():
            if k == 'cm':
                metrics_json[k] = v  # Already a list
            else:
                metrics_json[k] = float(v)
        
        results['subset_evaluations'][subset_name] = {
            'n_samples': len(subset_df),
            'metrics': metrics_json
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Confusion Matrix:\n{np.array(metrics['cm'])}")
    
    # Analysis summary
    print(f"\n{'='*60}")
    print("FORMAT BIAS ANALYSIS")
    print(f"{'='*60}")
    
    real_mp3_result = results['subset_evaluations'].get('real_mp3')
    real_wav_result = results['subset_evaluations'].get('real_wav')
    fake_mp3_result = results['subset_evaluations'].get('fake_mp3')
    fake_wav_result = results['subset_evaluations'].get('fake_wav')
    
    real_mp3_acc = real_mp3_result['metrics']['accuracy'] if real_mp3_result is not None else None
    real_wav_acc = real_wav_result['metrics']['accuracy'] if real_wav_result is not None else None
    fake_mp3_acc = fake_mp3_result['metrics']['accuracy'] if fake_mp3_result is not None else None
    fake_wav_acc = fake_wav_result['metrics']['accuracy'] if fake_wav_result is not None else None
    
    if real_mp3_acc is not None and real_wav_acc is not None:
        real_diff = abs(real_mp3_acc - real_wav_acc)
        print(f"\nReal class performance difference (MP3 vs WAV):")
        print(f"  Real MP3: {real_mp3_acc:.4f}")
        print(f"  Real WAV: {real_wav_acc:.4f}")
        print(f"  Difference: {real_diff:.4f}")
        if real_diff > 0.10:
            print(f"  ⚠️  SIGNIFICANT DIFFERENCE - Likely format bias!")
        else:
            print(f"  ✓ Similar performance - Format-agnostic")
    else:
        print(f"\n⚠️  Real MP3 or Real WAV subset missing, skipping real class analysis")
    
    if fake_mp3_acc is not None and fake_wav_acc is not None:
        fake_diff = abs(fake_mp3_acc - fake_wav_acc)
        print(f"\nFake class performance difference (MP3 vs WAV):")
        print(f"  Fake MP3: {fake_mp3_acc:.4f}")
        print(f"  Fake WAV: {fake_wav_acc:.4f}")
        print(f"  Difference: {fake_diff:.4f}")
        if fake_diff > 0.10:
            print(f"  ⚠️  SIGNIFICANT DIFFERENCE - Likely format bias!")
        else:
            print(f"  ✓ Similar performance - Format-agnostic")
    else:
        print(f"\n⚠️  Fake MP3 or Fake WAV subset missing, skipping fake class analysis")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze format bias in trained model")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--testset', type=str, 
                       default=os.path.join(DATA_DIR, 'testset', 'manifest.csv'),
                       help='Path to testset manifest')
    parser.add_argument('--output', type=str, default=None, help='Output analysis path')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio loading')
    
    args = parser.parse_args()
    
    analyze_format_bias(args.model, args.testset, args.output, args.sample_rate)
