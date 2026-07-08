#!/usr/bin/env python3
"""
Example: Using Neural Codec Confounders in Your AIGM Classifier Pipeline

This script demonstrates different ways to integrate neural codec confounders
into your preprocessing and training workflow.
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import json

# Import your preprocessing utilities
from preprocess import load_dataset_comprehensive, load_and_prep_audio
from neural_codec_confounders import NeuralCodecConfounder, get_available_codecs
from utils import normalize_audio


def example_1_check_available_codecs():
    """
    Example 1: List and check available codecs
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Checking Available Codec Confounders")
    print("="*70)
    
    confounder = NeuralCodecConfounder(sr=22500)
    available = confounder.get_available_codecs()
    
    print(f"\n✅ Available codecs: {available}")
    print(f"\nTotal: {len(available)} codec(s) ready to use")
    
    # Show codec info
    codec_descriptions = {
        "griffinmel": "Griffin-Lim mel-spectrogram codec (lightweight)",
        "audiolm": "Google AudioLM codec with spectral quantization",
        "valle": "Microsoft VALL-E hierarchical quantization codec",
        "encodec_meta": "Facebook EnCodec neural codec",
        "dac": "Descript Audio Codec (high-quality)"
    }
    
    print("\nCodec Details:")
    for codec in available:
        print(f"  • {codec:20s} - {codec_descriptions.get(codec, 'Custom codec')}")


def example_2_apply_codec_to_single_file():
    """
    Example 2: Apply a codec to a single audio file
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Apply Codec to Single Audio File")
    print("="*70)
    
    # For this example, we'll create a synthetic audio sample
    sr = 22500
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"\nCreated synthetic audio: {audio.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration} seconds")
    
    # Apply codec
    confounder = NeuralCodecConfounder(sr=sr)
    
    codec_name = "encodec_meta"  
    print(f"\nApplying codec: {codec_name}")
    
    processed = confounder.apply_codec(audio, codec_name)
    
    if processed is not None:
        print(f"✅ Successfully applied {codec_name}")
        print(f"   Output shape: {processed.shape}")
        print(f"   Length change: {processed.shape[0] - audio.shape[0]:+d} samples")
        
        # Normalize to consistent amplitude (linear RMS normalization)
        normalized = normalize_audio(processed, method='linear', target=0.1)
        
        # Calculate metrics
        original_rms = np.sqrt(np.mean(audio ** 2))
        processed_rms = np.sqrt(np.mean(processed ** 2))
        normalized_rms = np.sqrt(np.mean(normalized ** 2))
        original_peak = np.max(np.abs(audio))
        processed_peak = np.max(np.abs(processed))
        normalized_peak = np.max(np.abs(normalized))
        
        # Energy ratio instead of SNR (works even with different lengths)
        original_energy = np.sum(audio ** 2)
        processed_energy = np.sum(processed ** 2)
        normalized_energy = np.sum(normalized ** 2)
        energy_ratio_db = 10 * np.log10((original_energy + 1e-10) / (processed_energy + 1e-10))
        
        print(f"\n   Before normalization:")
        print(f"   - RMS: {processed_rms:.6f}")
        print(f"   - Peak: {processed_peak:.6f}")
        print(f"\n   After normalize_audio():")
        print(f"   - RMS: {normalized_rms:.6f}")
        print(f"   - Peak: {normalized_peak:.6f}")
        print(f"   - Energy ratio: {energy_ratio_db:.2f} dB")
    else:
        print(f"❌ Failed to apply {codec_name}")


def example_3_apply_all_codecs_comparison():
    """
    Example 3: Apply all available codecs and compare
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Compare All Available Codecs")
    print("="*70)
    
    # Create sample audio
    sr = 22500
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a more musical signal (multiple frequencies)
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 550 * t) +  # C#5
        0.1 * np.sin(2 * np.pi * 660 * t)    # E5
    ).astype(np.float32)
    
    print(f"\nCreated polyphonic test audio: {audio.shape}")
    
    confounder = NeuralCodecConfounder(sr=sr)
    available = confounder.get_available_codecs()
    
    print(f"\nApplying {len(available)} codecs...")
    print("-" * 70)
    print(f"{'Codec':<20} {'Status':<15} {'Peak Change':<15}")
    print("-" * 70)
    
    results = {}
    for codec_name in available:
        processed = confounder.apply_codec(audio, codec_name)
        
        if processed is not None:
            # Normalize audio to consistent RMS level (linear method)
            normalized = normalize_audio(processed, method='linear', target=0.1)
            
            # Calculate metrics
            peak_original = np.max(np.abs(audio))
            peak_normalized = np.max(np.abs(normalized))
            peak_change = (peak_normalized - peak_original) / (peak_original + 1e-10) * 100
            
            results[codec_name] = {
                "status": "✅ OK",
                "peak_change_percent": peak_change
            }
            
            print(f"{codec_name:<20} {'✅ OK':<15} {peak_change:>+6.2f}%")
        else:
            results[codec_name] = {"status": "❌ Failed"}
            print(f"{codec_name:<20} {'❌ Failed':<15} {'N/A':<15}")
    
    print("-" * 70)
    print(f"\n✅ Comparison complete. {len([r for r in results.values() if 'OK' in r['status']])}/{len(available)} codecs available")


def example_4_load_with_codec_confounder():
    """
    Example 4: Load dataset with codec confounder applied during preprocessing
    
    Note: This example requires actual audio files and a manifest CSV.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Load Dataset with Codec Confounder")
    print("="*70)
    
    print("""
This example loads an audio dataset with a specified codec applied during preprocessing.

Usage:
    
    # Load data WITHOUT codec confounder
    X_original, y = load_dataset_comprehensive(
        manifest_csv="data/test/manifest.csv",
        codec_name=None,
        workers=20
    )
    
    # Load data WITH codec confounder
    X_with_codec, y = load_dataset_comprehensive(
        manifest_csv="data/test/manifest.csv",
        codec_name="griffinmel",
        workers=20
    )
    
    # Compare shapes
    print(f"Original: {X_original.shape}")
    print(f"With codec: {X_with_codec.shape}")
    """)


def example_5_create_augmented_training_set():
    """
    Example 5: Create augmented training set with codec confounders
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Create Augmented Training Set")
    print("="*70)
    
    print("""
To improve model robustness, create training data with codec variations:

Step 1: Apply codecs to your real audio training data
    
    python scripts/apply_codecs.py \\
        --input-dir data/train/real_songs/ \\
        --codec griffinmel \\
        --output-dir data/train/real_songs_griffinmel/
    
    python scripts/apply_codecs.py \\
        --input-dir data/train/real_songs/ \\
        --codec audiolm \\
        --output-dir data/train/real_songs_audiolm/

Step 2: Update your manifest to include all variations

    original_manifest = pd.read_csv("data/train/real_songs/manifest.csv")
    codec_manifest = original_manifest.copy()
    codec_manifest["filepath"] = codec_manifest["filepath"].str.replace(
        "real_songs/", "real_songs_griffinmel/"
    )
    
    # Combine
    combined = pd.concat([original_manifest, codec_manifest], ignore_index=True)
    combined.to_csv("data/train/manifest_augmented.csv", index=False)

Step 3: Train on augmented dataset

    X_train, y_train = load_dataset_comprehensive(
        manifest_csv="data/train/manifest_augmented.csv",
        workers=20
    )
    
    # Train your model with X_train and y_train
    # This improves robustness to codec artifacts
    """)


def example_6_evaluate_codec_robustness():
    """
    Example 6: Evaluate model robustness to different codecs
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Evaluate Model Robustness to Codecs")
    print("="*70)
    
    print("""
Test how your trained model performs on audio processed with different codecs:

    from preprocess import load_dataset_comprehensive
    from train_cnn import load_model  # Your model loading function
    import numpy as np
    
    # Load trained model
    model = load_model("models/audio_classifier_model.keras")
    
    # Test on different codecs
    codecs_to_test = ["griffinmel", "audiolm", "valle", "encodec_meta"]
    
    results = {}
    for codec in codecs_to_test:
        print(f"\\nTesting with codec: {codec}")
        
        # Load test set with codec applied
        X_test, y_test = load_dataset_comprehensive(
            manifest_csv="data/test/manifest.csv",
            codec_name=codec,
            workers=20
        )
        
        # Evaluate
        predictions = model.predict(X_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_labels == y_test)
        
        results[codec] = accuracy
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Print summary
    print("\\n" + "="*50)
    print("Robustness Summary:")
    for codec, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {codec:20s}: {acc:.4f}")
    """)


def example_7_custom_confounder_logic():
    """
    Example 7: Using codecs in custom preprocessing logic
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Custom Confounder Logic")
    print("="*70)
    
    print("""
Use codecs in custom preprocessing pipelines:

    from neural_codec_confounders import NeuralCodecConfounder
    import librosa
    
    class AudioProcessor:
        def __init__(self, sr=22500):
            self.sr = sr
            self.confounder = NeuralCodecConfounder(sr=sr)
        
        def process_with_random_codec(self, filepath):
            '''Process audio with a random codec'''
            audio, sr = librosa.load(filepath, sr=self.sr)
            
            # Apply random codec
            processed, codec_used = self.confounder.apply_random_codec(audio)
            
            return processed, codec_used
        
        def process_with_all_codecs(self, filepath):
            '''Process audio with all available codecs'''
            audio, sr = librosa.load(filepath, sr=self.sr)
            
            results = self.confounder.apply_all_codecs(audio)
            
            return results
    
    # Usage
    processor = AudioProcessor()
    
    # Random codec
    audio, codec = processor.process_with_random_codec("audio.wav")
    print(f"Applied: {codec}")
    
    # All codecs
    all_results = processor.process_with_all_codecs("audio.wav")
    for codec_name, processed_audio in all_results.items():
        print(f"{codec_name}: {processed_audio.shape}")
    """)


def main():
    """Run all examples"""
    
    print("\n" * 2)
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█  NEURAL CODEC CONFOUNDERS: Usage Examples".ljust(69) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Run examples
    # example_1_check_available_codecs()
    # example_2_apply_codec_to_single_file()
    example_3_apply_all_codecs_comparison()
    # example_4_load_with_codec_confounder()
    # example_5_create_augmented_training_set()
    # example_6_evaluate_codec_robustness()
    # example_7_custom_confounder_logic()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
