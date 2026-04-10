"""Integrate topK SAE with CNN for enhanced feature representation.

This module provides utilities to:
1. Hook a trained topK SAE into the CNN
2. Extract enhanced features using SAE representations
3. Create an end-to-end SAE+CNN model for classification
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List, Dict
from einops import rearrange
import joblib

from train_cnn import SimpleCNN
from train_topk_sae import TopKSAE, extract_patches
from utils import ROOT_DIR, DATA_DIR


class SAEFeatureExtractor:
    """Extract and reconstruct features using trained topK SAE."""
    
    def __init__(self, sae_model_dir, device='cpu'):
        """
        Initialize SAE feature extractor.
        
        Args:
            sae_model_dir: Path to directory containing saved SAE files
            device: Device to load model on
        """
        self.device = torch.device(device)
        
        # Load config
        config_path = os.path.join(sae_model_dir, "topk_sae_config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"📋 SAE Config loaded:")
        print(f"   Input shape: {self.config['input_shape']}")
        print(f"   Nb concepts: {self.config['nb_concepts']}")
        print(f"   Top-k: {self.config['top_k'] if self.config['top_k'] else 'All'}")
        print(f"   Patch size: {self.config['patch_size']}")
        
        # Initialize SAE model
        self.sae = TopKSAE(
            input_shape=self.config['input_shape'],
            nb_concepts=self.config['nb_concepts'],
            top_k=self.config['top_k'],
            device=device
        )
        
        # Load weights
        weights_path = os.path.join(sae_model_dir, "topk_sae_weights.pt")
        self.sae.load_state_dict(torch.load(weights_path, map_location=device))
        self.sae.eval()
        print(f"   ✓ Weights loaded from {weights_path}")
        
        # Load scaler
        scaler_path = os.path.join(sae_model_dir, "patch_scaler.joblib")
        self.scaler = joblib.load(scaler_path)
        print(f"   ✓ Scaler loaded from {scaler_path}")
    
    def extract_and_enhance_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Process a single spectrogram through SAE for enhanced representation.
        
        Args:
            spectrogram: (H, W) spectrogram
        
        Returns:
            enhanced: (H, W) reconstructed spectrogram with SAE enhancement
        """
        patch_size = self.config['patch_size']
        
        # Extract patches
        spec_batch = spectrogram[np.newaxis, :, :]  # Add batch dimension
        patches, dims = extract_patches(spec_batch, (patch_size, patch_size))
        num_h, num_w, patch_dim = dims
        
        # Normalize patches
        patches_scaled = self.scaler.transform(patches)
        patches_tensor = torch.from_numpy(patches_scaled).float().to(self.device)
        
        # Process through SAE
        with torch.no_grad():
            recon_tensor, codes = self.sae.forward(patches_tensor)
        
        # Convert back to numpy
        recon = recon_tensor.cpu().numpy()
        
        # Inverse transform scaling
        recon_original = self.scaler.inverse_transform(recon)
        
        # Reconstruct patches back to spectrogram
        H, W = spectrogram.shape
        enhanced = rearrange(
            recon_original,
            '(nh nw) (ph pw) -> (nh ph) (nw pw)',
            nh=num_h, nw=num_w,
            ph=patch_size, pw=patch_size
        )
        
        # Ensure shape matches original
        enhanced = enhanced[:H, :W]
        
        return enhanced
    
    def get_sparse_codes(self, spectrogram: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get sparse codes from SAE for a spectrogram.
        
        Args:
            spectrogram: (H, W) spectrogram
        
        Returns:
            codes: Sparse codes tensor
            codes_np: Codes as numpy array
        """
        patch_size = self.config['patch_size']
        
        # Extract patches
        spec_batch = spectrogram[np.newaxis, :, :]
        patches, dims = extract_patches(spec_batch, (patch_size, patch_size))
        
        # Normalize patches
        patches_scaled = self.scaler.transform(patches)
        patches_tensor = torch.from_numpy(patches_scaled).float().to(self.device)
        
        # Get codes
        with torch.no_grad():
            codes = self.sae.encode(patches_tensor)
        
        return codes, codes.cpu().numpy()


class SAEEnhancedCNN(nn.Module):
    """CNN with SAE enhancement - processes spectrograms through SAE first."""
    
    def __init__(self, base_cnn: SimpleCNN, sae_extractor: SAEFeatureExtractor):
        """
        Initialize SAE-enhanced CNN.
        
        Args:
            base_cnn: Trained SimpleCNN model
            sae_extractor: SAEFeatureExtractor instance
        """
        super(SAEEnhancedCNN, self).__init__()
        self.base_cnn = base_cnn
        self.sae_extractor = sae_extractor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: enhance spectrograms with SAE, then classify.
        
        Args:
            x: (B, H, W) spectrograms
        
        Returns:
            logits: Class predictions
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Process each spectrogram through SAE
        enhanced = []
        for i in range(batch_size):
            spec = x[i].cpu().numpy()
            # Normalize to [0, 1] if needed
            if spec.max() > 1:
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-7)
            
            enhanced_spec = self.sae_extractor.extract_and_enhance_spectrogram(spec)
            enhanced.append(enhanced_spec)
        
        # Stack and convert back to tensor
        enhanced_tensor = torch.from_numpy(np.stack(enhanced)).float().to(device)
        
        # Pass through CNN
        return self.base_cnn(enhanced_tensor)


class IntegratedSAECNN(nn.Module):
    """Alternative: SAE codes as intermediate features for CNN."""
    
    def __init__(self, cnn_model: SimpleCNN, sae_extractor: SAEFeatureExtractor,
                 code_projection_dim: int = 256):
        """
        Initialize integrated SAE-CNN that uses SAE codes as features.
        
        Args:
            cnn_model: Base CNN model
            sae_extractor: SAE feature extractor
            code_projection_dim: Dimension for projecting SAE codes
        """
        super(IntegratedSAECNN, self).__init__()
        self.sae_extractor = sae_extractor
        
        # Code projection layer: from SAE codes to features
        num_codes = (128 // sae_extractor.config['patch_size']) ** 2
        code_dim = sae_extractor.config['nb_concepts']
        
        self.code_projection = nn.Sequential(
            nn.Linear(code_dim, code_projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(code_projection_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification (real/fake)
        )
        
        self.num_codes = num_codes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: extract SAE codes -> project -> classify.
        
        Args:
            x: (B, H, W) spectrograms
        
        Returns:
            logits: (B, 2) class predictions
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Extract SAE codes for all spectrograms
        all_codes = []
        for i in range(batch_size):
            spec = x[i].cpu().numpy()
            # Normalize to [0, 1]
            if spec.max() > 1:
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-7)
            
            codes, _ = self.sae_extractor.get_sparse_codes(spec)
            all_codes.append(codes)
        
        # Stack codes: (B, num_patches, hidden_dim)
        codes_stacked = torch.stack(all_codes).to(device)  # (B, num_patches, hidden_dim)
        
        # Average pool over patches to get single feature vector per sample
        codes_pooled = codes_stacked.mean(dim=1)  # (B, hidden_dim)
        
        # Project codes
        code_features = self.code_projection(codes_pooled)  # (B, code_projection_dim)
        
        # Classify
        logits = self.classifier(code_features)
        
        return logits


def load_training_data_with_sae(
    manifest_path: str,
    sae_extractor: SAEFeatureExtractor,
    n_mels: int = 128,
    target_shape: Tuple = (128, 128),
    segment_duration: float = 5.0,
    enhance: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data and optionally enhance with SAE.
    
    Args:
        manifest_path: Path to manifest CSV
        sae_extractor: SAEFeatureExtractor instance
        n_mels: Number of mel bins
        target_shape: Target spectrogram shape
        segment_duration: Audio segment duration
        enhance: Whether to enhance with SAE
    
    Returns:
        X: Enhanced spectrograms
        y: Labels
    """
    # Import here to avoid circular imports
    from preprocess import load_dataset_comprehensive
    
    print(f"📂 Loading dataset from {manifest_path}")
    X, y = load_dataset_comprehensive(
        manifest_path,
        n_mels=n_mels,
        target_shape=target_shape,
        segment_duration=segment_duration,
        target_loudness=-20.0,
        hp_freq=20,
        workers=0,  # CPU only
        codec_name=None
    )
    
    if enhance:
        print(f"\n🧠 Enhancing spectrograms with SAE...")
        X_enhanced = np.zeros_like(X)
        
        for i in range(len(X)):
            spec = X[i]
            # Normalize
            if spec.max() > 1:
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-7)
            
            X_enhanced[i] = sae_extractor.extract_and_enhance_spectrogram(spec)
            
            if (i + 1) % max(1, len(X) // 5) == 0:
                print(f"   Progress: {i+1}/{len(X)}")
        
        X = X_enhanced
    
    return X, y


# ============= Example Usage =============

if __name__ == '__main__':
    print("=" * 70)
    print("🔗 SAE + CNN Integration Example")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # 1. Load SAE
    print(f"\n🧠 Loading SAE...")
    sae_model_dir = os.path.join(ROOT_DIR, "models", "topk_sae")
    sae_extractor = SAEFeatureExtractor(sae_model_dir, device=device)
    
    # 2. Load CNN
    print(f"\n🖧 Loading CNN...")
    cnn_path = os.path.join(ROOT_DIR, "models", "audio_classifier_model.keras")
    if os.path.exists(cnn_path):
        print(f"   CNN found at {cnn_path}")
        # Note: You'll need to convert the Keras model to PyTorch
        # or load a PyTorch version if you have one
    else:
        print(f"   CNN not found at {cnn_path}")
    
    # 3. Example: Enhance a single spectrogram
    print(f"\n📐 Example: Enhancing a single activation...")
    data_dir = os.path.join(DATA_DIR, "processed")
    all_activations = np.load(os.path.join(data_dir, "intermediate_activations_conv2.npy"))
    
    if all_activations.ndim == 4 and all_activations.shape[-1] == 1:
        all_activations = np.squeeze(all_activations, axis=-1)
    
    test_activation = all_activations[0]
    
    # Normalize
    test_activation_norm = (test_activation - test_activation.min()) / (test_activation.max() - test_activation.min() + 1e-7)
    
    # Enhance
    enhanced_activation = sae_extractor.extract_and_enhance_spectrogram(test_activation_norm)
    
    print(f"   Original shape: {test_activation_norm.shape}")
    print(f"   Enhanced shape: {enhanced_activation.shape}")
    print(f"   Original range: [{test_activation_norm.min():.4f}, {test_activation_norm.max():.4f}]")
    print(f"   Enhanced range: [{enhanced_activation.min():.4f}, {enhanced_activation.max():.4f}]")
    
    print("\n" + "=" * 70)
    print("✅ Integration setup complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Train CNN on SAE-enhanced spectrograms")
    print("   2. Use SAEEnhancedCNN for end-to-end training")
    print("   3. Compare IntegratedSAECNN variant using code features")
