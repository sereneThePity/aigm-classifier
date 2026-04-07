"""Train a topK Sparse Autoencoder (SAE) on spectrograms using the overcomplete library.

CPU-compatible training for learning sparse representations of audio spectrograms.
The trained SAE can be hooked into the CNN for feature enhancement.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from einops import rearrange

try:
    import overcomplete
    from overcomplete.visualization import show
except ImportError:
    print("ERROR: overcomplete library not found. Install with: pip install overcomplete")
    exit(1)

from utils import ROOT_DIR, DATA_DIR


class TopKSAE:
    """Wrapper for topK Sparse Autoencoder from overcomplete library."""
    
    def __init__(self, input_dim, hidden_dim, k_percent=5, device='cpu'):
        """
        Initialize topK SAE.
        
        Args:
            input_dim: Size of input (flattened patch)
            hidden_dim: Hidden dimension / expansion ratio for overcomplete representation
            k_percent: Percentage of hidden units to activate (sparsity control)
            device: 'cpu' or 'cuda'
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_percent = k_percent
        self.device = device
        
        # Create overcomplete SAE model
        self.model = overcomplete.models.TopK(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k_percent=k_percent,
        ).to(device)
        
        self.device_obj = torch.device(device)
    
    def encode(self, x):
        """Get sparse codes from input."""
        return self.model.encode(x)
    
    def decode(self, codes):
        """Reconstruct from codes."""
        return self.model.decode(codes.detach())
    
    def forward(self, x):
        """Full forward pass."""
        codes = self.model.encode(x)
        recon = self.model.decode(codes.detach())
        return recon, codes
    
    def to(self, device):
        """Move model to device."""
        self.device = str(device).replace('cuda:', 'cuda').replace('cpu', 'cpu')
        self.device_obj = torch.device(self.device)
        self.model = self.model.to(device)
        return self
    
    def train(self):
        """Set to training mode."""
        self.model.train()
    
    def eval(self):
        """Set to eval mode."""
        self.model.eval()
    
    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)


def extract_patches(images, patch_size):
    """Extract patches from spectrogram images using einops.
    
    Args:
        images: (N, H, W) array of spectrograms
        patch_size: Tuple (patch_h, patch_w)
    
    Returns:
        patches: (N, num_patches_h, num_patches_w, patch_h*patch_w) array
        patch_dims: (num_patches_h, num_patches_w, patch_h*patch_w)
    """
    N, H, W = images.shape
    ph, pw = patch_size
    
    # Use einops to extract patches
    patches = rearrange(
        images,
        'n (nh ph) (nw pw) -> (n nh nw) (ph pw)',
        ph=ph, pw=pw
    )
    
    num_patches_h = H // ph
    num_patches_w = W // pw
    patch_dim = ph * pw
    
    return patches, (num_patches_h, num_patches_w, patch_dim)


def load_training_data(data_dir, normalize=True):
    """Load spectrograms from processed data directory.
    
    Args:
        data_dir: Path to data/processed/ directory
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        images: (N, H, W) array
    """
    print(f"📂 Loading spectrograms from {data_dir}")
    images = np.load(os.path.join(data_dir, "X_spectrograms.npy"))
    
    # Remove singleton dimension if present
    images = np.squeeze(images, axis=-1) if images.ndim == 4 else images
    
    if normalize:
        min_val = np.min(images)
        max_val = np.max(images)
        images = (images - min_val) / (max_val - min_val + 1e-7)
        print(f"   ✓ Normalized to [0, 1]")
    
    print(f"   Shape: {images.shape}")
    return images


def train_sae(model, train_loader, epochs, lr, device='cpu'):
    """Train the topK SAE.
    
    Args:
        model: TopKSAE instance
        train_loader: DataLoader for training patches
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        losses: List of average epoch losses
    """
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_patches in train_loader:
            batch_patches = batch_patches[0].to(device).float()
            
            # Forward pass
            recon, codes = model.forward(batch_patches)
            
            # Reconstruction loss
            loss = criterion(recon, batch_patches)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
    
    return losses


def save_sae(model, output_dir, config):
    """Save trained SAE model and configuration.
    
    Args:
        model: TopKSAE instance
        output_dir: Directory to save to
        config: Configuration dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, "topk_sae_weights.pt")
    torch.save(model.state_dict(), model_path)
    print(f"   ✓ Model weights: {model_path}")
    
    # Save config
    config_path = os.path.join(output_dir, "topk_sae_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   ✓ Config: {config_path}")


def main(args):
    print("=" * 70)
    print("🧠 Training topK Sparse Autoencoder (SAE) on Spectrograms")
    print("=" * 70)
    
    # Set device (CPU by default since user doesn't have CUDA)
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"\n🖥️ Device: {device}")
    
    # Load spectrograms
    data_dir = os.path.join(DATA_DIR, "processed")
    images = load_training_data(data_dir, normalize=True)
    
    # Extract patches
    print(f"\n📐 Extracting patches with size {args.patch_size}x{args.patch_size}")
    patch_size = (args.patch_size, args.patch_size)
    patches, patch_dims = extract_patches(images, patch_size)
    num_patches_h, num_patches_w, patch_dim = patch_dims
    
    print(f"   ✓ Patches shape: {patches.shape}")
    print(f"   ✓ Patch grid: {num_patches_h}x{num_patches_w}")
    print(f"   ✓ Patch dimension: {patch_dim}")
    
    # Normalize patches
    print(f"\n📊 Normalizing patches")
    scaler = StandardScaler()
    patches_scaled = scaler.fit_transform(patches)
    print(f"   ✓ Mean: {patches_scaled.mean():.6f}, Std: {patches_scaled.std():.6f}")
    
    # Create dataset and dataloader
    tensor_patches = torch.from_numpy(patches_scaled).float()
    dataset = TensorDataset(tensor_patches)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # CPU only
    )
    
    print(f"\n📦 DataLoader created")
    print(f"   ✓ Batch size: {args.batch_size}")
    print(f"   ✓ Total batches: {len(train_loader)}")
    
    # Initialize SAE
    print(f"\n🏗️  Initializing topK SAE")
    sae = TopKSAE(
        input_dim=patch_dim,
        hidden_dim=args.hidden_dim,
        k_percent=args.k_percent,
        device=device
    )
    print(f"   ✓ Input dim: {patch_dim}")
    print(f"   ✓ Hidden dim: {args.hidden_dim}")
    print(f"   ✓ Sparsity (k): {args.k_percent}%")
    
    # Train SAE
    print(f"\n🎓 Training for {args.epochs} epochs")
    sae.train()
    losses = train_sae(
        sae,
        train_loader,
        args.epochs,
        args.lr,
        device=device
    )
    
    # Save model
    print(f"\n💾 Saving model")
    config = {
        "input_dim": patch_dim,
        "hidden_dim": args.hidden_dim,
        "k_percent": args.k_percent,
        "patch_size": args.patch_size,
        "num_patches_h": num_patches_h,
        "num_patches_w": num_patches_w,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_loss": float(losses[-1]),
    }
    
    output_dir = os.path.join(ROOT_DIR, "models", "topk_sae")
    save_sae(sae, output_dir, config)
    
    # Save scaler for inference
    import joblib
    scaler_path = os.path.join(output_dir, "patch_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Patch scaler: {scaler_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"\n📌 Model saved to: {output_dir}/")
    print(f"   Use sae_cnn_integration.py to hook into your CNN")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train topK Sparse Autoencoder on spectrograms"
    )
    parser.add_argument(
        '--patch_size', type=int, default=32,
        help='Patch size for spectrogram (default: 32)'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=256,
        help='Hidden (expansion) dimension for SAE (default: 256)'
    )
    parser.add_argument(
        '--k_percent', type=float, default=5,
        help='Sparsity: percentage of hidden units to activate (default: 5)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--use_cuda', action='store_true',
        help='Use CUDA if available (default: CPU only)'
    )
    
    args = parser.parse_args()
    main(args)
