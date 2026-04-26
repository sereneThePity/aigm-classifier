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
    from overcomplete.sae import train_sae as overcomplete_train_sae
except ImportError:
    print("ERROR: overcomplete library not found. Install with: pip install overcomplete")
    exit(1)

from ..utils.utils import ROOT_DIR, DATA_DIR


class TopKSAE:
    """Wrapper for topK Sparse Autoencoder from overcomplete library."""
    
    def __init__(self, input_shape, nb_concepts, top_k=None, device='cpu'):
        """
        Initialize topK SAE.
        
        Args:
            input_shape: Shape of input (e.g., 1024 for flattened patch)
            nb_concepts: Number of concepts / hidden dimension for sparse codes
            top_k: Number of top activations to keep (if None, uses all)
            device: 'cpu' or 'cuda'
        """
        self.input_shape = input_shape
        self.nb_concepts = nb_concepts
        self.top_k = top_k
        self.device = device
        
        # Create overcomplete SAE model
        self.model = overcomplete.sae.TopKSAE(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            top_k=top_k,
            device=device
        )
        
        self.device_obj = torch.device(device)
    
    def encode(self, x):
        """Get sparse codes from input.
        
        overcomplete.sae.TopKSAE.encode() returns a tuple (codes, other_outputs)
        """
        result = self.model.encode(x)
        # Handle tuple return from overcomplete
        if isinstance(result, tuple):
            codes = result[0]
        else:
            codes = result
        return codes
    
    def decode(self, codes):
        """Reconstruct from codes."""
        if isinstance(codes, torch.Tensor):
            codes = codes.detach()
        return self.model.decode(codes)
    
    def forward(self, x):
        """Full forward pass."""
        codes = self.encode(x)
        recon = self.decode(codes)
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
    """Extract patches from images using einops.
    
    Handles 2D (H, W), 3D (N, H, W) and 4D (N, H, W, C) arrays.
    Automatically pads image dimensions to be divisible by patch_size.
    
    Args:
        images: (H, W), (N, H, W) or (N, H, W, C) array
        patch_size: Tuple (patch_h, patch_w)
    
    Returns:
        patches: (N*num_patches_h*num_patches_w, patch_dim) array
        patch_dims: (num_patches_h, num_patches_w, patch_dim)
    """
    ph, pw = patch_size
    
    if images.ndim == 2:
        # Single 2D image: (H, W)
        H, W = images.shape
        images = images[np.newaxis, :, :]  # Add batch dimension
    elif images.ndim == 3:
        N, H, W = images.shape
    elif images.ndim == 4:
        N, H, W, C = images.shape
    else:
        raise ValueError(f"Expected 2D, 3D or 4D array, got {images.ndim}D")
    
    # Pad to make divisible by patch size
    H_padded = ((H + ph - 1) // ph) * ph
    W_padded = ((W + pw - 1) // pw) * pw
    
    if H != H_padded or W != W_padded:
        print(f"   Padding from ({H}, {W}) to ({H_padded}, {W_padded})")
        if images.ndim == 3:
            images_padded = np.zeros((images.shape[0], H_padded, W_padded), dtype=images.dtype)
            images_padded[:, :H, :W] = images
        elif images.ndim == 4:
            images_padded = np.zeros((images.shape[0], H_padded, W_padded, images.shape[3]), dtype=images.dtype)
            images_padded[:, :H, :W, :] = images
        images = images_padded
    
    # Extract patches
    if images.ndim == 3:
        patches = rearrange(
            images,
            'n (nh ph) (nw pw) -> (n nh nw) (ph pw)',
            ph=ph, pw=pw
        )
    elif images.ndim == 4:
        # Extract patches per channel to avoid massive dimensionality
        # Shape: (N, H, W, C) -> (N*C*nh*nw, ph*pw)
        patches = rearrange(
            images,
            'n (nh ph) (nw pw) c -> (n c nh nw) (ph pw)',
            ph=ph, pw=pw
        )
    
    num_patches_h = H_padded // ph
    num_patches_w = W_padded // pw
    patch_dim = patches.shape[1]
    
    return patches, (num_patches_h, num_patches_w, patch_dim)


def load_training_data(data_dir, normalize=True, activation_layer='conv6'):
    """Load intermediate activations from processed data directory.
    
    Args:
        data_dir: Path to data/processed/ directory
        normalize: Whether to normalize to [0, 1]
        activation_layer: Which activation layer to load (default: 'conv6')
    
    Returns:
        images: (N, H, W, C) array of activations (reshaped if flattened)
    """
    print(f"📂 Loading intermediate activations from {data_dir}")
    
    # Try to load spatial version first (better for SAE)
    spatial_path = os.path.join(data_dir, f"intermediate_activations_{activation_layer}_spatial.npy")
    flattened_path = os.path.join(data_dir, f"intermediate_activations_{activation_layer}.npy")
    
    if os.path.exists(spatial_path):
        print(f"   ✓ Found spatial features")
        images = np.load(spatial_path)
        print(f"   Shape loaded: {images.shape}")
    elif os.path.exists(flattened_path):
        print(f"   ⚠️  Only flattened features found, reshaping...")
        images = np.load(flattened_path)
        print(f"   Shape loaded: {images.shape}")
        
        # If 2D (flattened), try to reshape back to spatial form
        if images.ndim == 2:
            N = images.shape[0]
            total_features = images.shape[1]
            print(f"   Data is flattened: ({N}, {total_features})")
            
            # Assume square spatial dims and try common channel counts
            for channels in [512, 256, 128, 64]:
                spatial_size = int(np.sqrt(total_features / channels))
                if spatial_size * spatial_size * channels == total_features:
                    images = images.reshape(N, channels, spatial_size, spatial_size)
                    print(f"   ✓ Reshaped to spatial: {images.shape} (C={channels}, H=W={spatial_size})")
                    break
    else:
        raise FileNotFoundError(f"Could not find activations for layer '{activation_layer}'")
    
    # Move channels to last dimension for uniformity
    if images.ndim == 4:
        images = np.transpose(images, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
        print(f"   ✓ Transposed to (N, H, W, C): {images.shape}")
    
    # Remove singleton dimension if present (but preserve channel dim if it exists)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    
    # If 2D, add batch dimension
    if images.ndim == 2:
        images = images[np.newaxis, :, :]
        print(f"   Added batch dimension for 2D data")
    
    if normalize:
        min_val = np.min(images)
        max_val = np.max(images)
        images = (images - min_val) / (max_val - min_val + 1e-7)
        print(f"   ✓ Normalized to [0, 1]")
    
    print(f"   Final shape: {images.shape}")
    return images


def train_sae(model_dict, train_loader, epochs, lr, device='cpu'):
    """Train the topK SAE using overcomplete's train_sae with detailed metrics.
    
    Args:
        model_dict: Dict with 'model' (TopKSAE instance or raw model) and optionally other params
        train_loader: DataLoader for training patches
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        logs: Training logs from overcomplete.sae.train_sae
    """
    # Extract model (handle both TopKSAE wrapper and raw model)
    if isinstance(model_dict, dict):
        model = model_dict.get('model', model_dict)
    else:
        model = model_dict
    
    # If it's a TopKSAE wrapper, get the underlying overcomplete model
    if hasattr(model, 'model'):
        actual_model = model.model
    else:
        actual_model = model
    
    # Define criterion function for SAE training
    def criterion(x, x_hat, pre_codes, codes, dictionary):
        """MSE reconstruction loss."""
        mse = (x - x_hat).square().mean()
        return mse
    
    # Create optimizer
    optimizer = torch.optim.Adam(actual_model.parameters(), lr=lr)
    
    # Create a wrapper dataloader that extracts tensors from tuples
    class DataLoaderWrapper:
        def __init__(self, dataloader):
            self.dataloader = dataloader
        
        def __iter__(self):
            for batch in self.dataloader:
                # Handle (tensor,) tuple from TensorDataset
                if isinstance(batch, (list, tuple)):
                    yield batch[0].to(device).float()
                else:
                    yield batch.to(device).float()
        
        def __len__(self):
            return len(self.dataloader)
    
    wrapped_loader = DataLoaderWrapper(train_loader)
    
    # Train using overcomplete's train_sae function with detailed logging
    print("Training SAE on patches...")
    logs = overcomplete_train_sae(
        actual_model,
        wrapped_loader,
        criterion,
        optimizer,
        nb_epochs=epochs,
        device=device
    )
    
    return logs


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
    images = load_training_data(data_dir, normalize=True, activation_layer=args.activation_layer)
    
    # Extract patches
    print(f"\n📐 Extracting patches with size {args.patch_size}x{args.patch_size}")
    patch_size = (args.patch_size, args.patch_size)
    patches, patch_dims = extract_patches(images, patch_size)
    num_patches_h, num_patches_w, patch_dim = patch_dims
    
    print(f"   ✓ Patches shape: {patches.shape}")
    print(f"   ✓ Patch grid: {num_patches_h}x{num_patches_w}")
    print(f"   ✓ Patch dimension: {patch_dim}")
    print(f"   ℹ️  Extracting spatial patches preserves local feature structure")
    
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
    )
    
    print(f"\n📦 DataLoader created")
    print(f"   ✓ Batch size: {args.batch_size}")
    print(f"   ✓ Total batches: {len(train_loader)}")
    
    # Initialize SAE
    print(f"\n🏗️  Initializing topK SAE")
    sae = TopKSAE(
        input_shape=patch_dim,
        nb_concepts=args.nb_concepts,
        top_k=args.top_k,
        device=device
    )
    print(f"   ✓ Input shape: {patch_dim}")
    print(f"   ✓ Nb concepts: {args.nb_concepts}")
    print(f"   ✓ Top-k: {args.top_k if args.top_k else 'All'}")
    
    # Train SAE
    print(f"\n🎓 Training for {args.epochs} epochs")
    sae.train()
    logs = train_sae(
        sae,
        train_loader,
        args.epochs,
        args.lr,
        device=device
    )
    
    # Extract final loss from logs for config
    final_loss = logs[-1]['loss'] if logs and 'loss' in logs[-1] else 0.0
    
    # Save model
    print(f"\n💾 Saving model")
    config = {
        "input_shape": patch_dim,
        "nb_concepts": args.nb_concepts,
        "top_k": args.top_k,
        "patch_size": args.patch_size,
        "num_patches_h": num_patches_h,
        "num_patches_w": num_patches_w,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_loss": float(final_loss),
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
        '--patch_size', type=int, default=8,
        help='Patch size for spectrogram (default: 8)'
    )
    parser.add_argument(
        '--nb_concepts', type=int, default=128,
        help='Number of concepts (expansion dimension) for SAE (default: 128)'
    )
    parser.add_argument(
        '--top_k', type=int, default=32,
        help='Number of top-k activations to keep. If None, keeps all (default: 32)'
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
    parser.add_argument(
        '--activation_layer', type=str, default='conv6',
        help='Which activation layer to train on (default: conv6)'
    )
    
    args = parser.parse_args()
    main(args)
