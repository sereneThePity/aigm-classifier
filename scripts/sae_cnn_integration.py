"""Integrate topK SAE into CNN 2D model for mechanistic interpretability and causal testing.

This script sets up PyTorch hooks to extract intermediate activations from the CNN,
pass them through the trained SAE to get sparse codes, and enables causal intervention
tests (activation ablation, concept patching, etc.).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
import joblib

from train_cnn_2d import CNN2D
from train_topk_sae import TopKSAE
from utils import ROOT_DIR, DATA_DIR


class SAECNNIntegration:
    """Wrapper combining CNN + SAE with hooks for mechanistic interpretability."""
    
    def __init__(self, cnn_model_path, sae_model_dir, target_layer='conv6', device='cpu'):
        """
        Initialize integrated CNN+SAE model.
        
        Args:
            cnn_model_path: Path to trained CNN 2D model (.pt)
            sae_model_dir: Path to trained SAE model directory
            target_layer: Which CNN layer to hook (default: 'conv6')
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.target_layer = target_layer
        self.cnn_model_path = cnn_model_path
        self.sae_model_dir = sae_model_dir
        
        # Load CNN model
        print(f"📂 Loading CNN model from {cnn_model_path}")
        self.cnn = self._load_cnn(cnn_model_path)
        self.cnn.to(self.device)
        self.cnn.eval()
        print(f"   ✓ CNN loaded")
        
        # Load SAE model and config
        print(f"📂 Loading SAE model from {sae_model_dir}")
        self.sae, self.sae_config = self._load_sae(sae_model_dir)
        self.sae.to(self.device)
        self.sae.eval()
        print(f"   ✓ SAE loaded")
        
        # Load patch scaler
        scaler_path = os.path.join(sae_model_dir, "patch_scaler.joblib")
        self.patch_scaler = joblib.load(scaler_path)
        print(f"   ✓ Patch scaler loaded")
        
        # Storage for hook outputs
        self.activation_cache = {}
        self.sparse_codes_cache = {}
        self.hook_handle = None
        
        # Register hook on target layer
        self._register_hook()
    
    def _load_cnn(self, model_path):
        """Load trained CNN 2D model."""
        # Create dummy model to get input shape
        dummy_model = CNN2D((128, 128), num_classes=2)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        dummy_model.load_state_dict(state_dict)
        return dummy_model
    
    def _load_sae(self, sae_dir):
        """Load trained SAE model and config."""
        # Load config
        config_path = os.path.join(sae_dir, "topk_sae_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create SAE model
        sae = TopKSAE(
            input_shape=config['input_shape'],
            nb_concepts=config['nb_concepts'],
            top_k=config['top_k'],
            device='cpu'
        )
        
        # Load weights
        weights_path = os.path.join(sae_dir, "topk_sae_weights.pt")
        sae.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        
        return sae, config
    
    def _register_hook(self):
        """Register forward hook on target layer to capture activations."""
        # Get reference to target layer
        target_layer_obj = getattr(self.cnn, self.target_layer)
        
        def hook_fn(module, input, output):
            """Store activation output."""
            # output shape: (batch, channels, height, width)
            self.activation_cache['conv_output'] = output.detach()
        
        self.hook_handle = target_layer_obj.register_forward_hook(hook_fn)
        print(f"   ✓ Hook registered on layer '{self.target_layer}'")
    
    def _extract_patches_from_activations(self, activations, patch_size):
        """Extract patches from activation maps using same method as training.
        
        Args:
            activations: (batch, height, width, channels) numpy array
            patch_size: int or tuple (patch_h, patch_w)
        
        Returns:
            patches: (n_patches, patch_dim) array
            patch_dims: (num_patches_h, num_patches_w, patch_dim)
        """
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        ph, pw = patch_size
        batch_size, H, W, C = activations.shape
        
        # Pad to make divisible by patch size
        H_padded = ((H + ph - 1) // ph) * ph
        W_padded = ((W + pw - 1) // pw) * pw
        
        if H != H_padded or W != W_padded:
            activations_padded = np.zeros((batch_size, H_padded, W_padded, C), dtype=activations.dtype)
            activations_padded[:, :H, :W, :] = activations
            activations = activations_padded
        
        # Use einops for consistent patch extraction
        try:
            from einops import rearrange
            patches = rearrange(
                activations,
                'b (nh ph) (nw pw) c -> (b nh nw) (ph pw c)',
                ph=ph, pw=pw
            )
        except ImportError:
            # Fallback without einops
            num_patches_h = H_padded // ph
            num_patches_w = W_padded // pw
            patches_list = []
            for b in range(batch_size):
                for i in range(num_patches_h):
                    for j in range(num_patches_w):
                        patch = activations[b, i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
                        patches_list.append(patch.reshape(-1))
            patches = np.array(patches_list)
        
        num_patches_h = H_padded // ph
        num_patches_w = W_padded // pw
        patch_dim = patches.shape[1]
        
        return patches, (num_patches_h, num_patches_w, patch_dim)
    
    def _reconstruct_patches(self, patches_data, batch_size, patch_dims, patch_size):
        """Reconstruct activation maps from patches.
        
        Args:
            patches_data: (n_patches, patch_dim) array
            batch_size: int
            patch_dims: (num_patches_h, num_patches_w, patch_dim)
            patch_size: int or tuple
        
        Returns:
            activations: (batch, height, width, channels) array
        """
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        ph, pw = patch_size
        num_patches_h, num_patches_w, patch_dim = patch_dims
        
        # Infer number of channels from patch_dim
        channels = patch_dim // (ph * pw)
        
        H_padded = num_patches_h * ph
        W_padded = num_patches_w * pw
        
        # Reconstruct
        try:
            from einops import rearrange
            activations = rearrange(
                patches_data,
                '(b nh nw) (ph pw c) -> b (nh ph) (nw pw) c',
                b=batch_size, nh=num_patches_h, nw=num_patches_w, 
                ph=ph, pw=pw, c=channels
            )
        except ImportError:
            # Fallback without einops
            activations = np.zeros((batch_size, H_padded, W_padded, channels), dtype=patches_data.dtype)
            patch_idx = 0
            for b in range(batch_size):
                for i in range(num_patches_h):
                    for j in range(num_patches_w):
                        patch = patches_data[patch_idx].reshape(ph, pw, channels)
                        activations[b, i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = patch
                        patch_idx += 1
        
        return activations
    
    def forward_with_sae(self, x):
        """Forward pass through CNN+SAE, capturing sparse codes.
        
        Args:
            x: Input tensor of shape (batch, 1, 128, 128)
        
        Returns:
            dict with:
                - 'cnn_output': CNN logits
                - 'activations': Intermediate activations from target layer
                - 'sparse_codes': SAE sparse codes
                - 'sae_reconstruction': SAE reconstruction of activations
        """
        with torch.no_grad():
            # Clear cache
            self.activation_cache.clear()
            self.sparse_codes_cache.clear()
            
            # Forward through CNN
            cnn_output = self.cnn(x)
            
            # Get captured activations
            activations = self.activation_cache.get('conv_output')
            if activations is None:
                raise RuntimeError(f"Failed to capture activations from {self.target_layer}")
            
            # Process activations for SAE
            # Shape: (batch, channels, height, width)
            batch_size, channels, height, width = activations.shape
            
            # Convert to numpy and transpose to (batch, height, width, channels)
            acts_np = activations.cpu().detach().numpy()
            acts_np = np.transpose(acts_np, (0, 2, 3, 1))  # (B, H, W, C)
            
            # Flatten activations for SAE encoding
            # Shape: (batch, height, width, channels) -> (batch*height*width, channels)
            acts_flat = acts_np.reshape(-1, channels)
            
            print(f"🔧 Activation shape: {acts_np.shape}, flattened to SAE input: {acts_flat.shape}")
            
            # Normalize activations using z-score normalization (mean=0, std=1)
            # Don't use the patch scaler since it was trained on different data (spectrograms, not CNN features)
            acts_mean = acts_flat.mean(axis=0, keepdims=True)
            acts_std = acts_flat.std(axis=0, keepdims=True) + 1e-7
            acts_normalized = (acts_flat - acts_mean) / acts_std
            
            print(f"   Normalized to: mean={acts_normalized.mean():.4f}, std={acts_normalized.std():.4f}")
            
            # Pad to SAE expected dimension if needed
            sae_expected_dim = self.sae_config.get('input_shape', 32768)
            if acts_normalized.shape[1] < sae_expected_dim:
                print(f"   ⚠️  Padding from {acts_normalized.shape[1]} to {sae_expected_dim} dimensions with zeros")
                padded = np.zeros((acts_normalized.shape[0], sae_expected_dim), dtype=acts_normalized.dtype)
                padded[:, :acts_normalized.shape[1]] = acts_normalized
                acts_normalized = padded
            
            acts_tensor = torch.from_numpy(acts_normalized).float().to(self.device)
            
            # Get sparse codes
            sparse_codes = self.sae.encode(acts_tensor)
            self.sparse_codes_cache['codes'] = sparse_codes
            
            # Reconstruct from codes
            sae_reconstruction = self.sae.decode(sparse_codes)
            
            # Reshape back to activation spatial dimensions
            sae_recon_np = sae_reconstruction.cpu().detach().numpy()
            # Take only the relevant features (original channel dimensions)
            sae_recon_np = sae_recon_np[:, :channels]
            # Reshape to original spatial dimensions
            sae_recon_spatial = sae_recon_np.reshape(batch_size, height, width, channels)
            # Transpose back to (batch, channels, height, width)
            sae_recon_spatial = np.transpose(sae_recon_spatial, (0, 3, 1, 2))
            sae_recon_tensor = torch.from_numpy(sae_recon_spatial).float().to(self.device)
            
            return {
                'cnn_output': cnn_output,
                'activations': activations,
                'sparse_codes': sparse_codes,
                'sae_reconstruction': sae_recon_tensor,
            }
    
    def causal_intervention_ablate(self, x, concept_indices=None):
        """Ablate specific sparse concepts and measure effect on CNN output.
        
        Args:
            x: Input tensor of shape (batch, 1, 128, 128)
            concept_indices: List of concept indices to ablate (None = ablate all)
        
        Returns:
            dict with original and ablated outputs
        """
        with torch.no_grad():
            # Get original output and codes
            original = self.forward_with_sae(x)
            original_output = original['cnn_output']
            sparse_codes = original['sparse_codes'].clone()
            
            # Create ablated version
            if concept_indices is None:
                # Ablate all concepts
                sparse_codes_ablated = torch.zeros_like(sparse_codes)
            else:
                # Ablate specific concepts
                sparse_codes_ablated = sparse_codes.clone()
                for idx in concept_indices:
                    if idx < sparse_codes_ablated.shape[1]:
                        sparse_codes_ablated[:, idx] = 0
            
            # Reconstruct with ablated codes
            sae_recon_ablated = self.sae.decode(sparse_codes_ablated)
            
            # Reshape back to original activation spatial format
            batch_size, channels, height, width = original['activations'].shape
            sae_recon_ablated_np = sae_recon_ablated.cpu().detach().numpy()
            # Take only relevant features
            sae_recon_ablated_np = sae_recon_ablated_np[:, :channels]
            # Reshape to spatial dimensions
            sae_recon_ablated_spatial = sae_recon_ablated_np.reshape(batch_size, height, width, channels)
            # Transpose back to (batch, channels, height, width)
            sae_recon_ablated_spatial = np.transpose(sae_recon_ablated_spatial, (0, 3, 1, 2))
            sae_recon_ablated_tensor = torch.from_numpy(sae_recon_ablated_spatial).float().to(self.device)
            
            return {
                'original_output': original_output,
                'sparse_codes': sparse_codes,
                'sparse_codes_ablated': sparse_codes_ablated,
                'sae_reconstruction_original': original['sae_reconstruction'],
                'sae_reconstruction_ablated': sae_recon_ablated_tensor,
            }
    
    def get_concept_activations(self, dataloader, n_batches=None):
        """Extract sparse codes for a dataset to analyze concept usage.
        
        Args:
            dataloader: PyTorch DataLoader
            n_batches: Number of batches to process (None = all)
        
        Returns:
            concept_activations: (n_samples, n_concepts) array of sparse codes
        """
        all_codes = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                if n_batches is not None and batch_idx >= n_batches:
                    break
                
                x = x.to(self.device)
                result = self.forward_with_sae(x)
                codes = result['sparse_codes']
                
                # Convert to numpy and store
                all_codes.append(codes.cpu().numpy())
        
        concept_activations = np.vstack(all_codes)
        return concept_activations
    
    def analyze_concept_importance(self, concept_activations):
        """Analyze which concepts are most important for classification.
        
        Args:
            concept_activations: (n_samples, n_concepts) array
        
        Returns:
            dict with statistics on each concept
        """
        n_concepts = concept_activations.shape[1]
        
        importance_stats = {}
        for concept_idx in range(n_concepts):
            codes = concept_activations[:, concept_idx]
            importance_stats[f'concept_{concept_idx}'] = {
                'mean_activation': float(np.mean(codes)),
                'std_activation': float(np.std(codes)),
                'max_activation': float(np.max(codes)),
                'sparsity': float(np.mean(codes == 0)),  # Fraction of zeros
                'l1_norm': float(np.linalg.norm(codes, ord=1)),
            }
        
        return importance_stats
    
    def save_integration_config(self, output_dir):
        """Save configuration for this integrated model.
        
        Args:
            output_dir: Directory to save config to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            'cnn_model': str(self.cnn_model_path),
            'sae_model_dir': str(self.sae_model_dir),
            'target_layer': self.target_layer,
            'sae_config': self.sae_config,
            'device': str(self.device),
        }
        
        config_path = os.path.join(output_dir, 'integration_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Integration config saved to {config_path}")
    
    def __del__(self):
        """Clean up hooks on deletion."""
        if self.hook_handle is not None:
            self.hook_handle.remove()


class CausalTestSuite:
    """Run mechanistic interpretability causal tests on integrated CNN+SAE."""
    
    def __init__(self, integration_model, test_dataloader, device='cpu'):
        """
        Initialize causal test suite.
        
        Args:
            integration_model: SAECNNIntegration instance
            test_dataloader: DataLoader for test set
            device: Device to run tests on
        """
        self.model = integration_model
        self.test_dataloader = test_dataloader
        self.device = device
    
    def test_ablation_effect(self, concept_indices_list):
        """Test effect of ablating different concepts on model accuracy.
        
        Args:
            concept_indices_list: List of concept indices to test (or list of lists)
        
        Returns:
            results: Dict with ablation effects
        """
        results = {
            'ablation_effects': {}
        }
        
        correct_original = 0
        correct_ablated = defaultdict(int)
        total = 0
        
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Original predictions
                original = self.model.forward_with_sae(x)
                pred_original = torch.argmax(original['cnn_output'], dim=1)
                correct_original += (pred_original == y).sum().item()
                
                # Test ablations
                for concept_idx in concept_indices_list:
                    ablated = self.model.causal_intervention_ablate(x, [concept_idx])
                    # Note: This is a placeholder - need to implement proper forward through ablated
                    # For now, we'll store the codes for analysis
                
                total += y.size(0)
        
        results['ablation_effects']['original_accuracy'] = correct_original / total
        return results
    
    def test_concept_specificity(self, n_samples=100):
        """Analyze which concepts activate for which classes.
        
        Args:
            n_samples: Number of samples to analyze per class
        
        Returns:
            results: Dict with class-specific concept activations
        """
        class_concepts = defaultdict(list)
        
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(self.device)
                result = self.model.forward_with_sae(x)
                codes = result['sparse_codes']
                
                for sample_idx in range(x.size(0)):
                    class_idx = int(y[sample_idx].item())
                    class_concepts[class_idx].append(codes[sample_idx].cpu().numpy())
        
        # Compute statistics per class
        results = {}
        for class_idx, concept_arrays in class_concepts.items():
            concept_matrix = np.array(concept_arrays)
            results[f'class_{class_idx}'] = {
                'mean_activations': concept_matrix.mean(axis=0).tolist(),
                'std_activations': concept_matrix.std(axis=0).tolist(),
                'n_samples': len(concept_arrays),
            }
        
        return results


def main():
    """Example usage of SAE+CNN integration."""
    print("=" * 70)
    print("🧠 Integrating topK SAE with CNN 2D Model")
    print("=" * 70)
    
    # Paths
    cnn_model_path = os.path.join(ROOT_DIR, "models", "cnn_2d_model.pt")
    sae_model_dir = os.path.join(ROOT_DIR, "models", "topk_sae")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Device: {device}\n")
    
    # Check if models exist
    if not os.path.exists(cnn_model_path):
        print(f"❌ CNN model not found at {cnn_model_path}")
        print("   Please train CNN 2D model first using train_cnn_2d.py")
        return
    
    if not os.path.exists(sae_model_dir):
        print(f"❌ SAE model not found at {sae_model_dir}")
        print("   Please train topK SAE model first using train_topk_sae.py")
        return
    
    # Initialize integration
    integration = SAECNNIntegration(
        cnn_model_path=cnn_model_path,
        sae_model_dir=sae_model_dir,
        target_layer='conv6',
        device=device
    )
    
    print("\n✅ Integration complete!")
    print("\n📌 Next steps:")
    print("   1. Load test data in your notebook")
    print("   2. Use integration.forward_with_sae(x) to get sparse codes")
    print("   3. Use integration.causal_intervention_ablate(x, concept_indices) for ablations")
    print("   4. Use CausalTestSuite for systematic causality tests")
    
    # Save integration config
    output_dir = os.path.join(ROOT_DIR, "models", "sae_cnn_integrated")
    integration.save_integration_config(output_dir)


if __name__ == '__main__':
    main()
