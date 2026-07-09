"""Train a 2D CNN classifier on mel-spectrogram latents."""
import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ROOT_DIR
from preprocess import load_spectrogram_latents_for_training, load_cached_spectrograms_for_training, collate_fn_skip_none


class CNN2D_Legacy(nn.Module):
    """Legacy 2-layer 2D CNN (smaller version used for cnn_model.pt)."""
    def __init__(self, input_shape, num_classes=2):
        super(CNN2D_Legacy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Input: (batch, 1, 128, 128)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)  # (batch, 32, 64, 64)
        x = self.relu2(self.conv2(x))
        x = self.global_pool(x).view(x.size(0), -1)  # (batch, 64)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class CNN2D(nn.Module):
    """6-layer 2D CNN for mel-spectrogram classification."""
    def __init__(self, input_shape, num_classes=2):
        super(CNN2D, self).__init__()
        # 6 convolutional layers with filters [16, 32, 64, 128, 256, 512]
        # Using Conv2d for 2D spectrogram images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Input: (batch, 1, 128, 128)
        
        # 6 Conv2d blocks
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)  
        
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)  
        
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)  
        
        x = self.relu4(self.conv4(x))
        x = self.pool4(x)  
        
        x = self.relu5(self.conv5(x))
        x = self.pool5(x)  
        
        x = self.relu6(self.conv6(x))
        x = self.pool6(x)  
        
        # Global average pooling and fully connected
        x = self.global_pool(x).view(x.size(0), -1)  # (batch, 1024)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 2D CNN on mel-spectrograms (no neural codec bias)")
    parser.add_argument('--use_cached', action='store_true', help='Use pre-computed cached spectrograms (fastest)')
    parser.add_argument('--cached_manifest', type=str, default=None, help='Path to cached spectrogram manifest')
    parser.add_argument('--latent_dir', type=str, default=None, help='Path to encoded latent directory (with neural codecs)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=None, help='Max samples to load (None = all data)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎵 Training 2D CNN on Mel-Spectrograms")
    print("=" * 70)
    
    # Determine data source
    data_source = None
    if args.use_cached:
        # Fastest approach: pre-computed spectrograms (no neural codecs)
        if not args.cached_manifest:
            args.cached_manifest = str(Path(ROOT_DIR) / "data/cached_spectrograms/manifest.csv")
        print(f"Mode: Cached Spectrograms (no neural codec bias)")
        print(f"Manifest: {args.cached_manifest}\n")
        data_source = args.cached_manifest
        train_data, val_data, test_data, input_shape = load_cached_spectrograms_for_training(
            args.cached_manifest,
            num_samples=args.samples
        )
    else:
        # Default approach: neural codec latents  
        if not args.latent_dir:
            args.latent_dir = str(Path(ROOT_DIR) / "data/encoded_trainset")
        print(f"Mode: Neural Codec Latents (encodec, dac, etc.)")
        print(f"Directory: {args.latent_dir}\n")
        data_source = args.latent_dir
        train_data, val_data, test_data, input_shape = load_spectrogram_latents_for_training(
            args.latent_dir,
            num_samples_per_class=args.samples
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Setup data loaders
    # Cached specs are already fast (just numpy I/O), latents benefit from workers
    num_workers = 0 if args.use_cached else min(4, os.cpu_count() or 1)
    collate_fn = collate_fn_skip_none if args.use_cached else None
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=(num_workers > 0), collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=(num_workers > 0), collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(num_workers > 0), collate_fn=collate_fn)
    
    # Build model
    model = CNN2D(input_shape, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")
    
    # Training loop
    print(f"Training ({args.epochs} epochs, LR={args.lr})\n")
    best_val_loss = float('inf')
    patience_counter = 0
    
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs")
    for epoch in epoch_pbar:
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"[Train]", leave=False)
        
        for X, y in train_pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * y.size(0)
            train_correct += (torch.max(outputs, 1)[1] == y).sum().item()
            train_total += y.size(0)
            train_pbar.set_postfix({'loss': train_loss/train_total})
        
        train_loss /= train_total
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[Val]", leave=False)
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item() * y.size(0)
                val_correct += (torch.max(outputs, 1)[1] == y).sum().item()
                val_total += y.size(0)
                val_pbar.set_postfix({'loss': val_loss/val_total})
        
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}', 'train_acc': f'{train_acc:.2f}%',
            'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.2f}%'
        })
        
        # Early stopping with patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping (val_loss not improving for 5 epochs)")
                break
    
    # Test
    print("\n" + "=" * 70)
    print("Testing...")
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[Test]", leave=False)
        for X, y in test_pbar:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            test_loss += loss.item() * y.size(0)
            test_correct += (torch.max(outputs, 1)[1] == y).sum().item()
            test_total += y.size(0)
    
    test_loss /= test_total
    test_acc = 100 * test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")
    print("=" * 70)
    
    # Save model and training info
    print("\nSaving model and training info...")
    model_dir = Path(ROOT_DIR) / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Save model checkpoint
    model_suffix = "CNN_cached_spec.pt"
    model_path = model_dir / model_suffix
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save training info
    training_info = {
        "model": "CNN2D",
        "data_mode": "cached_specs" if args.use_cached else "neural_codecs",
        "data_source": data_source,
        "num_training_samples": len(train_data),
        "epochs_trained": epoch + 1,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_train_loss": float(train_loss),
        "final_train_acc": float(train_acc),
        "final_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "parameters": int(params),
        "device": str(device)
    }
    
    info_path = model_dir / f"training_info_{model_suffix}.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✅ Training info saved to {info_path}")
    print("=" * 70)

