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
from preprocess import load_spectrogram_latents_for_training


class CNN2D(nn.Module):
    """6-layer 2D CNN for mel-spectrogram classification."""
    def __init__(self, input_shape, num_classes=2):
        super(CNN2D, self).__init__()
        # 6 convolutional layers with filters [16, 32, 64, 128, 256, 512]
        # Using Conv2d for 2D spectrogram images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Input: (batch, 1, 128, 128)
        
        # 6 Conv2d blocks
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)  # (batch, 32, 64, 64)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)  # (batch, 64, 32, 32)
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)  # (batch, 128, 16, 16)
        
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)  # (batch, 256, 8, 8)
        
        x = torch.relu(self.conv5(x))
        x = self.pool5(x)  # (batch, 512, 4, 4)
        
        x = torch.relu(self.conv6(x))
        x = self.pool6(x)  # (batch, 1024, 2, 2)
        
        # Global average pooling and fully connected
        x = self.global_pool(x).view(x.size(0), -1)  # (batch, 1024)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 2D CNN on mel-spectrogram latents")
    parser.add_argument('--latent_dir', type=str, required=True, help='Path to spectrogram directory (encoded_trainset)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples per class (default: None = all data)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎵 Training 2D CNN on Mel-Spectrogram Latents")
    print("=" * 70)
    
    # Load data from all codec subdirectories
    train_data, val_data, test_data, input_shape = load_spectrogram_latents_for_training(
        args.latent_dir,
        num_samples_per_class=args.samples
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Setup loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
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
    model_path = model_dir / "cnn_2d_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save training info
    training_info = {
        "model": "CNN2D",
        "input_shape": input_shape,
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
    
    info_path = model_dir / "training_info_2d.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✅ Training info saved to {info_path}")
    print("=" * 70)

