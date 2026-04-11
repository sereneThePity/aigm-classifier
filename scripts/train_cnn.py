"""Train a CNN classifier on encoded latents."""
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ROOT_DIR
from preprocess import load_encoded_latents_for_training


class SimpleCNN(nn.Module):
    """6-layer 1D CNN for latent vector classification."""
    def __init__(self, input_shape, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 6 convolutional layers with filters [16, 32, 64, 128, 256, 512]
        # Using Conv1d for 1D latent vectors
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(2)
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool1d(2)
        
        self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool1d(2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Handle shape variations
        while x.dim() > 3:
            x = x.squeeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 6 Conv1d blocks
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = torch.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = torch.relu(self.conv6(x))
        x = self.pool6(x)
        
        # Global average pooling and fully connected
        x = self.global_pool(x).view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN on encoded latents")
    parser.add_argument('--latent_dir', type=str, required=True, help='Path to encoded_latents directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples per class (default: None = all data)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎵 Training CNN on Encoded Latents")
    print("=" * 70)
    
    # Load data from all encoder subdirectories
    train_data, val_data, test_data, input_shape = load_encoded_latents_for_training(
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
    model = SimpleCNN(input_shape, num_classes=2).to(device)
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
        
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.1f}%',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.1f}%'
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'models', f'cnn_best_precomputed.pt'))
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test
    print("\nEvaluating test set...")
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            test_loss += loss.item() * y.size(0)
            test_correct += (torch.max(outputs, 1)[1] == y).sum().item()
            test_total += y.size(0)
    
    test_loss /= test_total
    test_acc = 100 * test_correct / test_total
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    
    # Save model and info
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    model_path = os.path.join(ROOT_DIR, 'models', 'cnn_model_precomputed.pt')
    torch.save(model.state_dict(), model_path)
    
    info = {
        'model': 'SimpleCNN',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'params': params,
        'input_shape': list(input_shape),
    }
    with open(os.path.join(ROOT_DIR, 'models', 'training_info_precomputed.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Done! Model saved to {model_path}")
