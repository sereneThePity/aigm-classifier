"""Train a simple CNN classifier on mel-spectrogram inputs."""
import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import ROOT_DIR, DATA_DIR
from preprocess import load_dataset_comprehensive
from neural_codec_confounders import get_available_codecs


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        # 6 convolutional layers with filters [16, 32, 64, 128, 256, 512]
        # Kernel size 3, pooling size 2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Remove any trailing singleton dimensions
        while x.dim() > 3:
            x = x.squeeze(-1)
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 6 convolutional blocks
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
        
        # Global average pooling and fully connected layers
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_simple_cnn(input_shape, num_classes):
    model = SimpleCNN(input_shape, num_classes)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN classifier on preprocessed audio data")
    parser.add_argument('--manifest', default=os.path.join(DATA_DIR, "trainset/manifest.csv"), type=str, help='Path to manifest CSV file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set fraction')
    parser.add_argument('--segment_duration', type=float, default=5.0, help='Audio segment duration in seconds')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bins')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for multiprocessing (reduce to 2-4 when using codecs)')
    parser.add_argument('--device_type', choices=['cpu', 'gpu', 'both'], default='gpu', 
                        help='Device type: "cpu" uses griffinmel/audiolm/valle, "gpu" uses encodec_meta/dac, "both" uses all available codecs')
    parser.add_argument('--codec', nargs='?', const='random', default=None, 
                        help='Apply neural codec to audio during training. Use "random" (default) to pick random codec per sample, or specify a codec name. Available on CPU: griffinmel, audiolm, valle. Available on GPU: encodec_meta, dac')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎵 Training CNN Classifier on Preprocessed Audio Data")
    print("=" * 70)
    
    # Check manifest exists
    if not os.path.exists(args.manifest):
        print(f"❌ Manifest file not found: {args.manifest}")
        print(f"   Please create a manifest CSV using build_manifest.py")
        exit(1)
    
    # Load preprocessed dataset
    print(f"\n📂 Loading dataset...")
    
    # Setup codec augmentation if requested
    codec_name = None
    if args.codec:
        # Get available codecs based on device_type
        if args.device_type == 'both':
            gpu_codecs = get_available_codecs(device_type='gpu')
            cpu_codecs = get_available_codecs(device_type='cpu')
            available_codecs = list(set(gpu_codecs + cpu_codecs))  # Combine and remove duplicates
        else:
            available_codecs = get_available_codecs(device_type=args.device_type)
        
        if available_codecs:
            # Validate codec_name if it's not 'random'
            if args.codec != 'random' and args.codec not in available_codecs:
                print(f"   ⚠️  Codec '{args.codec}' not available. Available: {', '.join(sorted(available_codecs))}")
                codec_name = 'random'
            else:
                codec_name = args.codec
        else:
            print(f"   ⚠️  No codecs available for {args.device_type}")
            codec_name = None
    
    X, y = load_dataset_comprehensive(
        args.manifest,
        n_mels=args.n_mels,
        target_shape=(args.n_mels, 128),
        segment_duration=args.segment_duration,
        target_loudness=-20.0,
        hp_freq=20,
        num_workers=args.num_workers,
        codec_name=codec_name,
        device_type=args.device_type
    )
    
    
    # Compute input shape from loaded data
    input_shape = X.shape[1:3]  # (freq_bins, time_steps)
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    
    print(f"✅ Dataset loaded: {X.shape}")
    
    # Split into train, val, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=42, stratify=y_train
    )
    
    print(f"📊 Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_simple_cnn(input_shape, num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    print(f"🏗️ Model built ({sum(p.numel() for p in model.parameters()):,} params, device={device})")
    
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\n🚀 Training ({args.epochs} epochs, LR={args.lr})")
    print("="*50)
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️ Early stopping triggered (patience={patience})")
                break
    
    # Test phase
    print(f"\n📈 Evaluating test set")
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
    
    test_loss /= test_total
    test_acc = 100 * test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    
    # Save model
    models_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'cnn_model_hpc.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save training info
    training_info = {
        'model_path': model_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'num_samples': int(len(X)),
        'num_classes': int(num_classes),
        'input_shape': list(input_shape),
        'segment_duration': args.segment_duration,
        'n_mels': args.n_mels,
    }
    
    info_path = os.path.join(models_dir, 'training_info_hpc.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✅ Training complete! Model: {model_path}")
