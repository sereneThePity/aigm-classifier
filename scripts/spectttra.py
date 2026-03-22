"""
Simple SpecTTTra Model Loader
Loads SONICS SpecTTTra model for audio classification (real vs. fake songs)
"""

import numpy as np
import librosa
import torch
from pathlib import Path

try:
    from sonics import HFAudioClassifier
except ImportError:
    raise ImportError("Install with: pip install git+https://github.com/awsaf49/sonics.git")


class SpecTTraModel:
    def __init__(self, model_name="awsaf49/sonics-spectttra-gamma-5s"):
        """
        Load SpecTTTra model
        
        Args:
            model_name: HuggingFace model ID
                - "awsaf49/sonics-spectttra-alpha-5s" (best accuracy)
                - "awsaf49/sonics-spectttra-beta-5s" (balanced)
                - "awsaf49/sonics-spectttra-gamma-5s" (most efficient)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HFAudioClassifier.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.sr = 16000  # SpecTTTra expects 16kHz
        print(f"✅ Loaded {model_name} on {self.device}")
    
    def predict(self, audio_path):
        """
        Predict if audio is real (0) or fake (1)
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            {'prediction': 0/1, 'confidence': float, 'label': 'REAL'/'FAKE'}
        """
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            outputs = self.model(audio_tensor)
            print(f"   Model output (logit): {outputs.cpu().numpy().squeeze():.4f}")
            # Model returns a single tensor (logit) - convert to probability
            # Shape is (1, 1), we need the scalar value
            logit = float(outputs.squeeze().item())
            pred = 1 if logit > 0.5 else 0  # Convert logit to binary prediction
            conf = abs(logit)  # Rough confidence estimate
            label = "FAKE" if pred == 1 else "REAL"
            
            return {
                'prediction': pred,
                'confidence': conf,
                'label': label
            }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python spectttra.py <audio_file>")
        sys.exit(1)
    
    model = SpecTTraModel()
    result = model.predict(sys.argv[1])
    
    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.4f})")
