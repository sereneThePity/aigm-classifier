"""
Neural Codec Confounders Module

Applies various neural audio codecs to audio samples as confounders.
This simulates compression artifacts that real AIGM systems may introduce.

Supported Codecs:
1. Meta EnCodec - Facebook's neural codec
2. DAC (Descript Audio Codec) - High-quality neural codec
3. AudioLM codec tokenizer - Google's codec from AudioLM
4. VALL-E codec tokenizer - Microsoft's codec from VALL-E
5. GriffinMel - Simplified mel-spectrogram based codec
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings
import time

# Try to import codec libraries, fall back gracefully
try:
    from encodec import EncodecModel
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False

try:
    from dac import DAC
    DAC_AVAILABLE = True
except ImportError:
    DAC_AVAILABLE = False


class BaseCodec(nn.Module):
    """Base class for audio codecs."""
    
    def __init__(self, sr: int = 44100):
        super().__init__()
        self.sr = sr
    
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio through the codec."""
        raise NotImplementedError
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode audio from codec representation."""
        raise NotImplementedError
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        encoded = self.encode(audio)
        decoded = self.decode(encoded)
        return decoded


class MetaEnCodecWrapper(BaseCodec):
    """Meta's EnCodec implementation."""
    
    def __init__(self, sr: int = 44100, bandwidth: float = 1.5):
        """
        Initialize Meta EnCodec.
        
        Args:
            sr: Sample rate (24000 Hz recommended)
            bandwidth: Bitrate in kbps (1.5, 3, 6, 12, 24)
        """
        super().__init__(sr)
        self.bandwidth = bandwidth
        
        if not ENCODEC_AVAILABLE:
            raise ImportError("encodec not installed. Install with: pip install encodec")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained EnCodec model
        # Note: EnCodec works best at 24kHz
        self.sr_encodec = 24000
        self.model = EncodecModel.encodec_model_24khz().to(self.device)
        self.model.eval()
    
    def encode(self, audio: np.ndarray) -> list:
        """Encode audio to EncodedFrame objects."""
        with torch.no_grad():
            # Resample if needed
            if self.sr != self.sr_encodec:
                audio_resampled = librosa.resample(
                    audio, orig_sr=self.sr, target_sr=self.sr_encodec
                )
            else:
                audio_resampled = audio
            
            # Convert to tensor and normalize
            audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Encode - returns list of EncodedFrame objects
            encoded_frames = self.model.encode(audio_tensor)
            
        return encoded_frames
    
    def decode(self, encoded_frames: list) -> np.ndarray:
        with torch.no_grad():
            # Decode - takes list of EncodedFrame objects
            decoded = self.model.decode(encoded_frames)
            
            # Convert back to numpy and resample if needed
            decoded_np = decoded.squeeze().cpu().numpy()
            
            if self.sr != self.sr_encodec:
                decoded_np = librosa.resample(
                    decoded_np, orig_sr=self.sr_encodec, target_sr=self.sr
                )
        
        return decoded_np
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        encoded_frames = self.encode(audio)
        decoded = self.decode(encoded_frames)
        # Ensure output length matches input
        result = decoded[:len(audio)]
        return result


class DACWrapper(BaseCodec):
    """Descript Audio Codec wrapper."""
    
    def __init__(self, sr: int = 44100, model_name: str = "44khz"):
        """
        Initialize DAC codec.
        
        Args:
            sr: Sample rate (16000 or 44100)
            model_name: Model variant ("16khz" or "44khz")
        """
        super().__init__(sr)
        self.model_name = model_name
        
        if not DAC_AVAILABLE:
            raise ImportError("dac not installed. Install with: pip install descript-audio-codec")
        
        self.sr_dac = 44100 if "44" in model_name else 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DAC model with default architecture
        # (Pre-trained weights would be loaded via DAC.load() if a path is provided)
        self.model = DAC(sample_rate=self.sr_dac).to(self.device)
        self.model.eval()
    
    def encode(self, audio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio through DAC."""
        with torch.no_grad():
            # Resample if needed
            if self.sr != self.sr_dac:
                audio_resampled = librosa.resample(
                    audio, orig_sr=self.sr, target_sr=self.sr_dac
                )
            else:
                audio_resampled = audio
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Encode
            code = self.model.encode(audio_tensor)[0]
            
        return code
    
    def decode(self, code: torch.Tensor) -> np.ndarray:
        """Decode from DAC codes."""
        t_start = time.time()
        with torch.no_grad():
            code = code.to(self.device)
            decoded = self.model.decode(code)
            
            decoded_np = decoded.squeeze().cpu().numpy()
            
            # Resample if needed
            if self.sr != self.sr_dac:
                decoded_np = librosa.resample(
                    decoded_np, orig_sr=self.sr_dac, target_sr=self.sr
                )

        
        return decoded_np
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        code = self.encode(audio)
        decoded = self.decode(code)
        result = decoded[:len(audio)]
        return result


class AudioLMCodecWrapper(BaseCodec):
    """AudioLM codec tokenizer simulation."""
    
    def __init__(self, sr: int = 44100, num_tokens: int = 1024):
        """
        Initialize AudioLM codec approximation.
        
        Args:
            sr: Sample rate
            num_tokens: Number of quantization tokens
        """
        super().__init__(sr)
        self.num_tokens = num_tokens
    
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode to discrete tokens using spectral quantization."""
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=256)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Quantize to tokens
        tokens = np.round(mel_spec_norm * (self.num_tokens - 1)).astype(int)
        tokens = np.clip(tokens, 0, self.num_tokens - 1)
        
        return tokens
    
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Reconstruct audio from tokens."""
        
        # Dequantize
        mel_spec_norm = tokens.astype(float) / (self.num_tokens - 1)
        
        # Scale back to dB range (rough approximation)
        mel_spec_db = mel_spec_norm * 80 - 40
        
        # Convert from dB to power
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Reconstruct audio using Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec, sr=self.sr, n_fft=2048
        )
        
        return audio
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        tokens = self.encode(audio)
        decoded = self.decode(tokens)
        return decoded[:len(audio)]


class VALLECodecWrapper(BaseCodec):
    """VALL-E codec tokenizer simulation."""
    
    def __init__(self, sr: int = 44100, num_quantizers: int = 8, num_tokens: int = 512):
        """
        Initialize VALL-E codec approximation.
        Uses hierarchical quantization similar to VALL-E.
        
        Args:
            sr: Sample rate
            num_quantizers: Number of quantization levels (hierarchy)
            num_tokens: Tokens per quantizer
        """
        super().__init__(sr)
        self.num_quantizers = num_quantizers
        self.num_tokens = num_tokens
    
    def encode(self, audio: np.ndarray) -> List[np.ndarray]:
        """Encode using hierarchical quantization."""
        
        # Short-time Fourier transform
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Log-magnitude for better representation
        log_mag = np.log(magnitude + 1e-8)
        
        # Recursive quantization
        tokens_hierarchical = []
        residual = log_mag.copy()
        
        for i in range(self.num_quantizers):
            # Normalize residual
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            
            # Quantize
            tokens = np.round(residual_norm * (self.num_tokens - 1)).astype(int)
            tokens = np.clip(tokens, 0, self.num_tokens - 1)
            tokens_hierarchical.append(tokens)
            
            # Update residual
            reconstructed = tokens.astype(float) / (self.num_tokens - 1) * (
                residual.max() - residual.min()
            ) + residual.min()
            residual = residual - reconstructed
        
        return tokens_hierarchical, phase
    
    def decode(self, tokens_hierarchical: List[np.ndarray], phase: np.ndarray) -> np.ndarray:
        """Reconstruct audio from hierarchical tokens."""
        
        # Reconstruct log-magnitude from hierarchy
        log_mag_reconstructed = np.zeros_like(phase)
        
        for tokens in tokens_hierarchical:
            tokens_norm = tokens.astype(float) / (self.num_tokens - 1)
            log_mag_reconstructed = log_mag_reconstructed + tokens_norm
        
        # Convert back to magnitude
        magnitude = np.exp(log_mag_reconstructed)
        
        # Reconstruct STFT
        D = magnitude * np.exp(1j * phase)
        audio = librosa.istft(D)
        
        return audio
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        tokens_hierarchical, phase = self.encode(audio)
        decoded = self.decode(tokens_hierarchical, phase)
        return decoded[:len(audio)]


class GriffinMelCodec(BaseCodec):
    """Simplified Griffin-Mel codec."""
    
    def __init__(self, sr: int = 44100, n_mels: int = 128, n_fft: int = 2048):
        """
        Initialize Griffin-Mel codec.
        
        Args:
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT size
        """
        super().__init__(sr)
        self.n_mels = n_mels
        self.n_fft = n_fft
    
    def encode(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to mel spectrogram."""
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def decode(self, mel_spec_db: np.ndarray) -> np.ndarray:
        """Decode from mel spectrogram using Griffin-Lim."""
        
        # Convert from dB back to power
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Reconstruct using mel_to_audio
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec, sr=self.sr, n_fft=self.n_fft
        )
        
        return audio
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        mel_spec_db = self.encode(audio)
        decoded = self.decode(mel_spec_db)
        return decoded[:len(audio)]


# Registry of all codecs
CODEC_REGISTRY = {
    "encodec_meta": MetaEnCodecWrapper,
    "dac": DACWrapper,
    "audiolm": AudioLMCodecWrapper,
    "valle": VALLECodecWrapper,
    "griffinmel": GriffinMelCodec,
}


class NeuralCodecConfounder:
    """
    Manages application of neural codec confounders to audio files.
    """
    
    def __init__(self, sr: int = 44100, init_only: Optional[str] = None, device_type: str = 'gpu'):
        """
        Initialize confounder manager.
        
        Args:
            sr: Sample rate for all codecs
            init_only: If specified, only initialize this codec (or all for 'random').
                       None initializes all available codecs filtered by device_type.
            device_type: 'cpu' initializes only CPU-friendly codecs (griffinmel, audiolm, valle)
                        'gpu' initializes all available codecs including GPU ones (encodec_meta, dac)
        """
        self.sr = sr
        self.device_type = device_type
        self.codecs = {}
        self._initialize_available_codecs(init_only=init_only, device_type=device_type)
    
    def _initialize_available_codecs(self, init_only: Optional[str] = None, device_type: str = 'gpu'):
        """Initialize only available codecs filtered by device type.
        
        Args:
            init_only: If specified, only initialize this codec.
                       Use 'random' or None to initialize based on device_type.
            device_type: 'cpu' for CPU-only codecs, 'gpu' for all codecs
        """
        # Map of codec names to lightweight (librosa-based) constructors
        lightweight = {
            "griffinmel": lambda: GriffinMelCodec(sr=self.sr),
            "audiolm": lambda: AudioLMCodecWrapper(sr=self.sr),
            "valle": lambda: VALLECodecWrapper(sr=self.sr),
        }
        
        # GPU-heavy codecs
        gpu_only = {
            "encodec_meta": lambda: MetaEnCodecWrapper(sr=self.sr),
            "dac": lambda: self._init_dac(),
        }
        
        # If a specific non-random codec is requested, only init that one
        if init_only is not None and init_only != 'random':
            if init_only in lightweight:
                self.codecs[init_only] = lightweight[init_only]()
                return
            elif init_only == 'encodec_meta' and ENCODEC_AVAILABLE:
                try:
                    self.codecs['encodec_meta'] = MetaEnCodecWrapper(sr=self.sr)
                except Exception as e:
                    warnings.warn(f"Failed to load EnCodec: {e}")
                return
            elif init_only == 'dac' and DAC_AVAILABLE:
                try:
                    self.codecs['dac'] = self._init_dac()
                except Exception as e:
                    warnings.warn(f"Failed to load DAC: {e}")
                return
            else:
                warnings.warn(f"Codec '{init_only}' not available.")
                return
        
        # Initialize based on device_type
        # Always initialize lightweight codecs
        for name, factory in lightweight.items():
            self.codecs[name] = factory()
        
        # Only initialize GPU-heavy codecs if device_type is 'gpu'
        if device_type == 'gpu':
            if ENCODEC_AVAILABLE:
                try:
                    self.codecs["encodec_meta"] = MetaEnCodecWrapper(sr=self.sr)
                except Exception as e:
                    warnings.warn(f"Failed to load EnCodec: {e}")
            
            if DAC_AVAILABLE:
                try:
                    self.codecs["dac"] = self._init_dac()
                except Exception as e:
                    warnings.warn(f"Failed to load DAC: {e}")
    
    def _init_dac(self):
        """Helper to initialize DAC codec."""
        if self.sr in [16000, 44100]:
            model_name = "44khz" if self.sr == 44100 else "16khz"
            return DACWrapper(sr=self.sr, model_name=model_name)
        raise ValueError(f"DAC doesn't support sample rate {self.sr}")
    
    def get_available_codecs(self) -> List[str]:
        """Return list of available codecs."""
        return list(self.codecs.keys())
    
    def apply_codec(self, audio: np.ndarray, codec_name: str) -> Optional[np.ndarray]:
        """
        Apply a specific codec to audio.
        
        Args:
            audio: Input audio array
            codec_name: Name of codec to apply
        
        Returns:
            Processed audio or None if codec not available
        """
        if codec_name not in self.codecs:
            warnings.warn(f"Codec '{codec_name}' not available. Available: {self.get_available_codecs()}")
            return None
        
        try:
            codec = self.codecs[codec_name]
            processed = codec.process_audio(audio)
            return processed
        except Exception as e:
            warnings.warn(f"Error applying codec '{codec_name}': {e}")
            return None
    
    def apply_random_codec(self, audio: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply a random codec from available ones.
        
        Args:
            audio: Input audio array
        
        Returns:
            Tuple of (processed audio, codec name used)
        """
        available = self.get_available_codecs()
        if not available:
            warnings.warn("No codecs available")
            return audio, "none"
        
        codec_name = np.random.choice(available)
        processed = self.apply_codec(audio, codec_name)
        
        if processed is None:
            return audio, "none"
        
        return processed, codec_name
    
    def apply_all_codecs(self, audio: np.ndarray) -> dict:
        """
        Apply all available codecs to audio.
        
        Args:
            audio: Input audio array
        
        Returns:
            Dictionary mapping codec names to processed audio
        """
        results = {}
        
        for codec_name in self.get_available_codecs():
            processed = self.apply_codec(audio, codec_name)
            if processed is not None:
                results[codec_name] = processed
        
        return results


# Convenience functions
def apply_neural_codec(audio: np.ndarray, codec_name: str, sr: int = 44100) -> np.ndarray:
    """
    Quick function to apply a single codec to audio.
    
    Args:
        audio: Input audio array
        codec_name: Name of codec
        sr: Sample rate
    
    Returns:
        Processed audio
    """
    confounder = NeuralCodecConfounder(sr=sr)
    processed = confounder.apply_codec(audio, codec_name)
    return processed if processed is not None else audio


def get_available_codecs(sr: int = 44100, device_type: str = 'gpu') -> List[str]:
    """Get list of available codecs.
    
    Args:
        sr: Sample rate for codecs
        device_type: 'cpu' for CPU-only codecs, 'gpu' for all codecs
    
    Returns:
        List of available codec names
    """
    confounder = NeuralCodecConfounder(sr=sr, device_type=device_type)
    return confounder.get_available_codecs()
