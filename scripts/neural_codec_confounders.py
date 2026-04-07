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
        
        print(f"[EnCodec INIT] Starting initialization...")
        t_init = time.time()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EnCodec INIT] Device: {self.device}")
        
        # Load pre-trained EnCodec model
        # Note: EnCodec works best at 24kHz
        self.sr_encodec = 24000
        t_load = time.time()
        self.model = EncodecModel.encodec_model_24khz().to(self.device)
        print(f"[EnCodec INIT] Model loaded in {time.time() - t_load:.4f}s")
        self.model.eval()
        print(f"[EnCodec INIT] Complete in {time.time() - t_init:.4f}s")
    
    def encode(self, audio: np.ndarray) -> list:
        """Encode audio to EncodedFrame objects."""
        t_start = time.time()
        with torch.no_grad():
            # Resample if needed
            t_resample = time.time()
            if self.sr != self.sr_encodec:
                audio_resampled = librosa.resample(
                    audio, orig_sr=self.sr, target_sr=self.sr_encodec
                )
                print(f"  [EnCodec RESAMPLE] {time.time() - t_resample:.4f}s (from {self.sr}Hz to {self.sr_encodec}Hz)")
            else:
                audio_resampled = audio
                print(f"  [EnCodec RESAMPLE] skipped (already {self.sr_encodec}Hz)")
            
            # Convert to tensor and normalize
            t_tensor = time.time()
            audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            print(f"  [EnCodec TO_TENSOR] {time.time() - t_tensor:.4f}s")
            
            # Encode - returns list of EncodedFrame objects
            t_encode = time.time()
            encoded_frames = self.model.encode(audio_tensor)
            print(f"  [EnCodec ENCODE] {time.time() - t_encode:.4f}s (frames: {len(encoded_frames)})")
            
        print(f"  [EnCodec ENCODE TOTAL] {time.time() - t_start:.4f}s")
        return encoded_frames
    
    def decode(self, encoded_frames: list) -> np.ndarray:
        t_start = time.time()
        with torch.no_grad():
            # Decode - takes list of EncodedFrame objects
            t_decode = time.time()
            decoded = self.model.decode(encoded_frames)
            print(f"  [EnCodec DECODE] {time.time() - t_decode:.4f}s")
            
            # Convert back to numpy and resample if needed
            t_numpy = time.time()
            decoded_np = decoded.squeeze().cpu().numpy()
            print(f"  [EnCodec TO_NUMPY] {time.time() - t_numpy:.4f}s")
            
            t_resample = time.time()
            if self.sr != self.sr_encodec:
                decoded_np = librosa.resample(
                    decoded_np, orig_sr=self.sr_encodec, target_sr=self.sr
                )
                print(f"  [EnCodec DECODE_RESAMPLE] {time.time() - t_resample:.4f}s")
            else:
                print(f"  [EnCodec DECODE_RESAMPLE] skipped")
        
        print(f"  [EnCodec DECODE TOTAL] {time.time() - t_start:.4f}s")
        return decoded_np
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        print(f"[EnCodec] Processing audio: len={len(audio)}, sr={self.sr}Hz, bandwidth={self.bandwidth}kbps")
        t_total = time.time()
        encoded_frames = self.encode(audio)
        decoded = self.decode(encoded_frames)
        # Ensure output length matches input
        result = decoded[:len(audio)]
        print(f"[EnCodec] Process complete in {time.time() - t_total:.4f}s")
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
        
        print(f"[DAC INIT] Starting initialization with model={model_name}...")
        t_init = time.time()
        
        self.sr_dac = 44100 if "44" in model_name else 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DAC INIT] Device: {self.device}, sr_dac={self.sr_dac}Hz")
        
        # Initialize DAC model with default architecture
        # (Pre-trained weights would be loaded via DAC.load() if a path is provided)
        t_load = time.time()
        self.model = DAC(sample_rate=self.sr_dac).to(self.device)
        print(f"[DAC INIT] Model loaded in {time.time() - t_load:.4f}s")
        self.model.eval()
        print(f"[DAC INIT] Complete in {time.time() - t_init:.4f}s")
    
    def encode(self, audio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio through DAC."""
        t_start = time.time()
        with torch.no_grad():
            # Resample if needed
            t_resample = time.time()
            if self.sr != self.sr_dac:
                audio_resampled = librosa.resample(
                    audio, orig_sr=self.sr, target_sr=self.sr_dac
                )
                print(f"  [DAC RESAMPLE] {time.time() - t_resample:.4f}s (from {self.sr}Hz to {self.sr_dac}Hz)")
            else:
                audio_resampled = audio
                print(f"  [DAC RESAMPLE] skipped (already {self.sr_dac}Hz)")
            
            # Convert to tensor
            t_tensor = time.time()
            audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            print(f"  [DAC TO_TENSOR] {time.time() - t_tensor:.4f}s")
            
            # Encode
            t_encode = time.time()
            code = self.model.encode(audio_tensor)[0]
            print(f"  [DAC ENCODE] {time.time() - t_encode:.4f}s (code shape: {code.shape})")
            
        print(f"  [DAC ENCODE TOTAL] {time.time() - t_start:.4f}s")
        return code
    
    def decode(self, code: torch.Tensor) -> np.ndarray:
        """Decode from DAC codes."""
        t_start = time.time()
        with torch.no_grad():
            t_device = time.time()
            code = code.to(self.device)
            print(f"  [DAC TO_DEVICE] {time.time() - t_device:.4f}s")
            
            t_decode = time.time()
            decoded = self.model.decode(code)
            print(f"  [DAC DECODE] {time.time() - t_decode:.4f}s")
            
            t_numpy = time.time()
            decoded_np = decoded.squeeze().cpu().numpy()
            print(f"  [DAC TO_NUMPY] {time.time() - t_numpy:.4f}s")
            
            # Resample if needed
            t_resample = time.time()
            if self.sr != self.sr_dac:
                decoded_np = librosa.resample(
                    decoded_np, orig_sr=self.sr_dac, target_sr=self.sr
                )
                print(f"  [DAC DECODE_RESAMPLE] {time.time() - t_resample:.4f}s")
            else:
                print(f"  [DAC DECODE_RESAMPLE] skipped")
        
        print(f"  [DAC DECODE TOTAL] {time.time() - t_start:.4f}s")
        return decoded_np
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full encode-decode cycle."""
        print(f"[DAC] Processing audio: len={len(audio)}, sr={self.sr}Hz, model={self.model_name}")
        t_total = time.time()
        code = self.encode(audio)
        decoded = self.decode(code)
        result = decoded[:len(audio)]
        print(f"[DAC] Process complete in {time.time() - t_total:.4f}s")
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
    
    def __init__(self, sr: int = 44100, init_only: Optional[str] = None):
        """
        Initialize confounder manager.
        
        Args:
            sr: Sample rate for all codecs
            init_only: If specified, only initialize this codec (or all for 'random').
                       None initializes all available codecs.
        """
        print(f"[NeuralCodecConfounder INIT] Starting initialization...")
        t_init = time.time()
        self.sr = sr
        self.codecs = {}
        self._initialize_available_codecs(init_only=init_only)
        print(f"[NeuralCodecConfounder INIT] Complete in {time.time() - t_init:.4f}s")
    
    def _initialize_available_codecs(self, init_only: Optional[str] = None):
        """Initialize only available codecs.
        
        Args:
            init_only: If specified, only initialize this codec.
                       Use 'random' or None to initialize all.
        """
        # Map of codec names to lightweight (librosa-based) constructors
        lightweight = {
            "griffinmel": lambda: GriffinMelCodec(sr=self.sr),
            "audiolm": lambda: AudioLMCodecWrapper(sr=self.sr),
            "valle": lambda: VALLECodecWrapper(sr=self.sr),
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
                    model_name = "44khz" if self.sr == 44100 else "16khz"
                    self.codecs['dac'] = DACWrapper(sr=self.sr, model_name=model_name)
                except Exception as e:
                    warnings.warn(f"Failed to load DAC: {e}")
                return
            else:
                warnings.warn(f"Codec '{init_only}' not available.")
                return
        
        # Otherwise init all available codecs
        for name, factory in lightweight.items():
            self.codecs[name] = factory()
        
        # Conditionally available
        if ENCODEC_AVAILABLE:
            try:
                self.codecs["encodec_meta"] = MetaEnCodecWrapper(sr=self.sr)
            except Exception as e:
                warnings.warn(f"Failed to load EnCodec: {e}")
        
        if DAC_AVAILABLE:
            try:
                if self.sr in [16000, 44100]:
                    model_name = "44khz" if self.sr == 44100 else "16khz"
                    self.codecs["dac"] = DACWrapper(sr=self.sr, model_name=model_name)
            except Exception as e:
                warnings.warn(f"Failed to load DAC: {e}")
    
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
            print(f"[NeuralCodecConfounder] Applying codec '{codec_name}' to audio len={len(audio)}")
            t_start = time.time()
            codec = self.codecs[codec_name]
            processed = codec.process_audio(audio)
            print(f"[NeuralCodecConfounder] Codec '{codec_name}' complete in {time.time() - t_start:.4f}s")
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


def get_available_codecs(sr: int = 44100) -> List[str]:
    """Get list of available codecs."""
    confounder = NeuralCodecConfounder(sr=sr)
    return confounder.get_available_codecs()
