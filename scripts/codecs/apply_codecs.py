#!/usr/bin/env python3
"""
Script to apply neural codec confounders to audio files.

This script processes audio files through various neural codecs to introduce
compression artifacts and other codec-specific distortions to the audio.

Usage:
    # Apply a specific codec to a directory
    python apply_codecs.py --input-dir data/real_songs/ --codec encodec --output-dir data/real_songs_encodec/
    
    # Apply all available codecs
    python apply_codecs.py --input-dir data/real_songs/ --codec all --output-dir data/real_songs_all_codecs/
    
    # List available codecs
    python apply_codecs.py --list-codecs
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from scripts.utils.neural_codec_confounders import NeuralCodecConfounder, get_available_codecs


class CodecApplier:
    """Applies neural codec confounders to audio files."""
    
    def __init__(self, sr=44100):
        """
        Initialize codec applier.
        
        Args:
            sr: Sample rate for all audio processing
        """
        self.sr = sr
        self.confounder = NeuralCodecConfounder(sr=sr)
        self.available_codecs = self.confounder.get_available_codecs()
    
    def apply_codec_to_file(self, audio_path, codec_name, output_path=None):
        """
        Apply codec to a single audio file.
        
        Args:
            audio_path: Path to input audio file
            codec_name: Name of codec to apply
            output_path: Path to save output (optional)
        
        Returns:
            Processed audio array, or None if failed
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Apply codec
            processed = self.confounder.apply_codec(audio, codec_name)
            
            if processed is None:
                print(f"❌ Failed to apply codec '{codec_name}' to {audio_path}")
                return None
            
            # Save if output path specified
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:  # Only create directory if path has a directory component
                    os.makedirs(output_dir, exist_ok=True)
                sf.write(output_path, processed, self.sr)
                print(f"✅ {audio_path} → {output_path}")
            
            return processed
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            return None
    
    def apply_codec_to_directory(self, input_dir, codec_name, output_dir, 
                                 audio_extensions=(".wav", ".mp3", ".flac", ".ogg")):
        """
        Apply codec to all audio files in a directory.
        
        Args:
            input_dir: Path to input directory
            codec_name: Name of codec to apply (or "all")
            output_dir: Path to output directory
            audio_extensions: Tuple of audio file extensions
        
        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return None
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
        
        if not audio_files:
            print(f"⚠️  No audio files found in {input_dir}")
            return None
        
        print(f"📊 Found {len(audio_files)} audio files")
        
        # Apply codec to all files
        results = {
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "codec": codec_name,
            "sr": self.sr,
            "total_files": len(audio_files),
            "successful": 0,
            "failed": 0,
            "files": []
        }
        
        if codec_name == "all":
            # Apply all available codecs
            for single_codec in self.available_codecs:
                print(f"\n🎵 Applying codec: {single_codec}")
                codec_output_dir = output_path / single_codec
                
                for audio_file in tqdm(audio_files, desc=f"Processing with {single_codec}"):
                    # Preserve relative directory structure
                    relative_path = audio_file.relative_to(input_path)
                    output_file = codec_output_dir / relative_path
                    
                    processed = self.apply_codec_to_file(str(audio_file), single_codec, str(output_file))
                    
                    if processed is not None:
                        results["successful"] += 1
                        results["files"].append({
                            "input": str(audio_file),
                            "output": str(output_file),
                            "codec": single_codec,
                            "status": "success"
                        })
                    else:
                        results["failed"] += 1
                        results["files"].append({
                            "input": str(audio_file),
                            "codec": single_codec,
                            "status": "failed"
                        })
        else:
            # Apply single codec
            for audio_file in tqdm(audio_files, desc=f"Processing with {codec_name}"):
                # Preserve relative directory structure
                relative_path = audio_file.relative_to(input_path)
                output_file = output_path / relative_path
                
                processed = self.apply_codec_to_file(str(audio_file), codec_name, str(output_file))
                
                if processed is not None:
                    results["successful"] += 1
                    results["files"].append({
                        "input": str(audio_file),
                        "output": str(output_file),
                        "status": "success"
                    })
                else:
                    results["failed"] += 1
                    results["files"].append({
                        "input": str(audio_file),
                        "status": "failed"
                    })
        
        # Save results
        results_file = output_path / "processing_results.json"
        os.makedirs(output_path, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results: {results['successful']} successful, {results['failed']} failed")
        print(f"   Results saved to: {results_file}")
        
        return results


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Apply neural codec confounders to audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available codecs
  python apply_codecs.py --list-codecs
  
  # Apply a specific codec to a directory
  python apply_codecs.py --input-dir data/real_songs/ --codec encodec --output-dir data/codecs/encodec/
  
  # Apply all available codecs
  python apply_codecs.py --input-dir data/real_songs/ --codec all --output-dir data/codecs/
  
  # Apply codec to a single file
  python apply_codecs.py --input-file data/sample.wav --codec griffinmel --output-file data/sample_griffinmel.wav
        """
    )
    
    parser.add_argument("--list-codecs", action="store_true",
                       help="List available codecs and exit")
    
    parser.add_argument("--input-dir", type=str,
                       help="Input directory containing audio files")
    
    parser.add_argument("--input-file", type=str,
                       help="Input audio file")
    
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for processed files")
    
    parser.add_argument("--output-file", type=str,
                       help="Output audio file")
    
    parser.add_argument("--codec", type=str, default="griffinmel",
                       help="Codec to apply (default: griffinmel). Use 'all' to apply all available codecs")
    
    parser.add_argument("--sr", type=int, default=44100,
                       help="Sample rate for audio processing (default: 44100)")
    
    args = parser.parse_args()
    
    # List available codecs
    if args.list_codecs:
        applier = CodecApplier(sr=args.sr)
        print("📋 Available neural codec confounders:")
        print()
        codecs_info = {
            "encodec": "Meta's neural codec (24 kHz optimal)",
            "dac": "Descript Audio Codec (high quality)",
            "audiolm": "Google's AudioLM codec tokenizer",
            "valle": "Microsoft's VALL-E codec (hierarchical quantization)",
            "griffinmel": "Griffin-Lim mel-spectrogram codec (lightweight)",
        }
        
        for codec in applier.available_codecs:
            info = codecs_info.get(codec, "")
            status = "✅ Available" if codec in applier.available_codecs else "❌ Not installed"
            print(f"  {codec:20s} {info:50s} {status}")
        
        return
    
    applier = CodecApplier(sr=args.sr)
    
    # Process single file
    if args.input_file:
        if not args.output_file:
            print("❌ --output-file required when using --input-file")
            sys.exit(1)
        
        applier.apply_codec_to_file(args.input_file, args.codec, args.output_file)
    
    # Process directory
    elif args.input_dir:
        if not args.output_dir:
            print("❌ --output-dir required when using --input-dir")
            sys.exit(1)
        
        applier.apply_codec_to_directory(args.input_dir, args.codec, args.output_dir)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
