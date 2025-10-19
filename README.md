# Music Classifier

Repository scaffold for a music authenticity classifier distinguishing real vs generated music.

Structure:

- data/: training and test datasets (real and synthetic)
- models/: saved model weights and checkpoints
- scripts/: training, preprocessing, evaluation utilities
- notebooks/: exploratory analysis
- results/: logs, metrics, and figures

Getting started

1. Create a Python virtual environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Place audio files into the appropriate `data/` subfolders.
3. Run `scripts/build_manifest.py` to create dataset manifests.
4. Use `scripts/train_cnn.py` to train a baseline CNN model.

License: MIT
