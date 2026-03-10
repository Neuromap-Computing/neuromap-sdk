# Prosthesis Cochlear-Implant Denoising

This example demonstrates using the Neuromap SDK to train an SNN for
audio denoising in a cochlear-implant context.

## Setup

```bash
# From the neuro-sim/ root
pip install -e .
pip install -r examples/prosthesis/requirements.txt
```

## Usage

```bash
cd examples/prosthesis

# 1. Bootstrap raw datasets (downloads LJSpeech + UrbanSound8K)
python run.py bootstrap

# 2. Generate train/val/test clean-noisy pairs
python run.py generate \
    --clean-dir data/prosthesis/raw/clean_source \
    --noise-dir data/prosthesis/raw/noise_source \
    --out-dir data/prosthesis/generated

# 3. Train
python run.py train --data-dir data/prosthesis/generated

# 4. Export to .nmap
python run.py export --checkpoint data/prosthesis/training_runs/best_prosthesis_snn_cls.pt
```

## Pipeline

1. **Bootstrap** — downloads LJSpeech (clean speech) and UrbanSound8K
   (environmental noise) into a local staging directory.
2. **Generate** — creates deterministic clean/noisy pairs at various SNR
   levels and extracts cochlear (gammatone) features.
3. **Train** — trains a DynamicSNN via the Neuromap `Trainer`, using a
   composite MSE + L1 loss on frame-level features.
4. **Export** — quantizes the trained model and packages it as a `.nmap`
   archive ready for chip deployment.
