# Alphanumerical CAPTCHA Solver (CNN + ViT)

A deep learning project exploring Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for solving distorted alphanumerical CAPTCHA images.

---

## Overview

This project investigates how different deep learning architectures perform on fixed-length CAPTCHA recognition tasks.

The main objectives are:
- build a working baseline CNN model  
- establish a reproducible training pipeline  
- experiment with alternative architectures (ViT, modified CNNs)  
- compare performance across models  

The focus is on **practical experimentation and iterative improvement**, not production deployment.

---

## Requirements

- Python 3.11 recommended  
- NVIDIA GPU with CUDA (strongly recommended for training)

---

## Setup (Windows + NVIDIA GPU)

### 1. Clone the repository

```cmd
git clone https://github.com/DerChinese69/captcha-ai.git
cd captcha-ai
```

### 2. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA support

PyTorch must be installed with CUDA before the rest of the dependencies.  
Go to **https://pytorch.org/get-started/locally** and select your OS, CUDA version, and pip to get the exact install command for your system. It will look like:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Replace `cu124` with the CUDA version that matches your driver (e.g. `cu121`, `cu118`).

### 4. Install remaining dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

pip will skip torch/torchvision/torchaudio since they are already installed.

To also run the Jupyter notebooks:

```cmd
pip install -r requirements-dev.txt
```

---

## Setup (macOS / Linux)

```bash
git clone https://github.com/DerChinese69/captcha-ai.git
cd captcha-ai

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# For notebooks:
# pip install -r requirements-dev.txt
```

On Apple Silicon (MPS) or CPU-only Linux, the standard install is sufficient.

---

## Verify Installation

Run the environment check to confirm all packages are importable:

```cmd
python check_setup.py
```

All packages should return `[OK]`.

Run the smoke test for a full end-to-end check:

```cmd
python smoke_test.py
```

This verifies imports, device selection, dataloader creation, and a forward pass through both the CNN and ViT models. If no processed dataset exists yet under `data/processed/`, it skips the data-dependent steps and still confirms the environment is ready.

---

## Project Structure

```
captcha-ai/
├── src/                        # Core source code
│   ├── dataset/                # Dataloader and dataset class
│   ├── models/                 # CaptchaCNN and CaptchaViT architectures
│   ├── training/               # Training engine, runner, setup, evaluation helpers
│   ├── evaluation/             # Full evaluation utilities (eval_utils.py)
│   ├── preprocessing/          # Grayscale preprocessing script
│   └── generator/              # Balanced CAPTCHA order generation
├── data/
│   ├── raw/                    # Raw unprocessed images
│   ├── processed/              # Preprocessed datasets (expected by training)
│   └── orders/                 # Generated order files
├── experiments/                # Output directory for training runs
├── evaluation/                 # Output directory for evaluation runs
├── notebooks/                  # Jupyter notebooks for exploration
├── run_experiments.py          # Main training runner (edit to configure experiments)
├── run_evaluation.py           # Full evaluation pipeline
├── compare_experiments.py      # Rank experiments by validation accuracy
├── check_setup.py              # Verify all dependencies are installed
├── smoke_test.py               # End-to-end sanity check for a new machine
└── requirements.txt
```

---

## Data

The dataset consists of synthetic CAPTCHA images generated using:  
https://github.com/Gregwar/Captcha  

Expected structure:

```
data/
└── processed/
    └── <dataset_name>/
        ├── *.png (images)
        └── ground_truth_index.csv
```

Available datasets (place under `data/processed/`):

| Dataset | Samples | Charset |
|---|---|---|
| `5Char_100k_Num_grayscale` | 100k | `0123456789` |
| `5Char_260k_Alp_grayscale` | 260k | `A-Z` |
| `5Char_360k_AlpNum_grayscale` | 360k | `0-9A-Z` |

### Preprocessing

- Grayscale conversion  
- Resizing to consistent resolution  
- Normalization  

Run preprocessing on raw images:

```cmd
python src/preprocessing/grayscale_preprocess_captcha_images.py --workers 4
```

### Generator Usage

To create a new balanced order for captcha rendering, run:

```cmd
python src/generator/generate_order.py ^
  --classes "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ^
  --length 5 ^
  --target-per-class-per-position 1000
```

`--classes` is the exact character set to sample from, `--length` is the number of characters in each captcha label, and `--target-per-class-per-position` is the target count for each class at each character position.

---

## Workflow

### Training

Edit the `EXPERIMENTS` list in `run_experiments.py` to configure your runs, then:

```cmd
python run_experiments.py
```

This runs a smoke test first (2 epochs, 0.5% of data), then all configured experiments sequentially. Results are saved under `experiments/<run_name>/`.

Each run saves:
- `best_model.pt` — best checkpoint by validation sequence accuracy
- `last_model.pt` — final checkpoint
- `config.json` — full configuration
- `metadata.json` — compact metrics summary
- `training_history.json` — per-epoch metrics
- Training curve plots (loss, accuracy, per-position)

### Evaluation

Edit the `EVALUATIONS` list in `run_evaluation.py` to point at experiment folders, then:

```cmd
python run_evaluation.py
```

Results are saved under `evaluation/<run_name>/`, including confusion matrices, qualitative examples, saliency maps, confidence analysis, and more.

### Compare Experiments

To rank all completed experiments by validation sequence accuracy:

```cmd
python compare_experiments.py
```

Outputs a ranked table and saves results as JSON and CSV.

### Quick Accuracy Check

To get a fast accuracy report on one or more experiment runs without running the full evaluation suite:

```cmd
python src/training/get_accuracy.py experiments/<run_name>
```

Multiple runs can be compared in one command:

```cmd
python src/training/get_accuracy.py experiments/cnn_alphabet_final experiments/vit_alphabet_final
```

This is useful for quick spot-checks. For full analysis (confusion matrices, saliency maps, etc.) use `run_evaluation.py` instead.

---

## Approach

### Baseline Model: CNN

The initial model uses a convolutional architecture to extract spatial features from CAPTCHA images.

Pipeline:
1. Image preprocessing  
2. Convolutional feature extraction  
3. Feature compression  
4. Character classification  

### Vision Transformer (ViT)

A compact transformer model (`SmallCaptchaViT`) was implemented as an alternative to the CNN baseline, using patch-based image embeddings and multi-head attention.

---

## Project Goals

- Build a working CNN-based CAPTCHA solver  
- Create a reproducible training and evaluation pipeline  
- Experiment with:
  - CNN architecture improvements  
  - Small Vision Transformer (ViT)  
  - Fine-tuned pretrained ViT  

- Compare models based on:
  - accuracy  
  - training stability  
  - generalization  

---

## Notes

AI-assisted tools were used to support development (debugging, refactoring, iteration speed).  
All architectural and methodological decisions were made by the project author.

---

## License

This repository is intended for educational and research purposes.
