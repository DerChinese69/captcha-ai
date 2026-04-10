# CAPTCHA Recognition — CNN vs Vision Transformer

An end-to-end deep learning project for solving 5-character alphanumeric CAPTCHA images.
Compares a custom CNN baseline against a compact Vision Transformer (ViT) on
synthetic datasets ranging from 10k to 360k images.

---

## Results at a glance

| Model | Dataset (36 classes, 5 chars) | Test seq acc | Test char acc |
|-------|-------------------------------|:------------:|:-------------:|
| CNN   | 5Char_360k_AlpNum             | **91.5%**    | **98.0%**     |
| ViT   | 5Char_360k_AlpNum             | **91.6%**    | **98.1%**     |

Both trained from scratch on a balanced synthetic dataset.
Full metrics, confusion matrices, saliency maps, and latent-space visualizations
are produced by `run_evaluation.py`.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Pipeline](#pipeline)
3. [Models](#models)
4. [Repository structure](#repository-structure)
5. [Setup](#setup)
6. [Dataset](#dataset)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Compare experiments](#compare-experiments)
10. [Top models](#top-models)
11. [Example workflow](#example-workflow)
12. [Student-friendly explanation](#student-friendly-explanation)
13. [Known limitations & future work](#known-limitations--future-work)
14. [License](#license)

---

## Project overview

This project investigates how different deep learning architectures perform on
**fixed-length CAPTCHA recognition** — predicting all 5 characters from a single
distorted image. It is not designed for production use; the focus is on
practical experimentation, reproducibility, and clean code.

**Key goals:**
- Build a working CNN baseline for CAPTCHA recognition
- Implement a compact ViT as an architectural alternative
- Create a reproducible experiment + evaluation pipeline
- Document the design tradeoffs between the two approaches

---

## Pipeline

```
[PHP generator] ──► data/raw/          raw CAPTCHA images (ignored by git)
                        │
                [Preprocessing] ──► data/processed/    grayscale-normalized images
                        │
                [run_experiments.py] ──► experiments/<run>/   checkpoints + history
                        │
                [run_evaluation.py]  ──► evaluation/<run>/    metrics + plots
                        │
                [compare_experiments.py] ──► evaluation/experiment_comparison.{json,csv}
```

Each stage is independent and runnable on its own.

---

## Models

### CaptchaCNN

A convolutional network designed specifically for fixed-length CAPTCHA decoding.

**Architecture:**
```
Input [B, 1, 64, 192]
  → 4 conv blocks (32→32→64→64→128→256 channels, BN + ReLU + MaxPool)
  → AdaptiveAvgPool2d((1, 5))     # forces one feature vector per character slot
  → Flatten → Linear(1280, 512) → Dropout → Linear(512, 5×num_classes)
  → Reshape [B, 5, num_classes]
```

The key design decision is `AdaptiveAvgPool2d((1, 5))`: it collapses the spatial
feature map into exactly 5 horizontal slots — one per character — so the
classifier head can predict each character independently.  A `WidthPad` layer
ensures the width is divisible by 5 for MPS compatibility.

### SmallCaptchaViT

A compact Vision Transformer using patch-based embeddings and slot-aware pooling.

**Architecture:**
```
Input [B, 1, 64, 192]
  → PatchEmbedding (conv stride=patch_size) → [B, 96, 128]   (96 patches)
  → + positional embedding
  → 4× TransformerBlock (multi-head attention + MLP + LayerNorm)
  → _slot_pool: reshape patches to 2-D grid, average over height,
                interpolate along width → [B, 5, 128]          (5 slots)
  → Linear head → [B, 5, num_classes]
```

`SmallCaptchaViTA` (Phase A) uses global mean-pooling instead — a simpler
baseline that broadcasts one vector to all slots.  `SmallCaptchaViT` (Phase B,
the default) uses true per-slot pooling for better character-level structure.

**Design rationale — CNN vs ViT:**

| | CNN | ViT |
|---|---|---|
| Inductive bias | Strong spatial locality | Weaker; learns from data |
| Per-character pooling | Exact (AdaptiveAvgPool2d) | Approximate (interpolation) |
| Training data needed | Lower | Higher |
| Interpretability | Saliency maps straightforward | Attention-based |
| Training speed | ~2× faster per epoch | Slower; benefits from GPU |

At ~360k samples both converge to similar accuracy (~91.5% sequence).

---

## Repository structure

```
captcha-ai/
│
├── src/                            Core source code
│   ├── models/
│   │   ├── CaptchaCNN.py           CNN architecture (CaptchaCNN)
│   │   └── CaptchaViT.py           ViT architectures (SmallCaptchaViTA, SmallCaptchaViT)
│   ├── training/
│   │   ├── engine.py               train_one_epoch / validate_one_epoch
│   │   ├── runner.py               run_one_experiment / run_experiment_sequence
│   │   ├── setup.py                initialize_training_run (model + optimizer + history)
│   │   ├── evaluate.py             Training curve plotting (linear + log scale)
│   │   └── get_accuracy.py         Quick CLI accuracy check on a checkpoint
│   ├── evaluation/
│   │   └── eval_utils.py           Full eval suite (metrics, confusion, saliency, t-SNE)
│   ├── dataset/
│   │   └── dataloader.py           CaptchaDataset + create_dataloaders
│   ├── preprocessing/
│   │   └── grayscale_preprocess_captcha_images.py   Raw→processed preprocessing
│   └── generator/
│       ├── generate_order.py       Balanced label order generator (Python)
│       └── render_captchas.php     CAPTCHA renderer (PHP / Gregwar)
│
├── data/                           All data directories (contents gitignored)
│   ├── raw/                        Raw rendered images (download from Kaggle)
│   ├── processed/                  Grayscale-normalized images (run preprocessing)
│   └── orders/                     Order manifests (generated locally)
│
├── experiments/                    Training outputs (gitignored; see top_models/)
├── evaluation/                     Evaluation outputs (gitignored; see top_models/)
│
├── top_models/                     Best model metadata committed to git
│   ├── cnn/training/               config.json, metadata.json, training_history.json
│   ├── cnn/evaluation/             metrics_test.json, error breakdown, per-position acc
│   ├── vit/training/
│   └── vit/evaluation/
│
├── notebooks/
│   ├── Core Code/                  Training_Notebook.ipynb, Model_Analysis.ipynb
│   └── Development/                Exploration notebooks (testbeds)
│
├── run_experiments.py              Main training entry point
├── run_evaluation.py               Main evaluation entry point
├── compare_experiments.py          Rank runs by validation accuracy
├── check_setup.py                  Verify all dependencies are importable
├── smoke_test.py                   End-to-end sanity check
├── requirements.txt                Python dependencies
├── requirements-dev.txt            + jupyter/ipykernel
├── composer.json                   PHP dependencies (Gregwar/Captcha renderer)
└── LICENSE
```

---

## Setup

### Requirements

- Python 3.11+ recommended (developed on 3.14)
- NVIDIA GPU with CUDA strongly recommended for ViT training
- Apple Silicon (MPS) works for CNN and smaller datasets

### Windows + NVIDIA GPU

```cmd
git clone https://github.com/DerChinese69/captcha-ai.git
cd captcha-ai

python -m venv .venv
.venv\Scripts\activate

:: Install PyTorch first — pick your CUDA version at https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/DerChinese69/captcha-ai.git
cd captcha-ai

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

On Apple Silicon, standard PyTorch (no CUDA flag) is sufficient; MPS is used automatically.

### Verify

```bash
python check_setup.py   # all packages importable
python smoke_test.py    # end-to-end forward pass with auto-discovered dataset
```

### Quick smoke test (no download required)

A 30-image synthetic dataset (`data/processed/Smoke_grayscale/`) is committed
to the repository so the full pipeline can be tested immediately after cloning.

```bash
# 1. Verify imports and model forward pass
python smoke_test.py

# 2. Run a 2-epoch CNN training on the smoke dataset
python run_experiments.py
# → creates experiments/CNN_test/

# 3. Run the evaluation suite on that checkpoint
python run_evaluation.py
# → creates evaluation/CNN_test/

# 4. Compare runs
python compare_experiments.py
```

The smoke dataset uses real image dimensions (64×192) and a valid 5-digit
label format, so every component of the pipeline exercises real code paths.
Accuracy on 30 synthetic images is meaningless — this only confirms the
pipeline is wired up correctly.

---

## Dataset

### Download from Kaggle

The datasets used in this project are publicly available on Kaggle:

> ### Datasets

- **Numeric (5-character, class-balanced)**  
  https://www.kaggle.com/datasets/derchinese69/captcha-dataset-5char-numeric-class-balanced  

- **Alphabetic (5-character, class-balanced)**  
  https://www.kaggle.com/datasets/derchinese69/captcha-dataset-5char-alphabet-class-balanced  

- **Alphanumeric (5-character, class-balanced)**  
  https://www.kaggle.com/datasets/derchinese69/5-char-alphanumeric-captcha-class-balanced 


Place the downloaded folders under `data/processed/` so they match this layout:

```
data/processed/
├── 5Char_100k_Num_grayscale/
│   ├── 000001.png
│   ├── ...
│   └── ground_truth_index.csv      filename,label
├── 5Char_260k_Alp_grayscale/
├── 5Char_360k_AlpNum_grayscale/
└── unseen_test_random_data/        (optional held-out set)
```

**Available datasets:**

| Name | Samples | Charset | Classes |
|------|---------|---------|---------|
| `5Char_100k_Num_grayscale`   | 100 000 | `0–9`    | 10 |
| `5Char_10k_Num_grayscale`    | 10 000  | `0–9`    | 10 |
| `5Char_260k_Alp_grayscale`   | 260 000 | `A–Z`    | 26 |
| `5Char_26k_Alp_grayscale`    | 26 000  | `A–Z`    | 26 |
| `5Char_360k_AlpNum_grayscale`| 360 000 | `0–9A–Z` | 36 |
| `5Char_36k_AlpNum_grayscale` | 36 000  | `0–9A–Z` | 36 |

Each dataset folder contains grayscale PNG images and a `ground_truth_index.csv`
with columns `filename` and `label`.

### Generate your own dataset

The repository includes a full data generation pipeline:

**Step 1 — Install PHP dependencies (Gregwar/Captcha renderer)**

```bash
composer install   # requires PHP + Composer
```

Library: [Gregwar/Captcha](https://github.com/Gregwar/Captcha.git)

**Step 2 — Generate a balanced order manifest**

```bash
python src/generator/generate_order.py \
  --classes "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
  --length 5 \
  --target-per-class-per-position 1000
# → writes data/orders/order_000X/
```

This produces a CSV of `(sample_id, label)` pairs with exact per-position class
balance: every class appears equally often at every character position.

**Step 3 — Render CAPTCHA images**

```bash
php src/generator/render_captchas.php data/orders/order_000X MyDatasetName
# → renders to data/raw/MyDatasetName/ with ground_truth_index.csv
```

**Step 4 — Preprocess to grayscale**

```bash
python src/preprocessing/grayscale_image_preprocessing.py --workers 4
# → data/raw/MyDatasetName/ → data/processed/MyDatasetName_grayscale/
```

Preprocessing inverts images where characters are lighter than the background
(polarity standardisation), so all images are dark-on-light.

### Dataset generation — same repo vs separate repo

**Recommendation: use the Kaggle dataset directly.** The PHP generator is
included for transparency and reproducibility, but most users do not need it.
If you want to extend or fork the generator into its own project, the relevant
files are:

```
src/generator/generate_order.py
src/generator/render_captchas.php
composer.json / composer.lock
vendor/  (PHP dependencies, gitignored)
```

---

## Training

Edit `EXPERIMENTS` in `run_experiments.py`, then:

```bash
python run_experiments.py
```

**What happens:**
1. A smoke test runs first (2 epochs, 0.5% of data) to catch config errors early.
2. All experiments in `EXPERIMENTS` run sequentially.
3. A summary table is printed at the end.

**Configuring experiments:**

```python
DEFAULTS = {
    "data_dir":        "data/processed/5Char_100k_Num_grayscale",
    "charset":         "0123456789",
    "train_ratio":     0.75,
    "val_ratio":       0.15,
    "test_ratio":      0.10,
    "random_seed":     69,
    "weight_decay":    0.0,
    "dropout":         0.0,
    "use_scheduler":   False,
    # ... (see run_experiments.py for all keys)
}

EXPERIMENTS = [
    {
        "run_name":      "cnn_numeric_baseline",
        "model_name":    "CNN",          # "CNN" or "ViT"
        "learning_rate": 1e-3,
        "batch_size":    128,
        "num_epochs":    20,
    },
]
```

Any key in an experiment dict overrides the corresponding default.

**Output per run** (`experiments/<run_name>/`):

| File | Contents |
|------|----------|
| `best_model.pt` | Best checkpoint by val sequence accuracy |
| `last_model.pt` | Final checkpoint |
| `config.json` | Full configuration (architecture + hyperparams) |
| `metadata.json` | Compact summary (loss, accuracy, epochs) |
| `training_history.json` | Per-epoch metrics |
| `loss_curves.png` | Train/val loss |
| `char_acc_curves.png` | Character accuracy |
| `seq_acc_curves.png` | Sequence accuracy |
| `val_pos_acc_curves.png` | Per-position accuracy |
| `log_*.png` | Log-scale versions of the above |

**Common failure points:**

- *No processed dataset found* → run preprocessing or download from Kaggle first
- *Windows multiprocessing error* → `num_workers` is automatically set to 0 on Windows
- *MPS out of memory* → reduce `batch_size` or use `num_workers: 0`
- *Charset mismatch* → ensure `charset` matches the characters in your dataset's labels

**Quick accuracy check without full evaluation:**

```bash
python src/training/get_accuracy.py experiments/<run_name>
```

---

## Evaluation

Edit `EVALUATIONS` in `run_evaluation.py` to point at one or more experiment
folders, then:

```bash
python run_evaluation.py
```

The script reads `config.json` from the experiment folder, reconstructs the
exact same test split used during training (same seed, same ratios), loads the
best checkpoint, and runs the full evaluation suite.

**Output per run** (`evaluation/<run_name>/`):

| File | Contents |
|------|----------|
| `metrics_test.json` | Sequence & character accuracy, counts |
| `error_breakdown_test.json` | Error distribution by number of wrong chars |
| `predictions_test.csv` | Per-sample predictions vs ground truth |
| `confusion_matrix_test.png` | Character-level confusion matrix |
| `qualitative_examples_test.png` | Correct and incorrect predictions visualized |
| `saliency_maps_test.png` | Gradient saliency overlay on input images |
| `per_position_accuracy_test.json/png` | Accuracy at each character slot |
| `sequence_error_distribution_test.json/png` | Distribution of error counts per sample |
| `top_confusions_test.csv/png` | Most frequent character confusion pairs |
| `confidence_analysis_*.png` | Confidence score histograms |
| `latent_space_*.png` | t-SNE of per-character embeddings |

Add `unseen_data_dir` to evaluate on a second held-out dataset:

```python
EVALUATIONS = [
    {
        "experiment_dir":  "experiments/cnn_alpnum",
        "unseen_data_dir": "data/processed/unseen_test_random_data",
        "unseen_charset":  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    },
]
```

---

## Compare experiments

```bash
# Rank all runs by validation sequence accuracy (default)
python compare_experiments.py

# Sort by test evaluation accuracy
python compare_experiments.py --sort_by eval_seq_acc

# Split output by model family
python compare_experiments.py --group_by_model
```

Writes `evaluation/experiment_comparison.json` and `.csv`.

---

## Top models

The best trained models are documented in [top_models/](top_models/README.md).
This folder contains the full training configuration, per-epoch history, and
evaluation metrics — everything needed to understand and reproduce the results.

Model weights (`best_model.pt`) are not committed to git.
To use a checkpoint, re-train from the config or contact the author.

---

## Example workflow

A minimal end-to-end run using the small numeric dataset:

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# 2. Verify setup
python smoke_test.py

# 3. Place dataset under data/processed/5Char_10k_Num_grayscale/

# 4. Run a quick CNN test (1% of data, 5 epochs)
#    Edit EXPERIMENTS in run_experiments.py:
#      "run_name":        "cnn_quick_test",
#      "model_name":      "CNN",
#      "data_dir":        "data/processed/5Char_10k_Num_grayscale",
#      "charset":         "0123456789",
#      "learning_rate":   1e-3,
#      "batch_size":      64,
#      "num_epochs":      5,
#      "subset_fraction": 0.01,
python run_experiments.py

# 5. Quick accuracy check
python src/training/get_accuracy.py experiments/cnn_quick_test

# 6. Full evaluation
#    Edit EVALUATIONS in run_evaluation.py:
#      {"experiment_dir": "experiments/cnn_quick_test"}
python run_evaluation.py

# 7. Compare runs
python compare_experiments.py
```

**Minimal CNN test command** (verifies pipeline works end-to-end):

```bash
# Requires data/processed/5Char_10k_Num_grayscale/ to exist
python run_experiments.py
# (with run_name="CNN_test", model_name="CNN", subset_fraction=0.01 in EXPERIMENTS)
```
---

## Known limitations & future work

**Limitations:**
- Fixed sequence length (5 characters). Variable-length CAPTCHA (e.g., using CTC loss) is not implemented.
- No data augmentation during training. Augmentation (rotations, noise) could improve robustness.
- The ViT uses linear interpolation for slot pooling, which is an approximation. Learned slot queries (cross-attention) would be more principled.
- Evaluation on truly "unseen" styles shows the model does not generalise well beyond its training distribution (0% sequence accuracy on out-of-distribution CAPTCHAs).
- Models were trained on one machine (MPS / CUDA); batch sizes may need tuning for other hardware.

**Future improvements:**
- Variable-length recognition with CTC or attention decoder
- Data augmentation (noise, distortion, colour jitter)
- Cross-architecture attention: applying attention heads to the CNN feature map
- Pretrained ViT backbone (e.g., ViT-Tiny from timm) with fine-tuning
- Hyperparameter search (Optuna or similar)
- REST API or Gradio demo for interactive inference

---

## AI-assisted development

AI tools (e.g. Claude) were used to assist with development tasks such as debugging, refactoring, and accelerating iteration. All design, architectural, and experimental decisions were made by the project author.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.
