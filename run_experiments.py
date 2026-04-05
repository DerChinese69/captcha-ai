"""
Experiment runner for captcha-ai.

Usage:
    python run_experiments.py

Workflow:
    1. A smoke test runs first (2 epochs, 0.5% of data) using the first
       experiment's config. If it fails, the sequence is aborted.
    2. All experiments in EXPERIMENTS run sequentially.
    3. A summary table is printed at the end.

How to configure:
    - Edit DEFAULTS for settings shared across all experiments.
    - Add one dict per experiment to EXPERIMENTS.
    - Any key in an experiment dict overrides the corresponding default.

Available datasets (data/processed/):
    5Char_100k_Num_grayscale            charset="0123456789"           (100k samples)
    5Char_260k_Alphabet_grayscale       charset="ABCDEFGHIJKLMNOPQRSTUVWXYZ"   (260k samples)
    5Char_360k_AlpNum_grayscale         charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" (360k samples)
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Repo root setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "src").exists():
    REPO_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.runner import run_experiment_sequence  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Defaults — applied to every experiment unless overridden
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Dataset
    "data_dir":        "data/processed/5Char_360k_AlpNum_grayscale",
    "charset":         "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "subset_fraction": 1.0,

    # Data splits
    "train_ratio":  0.75,
    "val_ratio":    0.15,
    "test_ratio":   0.10,
    "random_seed":  69,

    # DataLoader
    "num_workers": 0,
    "pin_memory":  False,
    "drop_last":   False,

    # Optimiser / regularisation
    "weight_decay": 0.0,
    "dropout":      0.0,

    # Scheduler
    "use_scheduler":       False,
    "scheduler_step_size": 4,
    "scheduler_gamma":     0.5,

    # Early stopping
    "val_loss_stop_threshold": 0.15,

    # ViT-specific
    "img_size":    (64, 192),
    "patch_size":  (8, 16),
    "embed_dim":   128,
    "depth":       4,
    "num_heads":   4,
}

# ---------------------------------------------------------------------------
# Experiments — each dict overrides DEFAULTS for that run only.
#
# Required keys per experiment:
#   run_name    — output folder name (unique; appends _01, _02 … if taken)
#   model_name  — "CNN" or "ViT"
#   learning_rate
#   batch_size
#   num_epochs
# ---------------------------------------------------------------------------
EXPERIMENTS = [

    # =========================================================
    # FINAL TRAINING CANDIDATES (leave commented for later)
    # =========================================================
    {
        "run_name":      "cnn_baseline_no_tune_correction",
        "model_name":    "CNN",
         "learning_rate": 3e-4,   # replace with best CNN setting
         "batch_size":    64,     # replace if another batch wins
         "num_epochs":    40,
         "dropout":       0.3,    # replace with best CNN setting
         "weight_decay":  1e-5,   # replace with best CNN setting
         "use_scheduler": True,
         "scheduler_step_size": 5,
         "scheduler_gamma": 0.5,
         "val_loss_stop_threshold": 0.15,
     },
     {
    "run_name": "vit_last_recovery",
    "model_name": "ViT",

    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 30,

    "weight_decay": 1e-4,
    "dropout": 0.1,

    "use_scheduler": True,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.5,

    "val_loss_stop_threshold": 0.15,

    "img_size": (64, 192),
    "patch_size": (8, 8),
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 6,
}
]

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment_sequence(EXPERIMENTS, DEFAULTS, EXPERIMENTS_DIR)
