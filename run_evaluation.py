"""
run_evaluation.py — Full evaluation pipeline for trained captcha-ai models.

Mirrors the style of run_experiments.py:
  - Edit DEFAULTS for settings shared across all evaluations.
  - Add one dict per run to EVALUATIONS.
  - Any key in an evaluation dict overrides the corresponding default.

For each entry, the script:
  1. Reads config.json from the experiment folder (model arch, splits, charset …)
  2. Loads best_model.pt (falls back to last_model.pt if missing)
  3. Reconstructs the same test split used during training
  4. Runs the full evaluation suite and saves everything under evaluation/<run_name>/

Output layout per run:
    evaluation/<run_name>/
        metrics_test.json
        error_breakdown_test.json
        predictions_test.csv
        confusion_matrix_test.png
        qualitative_examples_test.png
        saliency_maps_test.png
        (+ *_unseen.* variants if unseen_data_dir is configured)

Usage:
    python run_evaluation.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — same pattern as run_experiments.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "src").exists():
    REPO_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------
EVALUATION_DIR = REPO_ROOT / "evaluation"

# ---------------------------------------------------------------------------
# Defaults — applied to every evaluation unless overridden.
#
# Model architecture, dataset path, charset, and split ratios are read
# automatically from each experiment's config.json.  You only need to
# set overrides here if you want to deviate from the training config.
# ---------------------------------------------------------------------------
DEFAULTS = {
    # DataLoader settings (can be larger than training batch_size)
    "batch_size":  64,
    "num_workers": 0,
    "pin_memory":  False,

    # Output root (relative to repo root)
    "output_root": "evaluation",

    # How many examples to collect for qualitative / saliency outputs
    "num_correct_examples":   5,
    "num_incorrect_examples": 5,
    "num_saliency_samples":   5,

    # Latent-space visualization
    "latent_max_chars":  8000,          # total embeddings to collect (should be >= max tsne size)
    "latent_run_tsne":   True,          # generate t-SNE plots
    "latent_tsne_sizes": [2000, 5000, 8000],  # one plot per size

    # Device: None = auto-detect (CUDA > MPS > CPU)
    # Set to "cpu", "cuda", or "mps" to force a specific device.
    "device": None,
}

# ---------------------------------------------------------------------------
# Evaluations — each dict overrides DEFAULTS for that run only.
#
# Required key:
#   experiment_dir  — path to the experiment folder (relative to repo root or absolute)
#
# Optional overrides (all read from config.json by default):
#   data_dir          — override the dataset path
#   charset           — override the character set
#   device            — override device
#
# Optional unseen-data evaluation:
#   unseen_data_dir   — path to a second dataset to evaluate on
#   unseen_charset    — charset for that dataset (defaults to main charset)
# ---------------------------------------------------------------------------
EVALUATIONS = [
    {
        "experiment_dir": "experiments/Alphanumerical/CNN_AlpNum",
        # Examples of optional overrides — uncomment to use:
        # "data_dir":   "data/processed/5Char_360k_AlpNum_grayscale",
        # "charset":    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        # "device":     "cpu",
        # "unseen_data_dir": "data/processed/5Char_100k_Num_grayscale",
        # "unseen_charset":  "0123456789",
    },
    # Add more runs below, e.g.:
    # {"experiment_dir": "experiments/vit_01_baseline"},
    {
        "experiment_dir": "experiments/Alphanumerical/ViT_AlpNum",
    },
    {
        "experiment_dir": "experiments/Alphanumerical/vit_alpnum_01_baseline_long",
    },
    {
        "experiment_dir": "experiments/Alphanumerical/vit_alpnum_02_lower_reg/",
    },
    {
        "experiment_dir": "experiments/Alphanumerical/vit_alpnum_03_lower_lr/",
    },
    {
        "experiment_dir": "experiments/Alphanumerical/vit_alpnum_04_smaller_model/",
    },
]

# ===========================================================================
# Internal helpers — not intended to be edited
# ===========================================================================

import torch  # noqa: E402


def _save_json(data, path):
    path = Path(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"    [saved] {path.name}")


def _backfill_log_training_curves(exp_dir):
    """
    Generate log-scale training-history plots for exp_dir if they are missing.

    Saves into the experiment folder alongside the existing linear plots.
    Skips silently if both files already exist; logs a single line otherwise.
    All failures are caught so they never interrupt the evaluation pipeline.
    """
    from src.training.evaluate import plot_log_training_curves

    log_loss  = exp_dir / "log_loss_curves.png"
    log_error = exp_dir / "log_error_curves.png"
    if log_loss.exists() and log_error.exists():
        return  # already up to date

    history_path = exp_dir / "training_history.json"
    if not history_path.exists():
        print(f"  [skip] log-curve backfill: training_history.json not found in {exp_dir.name}")
        return

    try:
        with open(history_path) as f:
            history = json.load(f)
        plot_log_training_curves(history, save_dir=exp_dir, show=False)
        print(f"  [backfill] log training curves → {exp_dir.name}/")
    except Exception as exc:
        print(f"  [skip] log-curve backfill failed for {exp_dir.name}: {exc}")


def _run_eval_suite(model, loader, device, idx_to_char, out_dir, split, cfg):
    """Run every evaluation step for one split (test or unseen)."""
    from src.evaluation import eval_utils

    print(f"\n  [{split}] accuracy ...")
    metrics = eval_utils.evaluate_accuracy(model, loader, device, idx_to_char)
    _save_json(metrics, out_dir / f"metrics_{split}.json")

    print(f"  [{split}] error breakdown ...")
    breakdown = eval_utils.compute_error_breakdown(model, loader, device, idx_to_char)
    _save_json(breakdown, out_dir / f"error_breakdown_{split}.json")

    print(f"  [{split}] predictions CSV ...")
    eval_utils.export_predictions_csv(
        model, loader, device, idx_to_char,
        out_dir / f"predictions_{split}.csv",
    )

    print(f"  [{split}] confusion matrix ...")
    eval_utils.compute_and_plot_confusion_matrix(
        model, loader, device, idx_to_char,
        out_dir / f"confusion_matrix_{split}.png",
    )

    print(f"  [{split}] qualitative examples ...")
    eval_utils.generate_qualitative_examples(
        model, loader, device, idx_to_char,
        out_dir / f"qualitative_examples_{split}.png",
        num_correct=cfg.get("num_correct_examples", 5),
        num_incorrect=cfg.get("num_incorrect_examples", 5),
    )

    print(f"  [{split}] saliency maps ...")
    try:
        eval_utils.generate_saliency_maps(
            model, loader, device, idx_to_char,
            out_dir / f"saliency_maps_{split}.png",
            num_samples=cfg.get("num_saliency_samples", 5),
        )
    except Exception as exc:
        # Saliency requires autograd through the full model.
        # Some device/model combinations may not support it.
        print(f"    [skip] saliency maps failed: {exc}")

    # ---- Report-enhancement outputs (all additive, no existing files touched) ----

    print(f"\n  [{split}] collecting predictions for report analyses ...")
    collected = eval_utils.collect_all_predictions(model, loader, device, idx_to_char)

    print(f"  [{split}] split qualitative examples ...")
    eval_utils.generate_split_qualitative_examples(
        collected, out_dir, split,
        num_per_fig=cfg.get("num_qualitative_per_fig", 4),
    )

    print(f"  [{split}] saliency figures ...")
    try:
        eval_utils.generate_saliency_figures(
            model, loader, device, idx_to_char, out_dir, split,
            samples_per_fig=cfg.get("saliency_samples_per_fig", 2),
            total_samples=cfg.get("saliency_total_samples", 4),
        )
    except Exception as exc:
        print(f"    [skip] saliency figures failed: {exc}")

    print(f"  [{split}] per-position accuracy ...")
    eval_utils.compute_and_plot_per_position_accuracy(collected, idx_to_char, out_dir, split)

    print(f"  [{split}] sequence error distribution ...")
    eval_utils.plot_sequence_error_distribution(breakdown, out_dir, split)

    print(f"  [{split}] top confusion pairs ...")
    eval_utils.compute_and_plot_top_confusions(
        collected, idx_to_char, out_dir, split,
        top_k=cfg.get("top_confusions_k", 10),
    )

    print(f"  [{split}] confidence analysis ...")
    eval_utils.plot_confidence_analysis(collected, out_dir, split)

    print(f"  [{split}] hard examples ...")
    eval_utils.generate_hard_examples(
        collected, out_dir, split,
        num=cfg.get("num_hard_examples", 5),
    )

    print(f"  [{split}] latent space visualization ...")
    try:
        eval_utils.generate_latent_space_plots(
            model, loader, device, idx_to_char, out_dir, split,
            max_chars=cfg.get("latent_max_chars",  8000),
            run_tsne=cfg.get("latent_run_tsne",    True),
            tsne_sizes=cfg.get("latent_tsne_sizes", [2000, 5000, 8000]),
        )
    except Exception as exc:
        print(f"    [skip] latent space: {exc}")

    return metrics


def run_one_evaluation(eval_cfg):
    """
    Execute the full evaluation pipeline for one experiment.

    Parameters
    ----------
    eval_cfg : dict
        Merged config (DEFAULTS overridden by the per-evaluation dict).

    Returns
    -------
    dict  — metrics_test (sequence_accuracy, character_accuracy, …)
    """
    from src.dataset.dataloader import CaptchaDataset, create_dataloaders
    from src.evaluation import eval_utils
    from torch.utils.data import DataLoader

    # -- Locate experiment --
    exp_dir_raw = eval_cfg.get("experiment_dir")
    if not exp_dir_raw:
        raise ValueError("eval_cfg must contain 'experiment_dir'.")

    exp_dir = Path(exp_dir_raw)
    if not exp_dir.is_absolute():
        exp_dir = (REPO_ROOT / exp_dir).resolve()

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Skipping {exp_dir.name}: config.json not found "
            f"(experiment may have crashed before training started)."
        )

    with open(config_path) as f:
        exp_config = json.load(f)

    # -- Backfill log training-history plots if missing from the experiment folder --
    _backfill_log_training_curves(exp_dir)

    # -- Effective settings: exp_config provides base, eval_cfg overrides --
    def get(key, fallback=None):
        """Prefer eval_cfg, fall back to exp_config, then fallback."""
        if key in eval_cfg:
            return eval_cfg[key]
        if key in exp_config:
            return exp_config[key]
        return fallback

    data_dir    = get("data_dir")
    charset     = get("charset")
    batch_size  = get("batch_size",  64)
    num_workers = get("num_workers", 0)
    pin_memory  = get("pin_memory",  False)
    label_length = exp_config["label_length"]

    # Split ratios — must match training to reproduce the same test fold
    train_ratio = get("train_ratio", 0.75)
    val_ratio   = get("val_ratio",   0.15)
    test_ratio  = get("test_ratio",  0.10)
    random_seed = get("random_seed", 69)

    # -- Device --
    device_str = eval_cfg.get("device") or exp_config.get("device")
    if device_str:
        requested = torch.device(device_str)
        if requested.type == "cuda" and not torch.cuda.is_available():
            device = eval_utils.auto_device()
        elif requested.type == "mps" and not torch.backends.mps.is_available():
            device = eval_utils.auto_device()
        else:
            device = requested
    else:
        device = eval_utils.auto_device()

    # -- Run name and output directory --
    run_name = exp_config.get("run_name", exp_dir.name)
    out_dir  = (REPO_ROOT / eval_cfg.get("output_root", "evaluation") / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Checkpoint --
    checkpoint = exp_dir / "best_model.pt"
    if not checkpoint.exists():
        checkpoint = exp_dir / "last_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {exp_dir}. "
            "Expected best_model.pt or last_model.pt."
        )

    # -- Model --
    model_name = exp_config["model_name"]
    eval_utils.check_checkpoint_compatibility(exp_config, checkpoint)
    model = eval_utils.build_model(model_name, exp_config)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"\n{'='*62}")
    print(f"EVALUATING: {run_name}")
    print(f"  model      = {model_name}")
    print(f"  device     = {device}")
    print(f"  checkpoint = {checkpoint.name}")
    print(f"  data       = {data_dir}")
    print(f"  output     = {out_dir}")
    print(f"{'='*62}")

    # -- Test dataloader (same split as training) --
    _, _, test_loader, char_to_idx, idx_to_char, _ = create_dataloaders(
        data_dir=data_dir,
        charset=charset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        training=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        label_length=label_length,
        return_filenames=True,
    )

    # -- Test evaluation suite --
    metrics_test = _run_eval_suite(
        model, test_loader, device, idx_to_char, out_dir, "test", eval_cfg
    )

    # -- Optional unseen-data evaluation --
    unseen_data_dir = eval_cfg.get("unseen_data_dir")
    metrics_unseen  = None

    if unseen_data_dir:
        unseen_charset = eval_cfg.get("unseen_charset", charset)
        print(f"\n  [unseen] {unseen_data_dir}")
        try:
            unseen_dataset = CaptchaDataset(
                data_dir=unseen_data_dir,
                charset=unseen_charset,
                label_length=label_length,
                return_filenames=True,
            )
            unseen_loader = DataLoader(
                unseen_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            metrics_unseen = _run_eval_suite(
                model, unseen_loader, device,
                unseen_dataset.idx_to_char,
                out_dir, "unseen", eval_cfg,
            )
        except Exception as exc:
            print(f"  [skip] Unseen evaluation failed: {exc}")

    # -- Terminal summary --
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {run_name}")
    print(f"  Test  — seq_acc: {metrics_test['sequence_accuracy']:.4f}"
          f"  ({metrics_test['sequence_accuracy']*100:.2f}%)"
          f"   char_acc: {metrics_test['character_accuracy']:.4f}"
          f"  ({metrics_test['character_accuracy']*100:.2f}%)")
    if metrics_unseen:
        print(f"  Unseen — seq_acc: {metrics_unseen['sequence_accuracy']:.4f}"
              f"  ({metrics_unseen['sequence_accuracy']*100:.2f}%)"
              f"   char_acc: {metrics_unseen['character_accuracy']:.4f}"
              f"  ({metrics_unseen['character_accuracy']*100:.2f}%)")
    print(f"  Outputs saved to: {out_dir}")
    print(sep)

    return metrics_test


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def run_evaluation_sequence(evaluations, defaults, evaluation_dir):
    """Merge defaults into each evaluation config and run them sequentially."""
    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if not evaluations:
        print("No evaluations defined.")
        return []

    all_metrics = []
    errors      = []

    for idx, evaluation in enumerate(evaluations):
        cfg   = {**defaults, **evaluation}
        label = cfg.get("experiment_dir", f"[{idx}]")
        print(f"\n[{idx+1}/{len(evaluations)}] {label}")
        try:
            metrics = run_one_evaluation(cfg)
            all_metrics.append((label, metrics))
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            errors.append((label, str(exc)))

    # Final summary
    print(f"\n{'='*62}")
    print("EVALUATION COMPLETE")
    print(f"{'='*62}")
    for label, m in all_metrics:
        print(
            f"  {Path(label).name:<35}  "
            f"seq={m['sequence_accuracy']:.4f}  "
            f"char={m['character_accuracy']:.4f}  "
            f"({m['correct_sequences']}/{m['total_samples']} sequences correct)"
        )
    if errors:
        print(f"\n  {len(errors)} run(s) failed:")
        for label, msg in errors:
            print(f"    {label}: {msg}")

    return all_metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation_sequence(EVALUATIONS, DEFAULTS, EVALUATION_DIR)
