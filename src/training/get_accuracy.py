"""
get_accuracy.py — Evaluate best_model.pt checkpoints on the test split.

Reads config.json from each experiment folder to determine the model
architecture (model_name), hyperparameters, and split ratios, then
reports character accuracy, full-sequence accuracy, and per-position
accuracy on the test portion of the dataset.

Usage:
    python get_accuracy.py <run_dir> [<run_dir2> ...] [options]

Examples:
    # Single run
    python get_accuracy.py experiments/succesful/test_run_20260401_1644

    # Multiple runs
    python get_accuracy.py experiments/Debugging/test_run_20260401_2028 \\
                           experiments/succesful/test_run_20260401_1644

    # Override dataset
    python get_accuracy.py experiments/succesful/test_run_20260401_1644 \\
        --data-dir data/processed/5Char_100k_Num_grayscale \\
        --charset 0123456789

    # Override device
    python get_accuracy.py experiments/succesful/test_run_20260401_1644 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root resolution (mirrors smoke_test.py)
# ---------------------------------------------------------------------------

def find_project_root(marker_files=("requirements.txt", "README.md")):
    current = Path.cwd().resolve()
    for path in [current] + list(current.parents):
        if all((path / m).exists() for m in marker_files):
            return path
    raise FileNotFoundError(
        "Could not find project root. "
        "Run this script from inside the captcha-ai directory."
    )


def find_processed_dataset(project_root):
    """
    Return (relative_data_dir, charset) for the first valid processed dataset,
    or (None, None) if none exists. Mirrors smoke_test.py behaviour.
    """
    import pandas as pd

    processed = project_root / "data" / "processed"
    if not processed.exists():
        return None, None
    for d in sorted(processed.iterdir()):
        csv = d / "ground_truth_index.csv"
        if d.is_dir() and csv.exists():
            df = pd.read_csv(csv, nrows=500, dtype=str)
            if "label" not in df.columns:
                continue
            charset = "".join(sorted(set("".join(df["label"].dropna().tolist()))))
            if not charset:
                continue
            return str(d.relative_to(project_root)), charset
    return None, None


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name, config):
    """Instantiate the correct model class from config fields."""
    from src.models.CaptchaCNN import CaptchaCNN
    from src.models.CaptchaViT import SmallCaptchaViT

    num_char_classes = config["num_char_classes"]
    label_length = config["label_length"]

    if model_name == "CNN":
        dropout = config.get("dropout", 0.3)
        return CaptchaCNN(
            num_char_classes=num_char_classes,
            label_length=label_length,
            dropout=dropout,
        )

    if model_name == "ViT":
        img_size   = config.get("img_size",   (64, 192))
        patch_size = config.get("patch_size", (8, 16))
        if isinstance(img_size,   list): img_size   = tuple(img_size)
        if isinstance(patch_size, list): patch_size = tuple(patch_size)
        return SmallCaptchaViT(
            num_classes=num_char_classes,
            label_length=label_length,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=config.get("embed_dim", 128),
            depth=config.get("depth",         4),
            num_heads=config.get("num_heads",  4),
            dropout=config.get("dropout",      0.1),
        )

    raise ValueError(
        f"Unknown model_name: {model_name!r}. Expected 'CNN' or 'ViT'."
    )


# ---------------------------------------------------------------------------
# Evaluation loop (accuracy only — no loss needed)
# ---------------------------------------------------------------------------

def evaluate_on_test_loader(model, loader, device, label_length):
    """
    Run inference over the test loader.

    Returns:
        char_acc   float  — mean character-level accuracy
        seq_acc    float  — mean full-sequence accuracy
        pos_accs   list[float] — per-position accuracy (length = label_length)
    """
    import torch
    from src.training.engine import compute_metrics, unpack_batch

    model.eval()

    running_char_acc = 0.0
    running_seq_acc = 0.0
    running_pos_accs = [0.0] * label_length
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            char_acc, seq_acc, pos_accs = compute_metrics(outputs, labels)

            running_char_acc += char_acc
            running_seq_acc += seq_acc
            for i in range(label_length):
                running_pos_accs[i] += pos_accs[i]
            num_batches += 1

    return (
        running_char_acc / num_batches,
        running_seq_acc / num_batches,
        [x / num_batches for x in running_pos_accs],
    )


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(run_dir, data_dir, charset, device_override):
    """
    Load best_model.pt from run_dir and evaluate it on the test split.

    Returns a dict with results, or raises on any hard failure.
    """
    import torch
    from src.dataset.dataloader import create_dataloaders

    run_path = Path(run_dir).resolve()

    # -- Config --
    config_path = run_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_path}")

    with open(config_path) as f:
        config = json.load(f)

    model_name = config.get("model_name")
    if model_name is None:
        raise ValueError(
            f"{run_path.name}: config.json is missing 'model_name'. "
            "Add \"model_name\": \"CNN\" or \"model_name\": \"ViT\" to the config."
        )

    # -- Checkpoint --
    checkpoint_path = run_path / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {run_path}")

    # -- Device --
    device_str = device_override or config.get("device", "cpu")
    device = torch.device(device_str)

    # -- DataLoader (test split only) --
    label_length = config["label_length"]
    _, _, test_loader, _, _, _ = create_dataloaders(
        data_dir=data_dir,
        charset=charset,
        batch_size=config.get("batch_size", 32),
        train_ratio=config.get("train_ratio", 0.7),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        random_seed=config.get("random_seed", 42),
        training=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        label_length=label_length,
        subset_fraction=config.get("subset_fraction", 1.0),
    )

    # -- Model --
    model = build_model(model_name, config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # -- Evaluate --
    char_acc, seq_acc, pos_accs = evaluate_on_test_loader(
        model, test_loader, device, label_length
    )

    return {
        "run": run_path.name,
        "model_name": model_name,
        "device": str(device),
        "char_acc": char_acc,
        "seq_acc": seq_acc,
        "pos_accs": pos_accs,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results):
    sep = "-" * 52

    for r in results:
        print(sep)
        print(f"Run      : {r['run']}")
        print(f"Model    : {r['model_name']}   Device: {r['device']}")
        print(f"Char Acc : {r['char_acc']:.4f}  ({r['char_acc']*100:.2f}%)")
        print(f"Seq  Acc : {r['seq_acc']:.4f}  ({r['seq_acc']*100:.2f}%)")
        pos_str = "  ".join(
            f"[{i}] {acc*100:.1f}%" for i, acc in enumerate(r["pos_accs"])
        )
        print(f"Pos Accs : {pos_str}")

    print(sep)

    if len(results) > 1:
        print("\nRanking by sequence accuracy:")
        for rank, r in enumerate(
            sorted(results, key=lambda x: x["seq_acc"], reverse=True), start=1
        ):
            print(f"  {rank}. {r['run']}  — seq {r['seq_acc']*100:.2f}%  char {r['char_acc']*100:.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best_model.pt checkpoints on the test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        metavar="run_dir",
        help="Path(s) to experiment folder(s) containing best_model.pt and config.json",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Dataset directory relative to project root "
            "(default: auto-discover first available processed dataset)"
        ),
    )
    parser.add_argument(
        "--charset",
        default=None,
        help="Character set string, e.g. '0123456789' or 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
             "(default: inferred from dataset)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cpu', 'cuda', 'mps' (default: from config.json)",
    )
    args = parser.parse_args()

    # -- Project root & path setup --
    project_root = find_project_root()
    sys.path.insert(0, str(project_root))

    # -- Resolve dataset --
    data_dir = args.data_dir
    charset = args.charset

    if data_dir is None or charset is None:
        discovered_dir, discovered_charset = find_processed_dataset(project_root)
        if discovered_dir is None:
            parser.error(
                "No processed dataset found under data/processed/. "
                "Provide --data-dir and --charset explicitly."
            )
        data_dir = data_dir or discovered_dir
        charset = charset or discovered_charset

    print(f"Dataset  : {data_dir}")
    print(f"Charset  : {charset!r}  ({len(charset)} classes)\n")

    # -- Evaluate each run --
    results = []
    errors = []

    for run_dir in args.run_dirs:
        print(f"Evaluating: {run_dir} ...")
        try:
            result = evaluate_run(run_dir, data_dir, charset, args.device)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append((run_dir, str(e)))

    # -- Print results --
    if results:
        print()
        print_results(results)

    if errors:
        print(f"\n{len(errors)} run(s) failed:")
        for run_dir, msg in errors:
            print(f"  {run_dir}: {msg}")

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
