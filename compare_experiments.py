"""
compare_experiments.py — Rank experiment runs by validation sequence accuracy.

Scans all subfolders under experiments/ for metadata.json, then ranks runs
within each model family (CNN / ViT) by best_val_seq_acc.

Outputs:
    evaluation/experiment_comparison_cnn.json   (if CNN runs found)
    evaluation/experiment_comparison_vit.json   (if ViT runs found)
    evaluation/experiment_comparison_cnn.csv
    evaluation/experiment_comparison_vit.csv

Usage:
    # Compare all models
    python compare_experiments.py

    # CNN only
    python compare_experiments.py --model CNN

    # ViT only
    python compare_experiments.py --model ViT

    # Custom directories
    python compare_experiments.py --experiments-dir experiments/ --output-dir evaluation/
"""

import argparse
import csv
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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_OUTPUT_DIR      = REPO_ROOT / "evaluation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run_record(metadata_path):
    """
    Parse a single metadata.json and return a flat record dict.

    Returns None (with a printed warning) on any parse error.
    """
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"  [skip] Could not read {metadata_path}: {exc}")
        return None

    run_name = meta.get("run_name", metadata_path.parent.name)

    # Skip smoke tests
    if run_name.startswith("smoke_test"):
        return None

    # Required field
    best_val_seq_acc = meta.get("best_val_seq_acc")
    if best_val_seq_acc is None:
        print(f"  [skip] {run_name}: missing best_val_seq_acc")
        return None

    model_type = meta.get("model_type", "").strip()
    if not model_type:
        print(f"  [skip] {run_name}: missing model_type")
        return None

    return {
        "run_name":          run_name,
        "model_type":        model_type,
        "best_val_seq_acc":  float(best_val_seq_acc),
        "best_val_loss":     meta.get("best_val_loss"),
        "best_epoch":        meta.get("best_epoch"),
        "epochs_completed":  meta.get("epochs_completed"),
        "stop_reason":       meta.get("stop_reason", ""),
        "timestamp":         meta.get("timestamp", ""),
        "path":              str(metadata_path.parent),
    }


def collect_records(experiments_dir, model_filter):
    """
    Walk experiments_dir recursively, parse every metadata.json,
    and return a list of valid run records matching model_filter.

    model_filter: "CNN", "ViT", or "all"
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        print(f"[error] experiments directory not found: {experiments_dir}")
        return []

    all_records = []
    for metadata_path in sorted(experiments_dir.rglob("metadata.json")):
        record = load_run_record(metadata_path)
        if record is None:
            continue
        mt = record["model_type"].upper()
        if model_filter == "all" or mt == model_filter.upper():
            all_records.append(record)

    return all_records


def rank_records(records):
    """Sort records by best_val_seq_acc descending.  Returns a new list."""
    return sorted(records, key=lambda r: r["best_val_seq_acc"], reverse=True)


def print_ranking_table(records, model_type):
    """Pretty-print a ranked table to stdout."""
    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  Ranking: {model_type} runs  ({len(records)} total)")
    print(f"{'Rank':<5} {'Run name':<38} {'seq_acc':>8} {'char?':>5} "
          f"{'loss':>8} {'epoch':>6} {'epochs':>7}")
    print(sep)
    for rank, r in enumerate(records, start=1):
        print(
            f"  {rank:<3}  {r['run_name']:<38}"
            f"  {r['best_val_seq_acc']:>7.4f}"
            f"  {'':>5}"          # char acc not in metadata; placeholder
            f"  {(r['best_val_loss'] or 0.0):>8.4f}"
            f"  {(r['best_epoch'] or 0):>5}"
            f"  {(r['epochs_completed'] or 0):>5}"
        )
    print(sep)
    if records:
        best = records[0]
        print(f"\n  Best {model_type} run : {best['run_name']}")
        print(f"  best_val_seq_acc  : {best['best_val_seq_acc']:.6f}"
              f"  ({best['best_val_seq_acc']*100:.2f}%)")
        if best["best_val_loss"] is not None:
            print(f"  best_val_loss     : {best['best_val_loss']:.6f}")
        if best["best_epoch"] is not None:
            print(f"  best_epoch        : {best['best_epoch']}")
        print(f"  path              : {best['path']}")


def save_json_summary(records, model_type, output_dir):
    """Save a JSON summary file and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_type":   model_type,
        "total_runs":   len(records),
        "best_run":     records[0] if records else None,
        "ranked_runs":  records,
    }
    out_path = output_dir / f"experiment_comparison_{model_type.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    return out_path


def save_csv_ranking(records, model_type, output_dir):
    """Save a CSV ranking file and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"experiment_comparison_{model_type.lower()}.csv"
    fieldnames = [
        "rank", "run_name", "model_type",
        "best_val_seq_acc", "best_val_loss", "best_epoch",
        "epochs_completed", "stop_reason", "timestamp", "path",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(records, start=1):
            writer.writerow({"rank": rank, **r})
    return out_path


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare(experiments_dir, output_dir, model_filter):
    """
    Collect, rank, print, and save comparisons for the requested model family.

    model_filter: "CNN", "ViT", or "all"
    """
    all_records = collect_records(experiments_dir, model_filter)

    if not all_records:
        print(f"  No valid runs found for filter={model_filter!r} in {experiments_dir}")
        return

    # Determine which families to report.
    # Preserve original casing from metadata (e.g. "ViT" not "VIT") while
    # still deduplicating case-insensitively.
    if model_filter == "all":
        seen: dict[str, str] = {}
        for r in all_records:
            seen.setdefault(r["model_type"].upper(), r["model_type"])
        families = [seen[k] for k in sorted(seen)]
    else:
        families = [model_filter]

    for family in families:
        family_records = [r for r in all_records if r["model_type"].upper() == family.upper()]
        ranked = rank_records(family_records)

        print_ranking_table(ranked, family)

        json_path = save_json_summary(ranked, family, output_dir)
        csv_path  = save_csv_ranking(ranked, family, output_dir)

        print(f"\n  [saved] {json_path}")
        print(f"  [saved] {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare trained captcha-ai experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["CNN", "ViT", "all"],
        help="Filter by model family (default: all)",
    )
    parser.add_argument(
        "--experiments-dir",
        default=str(DEFAULT_EXPERIMENTS_DIR),
        metavar="DIR",
        help=f"Path to experiments directory (default: {DEFAULT_EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"Where to write comparison files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    print(f"Experiments dir : {args.experiments_dir}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Model filter    : {args.model}")

    compare(
        experiments_dir=Path(args.experiments_dir),
        output_dir=Path(args.output_dir),
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()
