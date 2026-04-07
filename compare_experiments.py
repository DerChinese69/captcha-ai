"""
compare_experiments.py — Unified ranking of all experiment runs.

Scans all subfolders under experiments/ for metadata.json, optionally
joins evaluation results from evaluation/<run_name>/metrics_test.json,
and produces one combined comparison table sorted by a chosen metric.

Outputs:
    evaluation/experiment_comparison.json
    evaluation/experiment_comparison.csv

Usage:
    # Default: compare all runs, sort by best_val_seq_acc
    python compare_experiments.py

    # Sort by evaluation sequence accuracy
    python compare_experiments.py --sort_by eval_seq_acc

    # Custom directories
    python compare_experiments.py --experiments_dir experiments/ --evaluation_dir evaluation/

    # Split output by model family (CNN / ViT / …)
    python compare_experiments.py --group_by_model
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root
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
DEFAULT_EVALUATION_DIR  = REPO_ROOT / "evaluation"

SORT_CHOICES = ["eval_seq_acc", "eval_char_acc", "best_val_seq_acc", "best_val_loss", ]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_index(evaluation_dir: Path) -> dict[str, dict]:
    """
    Build a dict mapping run_name → eval metrics by scanning evaluation_dir
    for subfolders that contain metrics_test.json.
    """
    index: dict[str, dict] = {}
    if not evaluation_dir.exists():
        return index
    for metrics_path in evaluation_dir.rglob("metrics_test.json"):
        run_key = metrics_path.parent.name   # folder name = run identifier
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            index[run_key] = data
        except Exception as exc:
            print(f"  [skip] Could not read {metrics_path}: {exc}")
    return index


def load_run_record(metadata_path: Path, eval_index: dict) -> dict | None:
    """
    Parse metadata.json and join evaluation metrics where available.
    Returns None (with a warning) on parse errors or missing required fields.
    """
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"  [skip] Could not read {metadata_path}: {exc}")
        return None

    run_name = meta.get("run_name", metadata_path.parent.name)

    if run_name.startswith("smoke_test"):
        return None

    best_val_seq_acc = meta.get("best_val_seq_acc")
    if best_val_seq_acc is None:
        print(f"  [skip] {run_name}: missing best_val_seq_acc")
        return None

    model_type = meta.get("model_type", "").strip()
    if not model_type:
        print(f"  [skip] {run_name}: missing model_type")
        return None

    # Look up evaluation metrics — try run_name, then experiment folder name
    eval_data = eval_index.get(run_name) or eval_index.get(metadata_path.parent.name)
    eval_seq_acc  = eval_data.get("sequence_accuracy")  if eval_data else None
    eval_char_acc = eval_data.get("character_accuracy") if eval_data else None

    return {
        "run_name":          run_name,
        "model_type":        model_type,
        "best_val_seq_acc":  float(best_val_seq_acc),
        "best_val_loss":     meta.get("best_val_loss"),
        "best_epoch":        meta.get("best_epoch"),
        "epochs_completed":  meta.get("epochs_completed"),
        "eval_seq_acc":      eval_seq_acc,
        "eval_char_acc":     eval_char_acc,
        "stop_reason":       meta.get("stop_reason", ""),
        "timestamp":         meta.get("timestamp", ""),
        "path":              str(metadata_path.parent),
    }


def collect_records(experiments_dir: Path, eval_index: dict) -> list[dict]:
    """Walk experiments_dir recursively and return all valid run records."""
    if not experiments_dir.exists():
        print(f"[error] experiments directory not found: {experiments_dir}")
        return []

    records = []
    for metadata_path in sorted(experiments_dir.rglob("metadata.json")):
        record = load_run_record(metadata_path, eval_index)
        if record is not None:
            records.append(record)
    return records


def sort_records(records: list[dict], sort_by: str) -> list[dict]:
    """Sort records by sort_by metric, descending. N/A values go last."""
    reverse = sort_by != "best_val_loss"   # loss: ascending; accuracy: descending

    def key(r):
        v = r.get(sort_by)
        if v is None:
            # Push missing values to the end regardless of direction
            return (1, 0.0)
        return (0, -v if reverse else v)

    return sorted(records, key=key)


def fmt(value, fmt_spec=".4f") -> str:
    """Format a float or return 'N/A'."""
    if value is None:
        return "N/A"
    return format(float(value), fmt_spec)


def print_comparison_table(records: list[dict], title: str, sort_by: str) -> None:
    """Pretty-print a unified ranked table to stdout."""
    sep = "-" * 105
    print(f"\n{sep}")
    print(f"  {title}  ({len(records)} runs, sorted by {sort_by})")
    print(f"  {'Rank':<4} {'Run name':<38} {'Model':<5} {'val_seq':>8} "
          f"{'val_loss':>9} {'eval_seq':>9} {'eval_char':>10} {'ep':>4}/{'>4'}")
    print(sep)
    for rank, r in enumerate(records, start=1):
        print(
            f"  {rank:<4} {r['run_name']:<38} {r['model_type']:<5}"
            f"  {fmt(r['best_val_seq_acc']):>8}"
            f"  {fmt(r['best_val_loss']):>9}"
            f"  {fmt(r['eval_seq_acc']):>9}"
            f"  {fmt(r['eval_char_acc']):>10}"
            f"  {(r['best_epoch'] or 0):>3}/{(r['epochs_completed'] or 0):<3}"
        )
    print(sep)
    if records:
        best = records[0]
        print(f"\n  Best run : {best['run_name']}  ({best['model_type']})")
        print(f"  {sort_by:<20}: {fmt(best.get(sort_by), '.6f')}")


def save_json(records: list[dict], sort_by: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sort_by":      sort_by,
        "total_runs":   len(records),
        "best_run":     records[0] if records else None,
        "ranked_runs":  records,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)


def save_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "run_name", "model_type",
        "best_val_seq_acc", "best_val_loss", "best_epoch", "epochs_completed",
        "eval_seq_acc", "eval_char_acc",
        "stop_reason", "timestamp", "path",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(records, start=1):
            writer.writerow({"rank": rank, **r})


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare(experiments_dir: Path, evaluation_dir: Path, output_dir: Path,
            sort_by: str, group_by_model: bool) -> None:

    eval_index = load_eval_index(evaluation_dir)
    if eval_index:
        print(f"Evaluation runs found : {sorted(eval_index)}")

    all_records = collect_records(experiments_dir, eval_index)
    if not all_records:
        print(f"  No valid runs found in {experiments_dir}")
        return

    if group_by_model:
        # Separate output per model family
        families: dict[str, list] = {}
        for r in all_records:
            families.setdefault(r["model_type"].upper(), []).append(r)

        for family_key in sorted(families):
            ranked = sort_records(families[family_key], sort_by)
            print_comparison_table(ranked, f"Ranking: {family_key}", sort_by)
            suffix = family_key.lower()
            json_path = output_dir / f"experiment_comparison_{suffix}.json"
            csv_path  = output_dir / f"experiment_comparison_{suffix}.csv"
            save_json(ranked, sort_by, json_path)
            save_csv(ranked, csv_path)
            print(f"\n  [saved] {json_path}")
            print(f"  [saved] {csv_path}")
    else:
        # One unified table (default)
        ranked = sort_records(all_records, sort_by)
        print_comparison_table(ranked, "Experiment comparison (all runs)", sort_by)
        json_path = output_dir / "experiment_comparison.json"
        csv_path  = output_dir / "experiment_comparison.csv"
        save_json(ranked, sort_by, json_path)
        save_csv(ranked, csv_path)
        print(f"\n  [saved] {json_path}")
        print(f"  [saved] {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare captcha-ai experiment runs in a unified table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments_dir",
        default=str(DEFAULT_EXPERIMENTS_DIR),
        metavar="DIR",
        help=f"Path to experiments directory (default: {DEFAULT_EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--evaluation_dir",
        default=str(DEFAULT_EVALUATION_DIR),
        metavar="DIR",
        help=f"Path to evaluation directory (default: {DEFAULT_EVALUATION_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_EVALUATION_DIR),
        metavar="DIR",
        help="Where to write comparison files (default: same as evaluation_dir)",
    )
    parser.add_argument(
        "--sort_by",
        default="best_val_seq_acc",
        choices=SORT_CHOICES,
        help="Metric to sort by (default: best_val_seq_acc)",
    )
    parser.add_argument(
        "--group_by_model",
        action="store_true",
        help="Split output by model family instead of one unified table",
    )
    args = parser.parse_args()

    print(f"Experiments dir  : {args.experiments_dir}")
    print(f"Evaluation dir   : {args.evaluation_dir}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Sort by          : {args.sort_by}")
    print(f"Group by model   : {args.group_by_model}")

    compare(
        experiments_dir = Path(args.experiments_dir),
        evaluation_dir  = Path(args.evaluation_dir),
        output_dir      = Path(args.output_dir),
        sort_by         = args.sort_by,
        group_by_model  = args.group_by_model,
    )


if __name__ == "__main__":
    main()
