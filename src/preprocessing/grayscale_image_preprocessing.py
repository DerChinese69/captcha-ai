"""
grayscale_image_preprocessing.py — Preprocess raw CAPTCHA images.

Converts each raw image to grayscale and standardizes polarity so that
CAPTCHA characters are always dark-on-light. Copies ground_truth_index.csv
alongside the processed images.

Input:  data/raw/<batch_name>/
Output: data/processed/<batch_name>_grayscale/

Usage:
    # Process all batches found under data/raw/
    python src/preprocessing/grayscale_image_preprocessing.py

    # Process specific batches
    python src/preprocessing/grayscale_image_preprocessing.py \\
        --batches 5Char_360k_AlpNum 5Char_100k_Num

    # Faster with multiple workers
    python src/preprocessing/grayscale_image_preprocessing.py --workers 4
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parents[2]
input_root = repo_root / "data" / "raw"
output_root = repo_root / "data" / "processed"

PROGRESS_EVERY = 500
OUTPUT_SUFFIX = "_grayscale"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grayscale-preprocess raw CAPTCHA images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        help="Names of batch folders to process (default: all under data/raw/)",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=6,
        help="PNG compression level 0–9 (lower is faster, larger files). Default: 6",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads. Default: 1",
    )
    return parser.parse_args()


def compute_region_means(arr):
    """Split image into 4 vertical regions; return (centre_mean, outer_mean)."""
    h, w = arr.shape
    s = w // 4
    left_outer   = arr[:, :s]
    left_centre  = arr[:, s:2 * s]
    right_centre = arr[:, 2 * s:3 * s]
    right_outer  = arr[:, 3 * s:]
    centre_mean = (np.mean(left_centre) + np.mean(right_centre)) / 2
    outer_mean  = (np.mean(left_outer)  + np.mean(right_outer))  / 2
    return centre_mean, outer_mean


def standardize_polarity(img_path):
    """Convert to grayscale and invert when characters are lighter than background."""
    with Image.open(img_path) as img:
        arr = np.array(img.convert("L"))
    centre_mean, outer_mean = compute_region_means(arr)
    if centre_mean > outer_mean:
        arr = 255 - arr
    return Image.fromarray(arr.astype(np.uint8))


def process_single_image(file_path, out_dir, png_compress_level):
    processed = standardize_polarity(file_path)
    out_path = out_dir / file_path.name
    if out_path.suffix.lower() == ".png":
        processed.save(out_path, compress_level=png_compress_level)
    else:
        processed.save(out_path)
    processed.close()


def main():
    args = parse_args()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.batches) if args.batches else None
    count = skipped = failed = 0

    for item in sorted(input_root.iterdir()):
        if not item.is_dir():
            if item.suffix.lower() == ".csv":
                shutil.copy2(item, output_root / item.name)
                print(f"Copied root CSV: {item.name}")
            continue

        batch_name = item.name
        if selected is not None and batch_name not in selected:
            print(f"Skipping: {batch_name}")
            continue

        out_batch_dir = output_root / f"{batch_name}{OUTPUT_SUFFIX}"
        out_batch_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing: {batch_name}")

        image_files = []
        for f in item.iterdir():
            if f.suffix.lower() in VALID_EXTENSIONS:
                image_files.append(f)
            elif f.suffix.lower() == ".csv":
                shutil.copy2(f, out_batch_dir / f.name)
            else:
                skipped += 1

        if args.workers <= 1:
            for fp in image_files:
                try:
                    process_single_image(fp, out_batch_dir, args.png_compress_level)
                    count += 1
                    if count % PROGRESS_EVERY == 0:
                        print(f"  Processed {count} images...")
                except Exception as exc:
                    failed += 1
                    print(f"  Failed {fp.name}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_single_image, fp, out_batch_dir, args.png_compress_level): fp
                    for fp in image_files
                }
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        future.result()
                        count += 1
                        if count % PROGRESS_EVERY == 0:
                            print(f"  Processed {count} images...")
                    except Exception as exc:
                        failed += 1
                        print(f"  Failed {fp.name}: {exc}")

    print(f"\nDone. Processed {count} images.")
    if skipped:
        print(f"Skipped {skipped} non-image files.")
    if failed:
        print(f"Failed {failed} images.")


if __name__ == "__main__":
    main()
