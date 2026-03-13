from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import argparse

# automatically detect project root
repo_root = Path(__file__).resolve().parents[2]

# --- Paths ---
input_root = repo_root / "data" / "raw"
output_root = repo_root / "data" / "processed"
output_root.mkdir(parents=True, exist_ok=True)

# --- Settings ---
progress_every = 500
output_suffix = "_grayscale"
valid_extensions = {".jpg", ".jpeg", ".png"}

# Choose what to process (from command line args or process all if none provided):
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batches",
    nargs="+",
    help="Names of batch folders to process (e.g. 4Char_2000_CapGen)"
)

args = parser.parse_args()

selected_batches = set(args.batches) if args.batches else None

def should_process_batch(batch_name, selected_batches=None):
    # Returns True when no filter is provided, or when the batch name is in the allow-list.
    if selected_batches is None:
        return True
    return batch_name in selected_batches


def compute_region_means(arr):
    # Splits image into 4 vertical regions and compares center vs outer brightness.
    h, w = arr.shape
    slice_width = w // 4

    left_outer   = arr[:, 0:slice_width]
    left_centre  = arr[:, slice_width:2 * slice_width]
    right_centre = arr[:, 2 * slice_width:3 * slice_width]
    right_outer  = arr[:, 3 * slice_width:w]

    left_outer_mean = np.mean(left_outer)
    left_centre_mean = np.mean(left_centre)
    right_centre_mean = np.mean(right_centre)
    right_outer_mean = np.mean(right_outer)

    centre_mean = (left_centre_mean + right_centre_mean) / 2
    outer_mean = (left_outer_mean + right_outer_mean) / 2

    return centre_mean, outer_mean


def standardize_polarity(img_path):
    # Converts to grayscale and inverts when center is brighter than outer regions.
    img = Image.open(img_path).convert("L")
    arr = np.array(img)

    centre_mean, outer_mean = compute_region_means(arr)

    if centre_mean > outer_mean:
        arr = 255 - arr

    return Image.fromarray(arr.astype(np.uint8)), centre_mean, outer_mean


count = 0
skipped = 0

# Walk each item in the raw input root (batch folders and possible root-level CSV files).
for item in input_root.iterdir():
    if item.is_dir():
        batch_name = item.name

        if not should_process_batch(batch_name, selected_batches):
            print(f"Skipping batch: {batch_name}")
            continue

        # Mirror output as <batch_name>_grayscale.
        out_batch_dir = output_root / f"{batch_name}{output_suffix}"
        out_batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing batch: {batch_name}")

        for file_path in item.iterdir():

            # --- Image processing ---
            if file_path.suffix.lower() in valid_extensions:

                processed_img, centre_mean, outer_mean = standardize_polarity(file_path)

                output_path = out_batch_dir / file_path.name
                processed_img.save(output_path)

                count += 1
                if count % progress_every == 0:
                    print(f"Processed {count} images...")

            # --- CSV handling ---
            elif file_path.suffix.lower() == ".csv":

                # Preserve batch metadata/index files alongside processed images.
                shutil.copy2(file_path, out_batch_dir / file_path.name)

            else:
                skipped += 1


    # Root-level CSV files (if any exist).
    elif item.is_file() and item.suffix.lower() == ".csv":
        shutil.copy2(item, output_root / item.name)
        print(f"Copied CSV for batch {batch_name}")

print(f"Done. Processed {count} images.")
print(f"Skipped {skipped} non-image files.")

if __name__ == "__main__":
    print("Starting preprocessing...")