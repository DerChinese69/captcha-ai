"""
Smoke test — run this on a new machine to verify the environment is set up correctly.

Checks:
  1. Project root can be located
  2. Third-party dependencies are installed (torch, PIL, pandas, torchvision)
  3. All src/ modules can be imported
  4. Device selection works (cuda / mps / cpu)
  5. Processed data exists and a dataloader can be created
  6. A batch can be loaded
  7. CaptchaCNN forward pass produces the correct output shape
  8. SmallCaptchaViT forward pass produces the correct output shape

Usage:
  python smoke_test.py
"""

from pathlib import Path
import sys


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
    Return (relative_data_dir, charset) for the first valid processed dataset found,
    or (None, None) if none exists.
    """
    import pandas as pd

    processed = project_root / "data" / "processed"
    if not processed.exists():
        return None, None
    for d in sorted(processed.iterdir()):
        csv = d / "ground_truth_index.csv"
        if d.is_dir() and csv.exists():
            # Infer charset from the first 500 labels so we don't read the full file.
            df = pd.read_csv(csv, nrows=500, dtype=str)
            if "label" not in df.columns:
                continue
            charset = "".join(sorted(set("".join(df["label"].dropna().tolist()))))
            if not charset:
                continue
            return str(d.relative_to(project_root)), charset
    return None, None


def main():
    print("Running smoke test...\n")

    # ---- 1. Project root ----
    project_root = find_project_root()
    sys.path.insert(0, str(project_root))
    print(f"[OK] Project root: {project_root}")

    # ---- 2. Third-party dependencies ----
    try:
        import torch
        from PIL import Image  # noqa: F401
        import pandas  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Run:  pip install -r requirements.txt"
        ) from e
    print(f"[OK] Dependencies — torch {torch.__version__}")

    # ---- 3. src/ module imports ----
    try:
        from src.dataset.dataloader import create_dataloaders
        from src.models.CaptchaCNN import CaptchaCNN
        from src.models.CaptchaViT import SmallCaptchaViT
        from src.training.engine import unpack_batch, train_one_epoch, validate_one_epoch  # noqa: F401
        from src.training.setup import initialize_training_run  # noqa: F401
    except Exception as e:
        raise ImportError(f"Could not import src/ modules: {e}") from e
    print("[OK] src/ modules (dataset, models, training)")

    # ---- 4. Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[OK] Device: {device}")

    # ---- 5. Processed data ----
    data_dir, charset = find_processed_dataset(project_root)
    if data_dir is None:
        print(
            "\n[SKIP] No processed dataset found under data/processed/\n"
            "       Generate and preprocess data first, then re-run.\n"
            "\nImports and dependencies are OK — environment is otherwise ready."
        )
        return

    try:
        train_loader, val_loader, test_loader, char_to_idx, idx_to_char, label_length = (
            create_dataloaders(
                data_dir=data_dir,
                charset=charset,
                batch_size=4,
                subset_fraction=0.002,
                num_workers=0,
            )
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataloaders: {e}") from e

    num_char_classes = len(char_to_idx)
    print(f"[OK] Dataloaders — dataset: {data_dir}")
    print(f"     classes: {num_char_classes}, label_length: {label_length}")

    # ---- 6. One batch ----
    try:
        batch = next(iter(train_loader))
        images = batch[0].to(device)
        labels = batch[1].to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load a batch: {e}") from e
    print(f"[OK] Batch — images: {tuple(images.shape)}, labels: {tuple(labels.shape)}")

    expected_shape = (images.shape[0], label_length, num_char_classes)

    # ---- 7. CaptchaCNN forward pass ----
    try:
        cnn = CaptchaCNN(
            num_char_classes=num_char_classes,
            label_length=label_length,
        ).to(device)
        with torch.no_grad():
            cnn_out = cnn(images)
    except Exception as e:
        raise RuntimeError(f"CaptchaCNN forward pass failed: {e}") from e

    if tuple(cnn_out.shape) != expected_shape:
        raise ValueError(
            f"CaptchaCNN output shape {tuple(cnn_out.shape)} != expected {expected_shape}"
        )
    print(f"[OK] CaptchaCNN — output: {tuple(cnn_out.shape)}")

    # ---- 8. SmallCaptchaViT forward pass ----
    try:
        vit = SmallCaptchaViT(
            num_classes=num_char_classes,
            label_length=label_length,
        ).to(device)
        with torch.no_grad():
            vit_out = vit(images)
    except Exception as e:
        raise RuntimeError(f"SmallCaptchaViT forward pass failed: {e}") from e

    if tuple(vit_out.shape) != expected_shape:
        raise ValueError(
            f"SmallCaptchaViT output shape {tuple(vit_out.shape)} != expected {expected_shape}"
        )
    print(f"[OK] SmallCaptchaViT — output: {tuple(vit_out.shape)}")

    print("\nSmoke test passed. Environment is ready.")


if __name__ == "__main__":
    main()
