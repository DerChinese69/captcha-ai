from pathlib import Path
import math
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def find_project_root(marker_folder="data"):
    """
    Tries to find the project root by walking upward until it finds /data.
    Works from scripts, notebooks, or different working directories.
    """
    current = Path.cwd().resolve()

    for path in [current] + list(current.parents):
        if (path / marker_folder).exists():
            return path

    raise FileNotFoundError(
        f"Could not find project root. No folder named '{marker_folder}' found upward from {current}"
    )

class CaptchaDataset(Dataset):
    def __init__(
        self,
        data_dir="data/processed/5Char_100000_CapGen_grayscale",
        csv_name="ground_truth_index.csv",
        charset="2346789ABCDEFGHJKLMNPQRTUVWXYabcdefghijkmnpqrtuvwxy",
        label_length=5,
        valid_extensions={".jpg", ".jpeg", ".png"},
        return_filenames=False,
        subset_fraction=1.0,
    ):
        self.project_root = find_project_root()
        self.data_dir = (self.project_root / data_dir).resolve()
        self.csv_path = self.data_dir / csv_name

        self.charset = charset
        self.label_length = label_length
        self.valid_extensions = {ext.lower() for ext in valid_extensions}
        self.return_filenames = return_filenames

        # Character mappings as dictionaries for encoding/decoding
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # Tensor conversion + normalize to [0,1]
        self.transform = transforms.ToTensor()

        # Basic path checks
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.data_dir}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Read labels CSV
        self.labels_df = pd.read_csv(self.csv_path)

        # Validate existence of required columns
        required_columns = {"filename", "label"}
        if not required_columns.issubset(self.labels_df.columns):
            raise ValueError(
                f"CSV must contain columns {required_columns}, found {set(self.labels_df.columns)}"
            )

        # Optional subset for quick testing
        if not (0 < subset_fraction <= 1.0):
            raise ValueError("subset_fraction must be in (0, 1].")

        if subset_fraction < 1.0:
            n_subset = max(1, int(len(self.labels_df) * subset_fraction))
            self.labels_df = self.labels_df.iloc[:n_subset].reset_index(drop=True)

        # Validate rows
        self.samples = []
        for _, row in self.labels_df.iterrows():
            filename = str(row["filename"]).strip()
            label = str(row["label"]).strip()

            file_path = self.data_dir / filename
            file_ext = file_path.suffix.lower()

            if file_ext not in self.valid_extensions:
                continue

            if not file_path.exists():
                continue

            if len(label) != self.label_length:
                continue

            if any(char not in self.char_to_idx for char in label):
                continue

            self.samples.append((file_path, label))

        if len(self.samples) == 0:
            raise ValueError("No valid samples found after validation.")

    def __len__(self):
        return len(self.samples)

    def encode_label(self, label):
        return torch.tensor(
            [self.char_to_idx[char] for char in label],
            dtype=torch.long
        )

    def decode_label(self, indices):
        return "".join(self.idx_to_char[int(idx)] for idx in indices)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        image = Image.open(file_path).convert("L")   # force grayscale
        image = self.transform(image)                # shape: [1, H, W], scaled to [0,1]

        label_tensor = self.encode_label(label)

        if self.return_filenames:
            return image, label_tensor, file_path.name

        return image, label_tensor


def create_dataloaders(
    data_dir="data/processed/5Char_2000_CapGen_grayscale",
    csv_name="ground_truth_index.csv",
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    training=True,
    shuffle_train=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    charset="2346789ABCDEFGHJKLMNPQRTUVWXYabcdefghijkmnpqrtuvwxy",
    subset_fraction=1.0,
    return_filenames=False,
):
    # Split check
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    #Load full dataset
    dataset = CaptchaDataset(
        data_dir=data_dir,
        csv_name=csv_name,
        charset=charset,
        label_length=5,
        return_filenames=return_filenames,
        subset_fraction=subset_fraction,
    )

    # Create splits
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Use a fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    # Create individual DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train if training else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, dataset.char_to_idx, dataset.idx_to_char