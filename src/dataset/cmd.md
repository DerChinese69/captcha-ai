# Data Loader — Quick Use Guide

## What this file does

This module creates the full PyTorch data pipeline for the **5-character CAPTCHA dataset**.

It does 4 things:

1. Finds the project root automatically  
2. Reads image filenames and labels from `ground_truth_index.csv`  
3. Encodes each 5-character label into numeric tensors  
4. Splits the dataset into **train / val / test** and wraps them in PyTorch `DataLoader`s

## Basic Call
```python
from src.data_loader import create_dataloaders

train_loader, val_loader, test_loader, char_to_idx, idx_to_char = create_dataloaders()
```
**What gets Returned:**
1. Batched: Training, Validation, Test data
2. Dictionary for encoding labels

## Example
```python
from src.data_loader import create_dataloaders

train_loader, val_loader, test_loader, char_to_idx, idx_to_char = create_dataloaders(
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
    subset_fraction=1.0,
    return_filenames=False
)
```
### Param explained:
*batch_size*
How many images per batch

*train_ratio, val_ratio, test_ratio*
Controls data set split. Must add up to 1

*random_seed*
Makes the split reproducible

*training*
Controls whether train loader should shuffle like training mode.

*shuffle_train*
Whether to shuffle training batches. (Keep this on for training)

*num_workers*
Controls how many subprocesses load data in parallel.

*pin_memory*
Useful mainly when training on GPU.
Default: False
Can later set to True for CUDA training, faster transfer to GPU
MPS not supported - keep default

*drop_last*
Drops the last incomplete batch if it is smaller than batch_size. (Optional use)

*subset_fraction*
Lets you test on only part of the dataset. Range: 0-1
-> Useful for debugging

*return_filenames*
If True, each batch also returns the filename.

**recommended debug call**
```python
train_loader, val_loader, test_loader, char_to_idx, idx_to_char = create_dataloaders(
    batch_size=8,
    subset_fraction=0.1,
    num_workers=0
)
```