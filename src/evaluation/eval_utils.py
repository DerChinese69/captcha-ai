"""
eval_utils.py — Evaluation helper functions for captcha-ai.

All public functions share a consistent signature:
    fn(model, dataloader, device, idx_to_char, ...) -> result or saved file

They are intentionally stateless — pass in a model, loader, device, and
character mapping, get outputs back.  run_evaluation.py orchestrates them.

Note: matplotlib is set to the non-interactive "Agg" backend at import time
so this module is safe to use in scripts.  Remove the matplotlib.use() call
if you import this from a Jupyter notebook that manages its own backend.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def auto_device():
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model factory  (mirrors get_accuracy.py — duplicated to avoid importing a
# top-level script as a module)
# ---------------------------------------------------------------------------

def build_model(model_name, config):
    """Instantiate CNN or ViT from a config dict loaded from config.json.

    All architecture parameters are taken from config so the reconstructed
    model exactly matches the one that was trained, regardless of any
    subsequent changes to source-code defaults.
    """
    from src.models.CaptchaCNN import CaptchaCNN
    from src.models.CaptchaViT import SmallCaptchaViT

    num_char_classes = config["num_char_classes"]
    label_length     = config["label_length"]

    if model_name == "CNN":
        return CaptchaCNN(
            num_char_classes=num_char_classes,
            label_length=label_length,
            dropout=config.get("dropout", 0.3),
        )
    if model_name == "ViT":
        # Read every architecture dimension from config so checkpoints load
        # correctly even if source-code defaults have changed since training.
        img_size   = config.get("img_size",   (64, 192))
        patch_size = config.get("patch_size", (8, 16))
        # JSON round-trips lists, not tuples; SmallCaptchaViT expects tuples.
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
    raise ValueError(f"Unknown model_name: {model_name!r}. Expected 'CNN' or 'ViT'.")


def check_checkpoint_compatibility(config, checkpoint_path):
    """
    Load a checkpoint and verify its keys match the model built from config.

    Prints a concise summary and raises RuntimeError on mismatch so the
    caller can skip the run gracefully rather than seeing a wall of size errors.
    """
    import torch

    model_name = config.get("model_name", "?")
    print(f"  [compat] {model_name}  embed_dim={config.get('embed_dim','?')}"
          f"  patch_size={config.get('patch_size','?')}"
          f"  depth={config.get('depth','?')}  num_heads={config.get('num_heads','?')}"
          f"  num_classes={config.get('num_char_classes','?')}"
          f"  label_length={config.get('label_length','?')}")
    print(f"  [compat] checkpoint: {checkpoint_path}")

    model = build_model(model_name, config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_keys  = set(model.state_dict().keys())
    ckpt_keys   = set(state_dict.keys())
    missing     = model_keys - ckpt_keys
    unexpected  = ckpt_keys  - model_keys

    shape_errors = []
    for k in model_keys & ckpt_keys:
        ms = model.state_dict()[k].shape
        cs = state_dict[k].shape
        if ms != cs:
            shape_errors.append(f"    {k}: model={tuple(ms)}  ckpt={tuple(cs)}")

    if missing or unexpected or shape_errors:
        lines = ["Checkpoint / config mismatch — cannot load safely:"]
        if shape_errors:
            lines += ["  Shape mismatches:"] + shape_errors
        if missing:
            lines.append(f"  Keys missing from checkpoint: {sorted(missing)}")
        if unexpected:
            lines.append(f"  Unexpected keys in checkpoint: {sorted(unexpected)}")
        raise RuntimeError("\n".join(lines))

    print(f"  [compat] OK — architecture matches checkpoint.")


# ---------------------------------------------------------------------------
# A. decode_sequence
# ---------------------------------------------------------------------------

def decode_sequence(sequence_indices, idx_to_char):
    """Convert a 1-D sequence of integer class indices to a string label."""
    return "".join(idx_to_char[int(i)] for i in sequence_indices)


# ---------------------------------------------------------------------------
# B. evaluate_accuracy
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, dataloader, device, idx_to_char):
    """
    Evaluate model on a dataloader using sample-level counts (not batch means).

    Returns
    -------
    dict with keys:
        sequence_accuracy, character_accuracy,
        total_samples, total_characters,
        correct_sequences, correct_characters
    """
    from src.training.engine import unpack_batch

    model.eval()
    total_sequences    = 0
    correct_sequences  = 0
    total_characters   = 0
    correct_characters = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).argmax(dim=-1)   # [B, L]

            B, L = labels.shape
            char_correct = (preds == labels)        # [B, L] bool mask

            total_sequences    += B
            correct_sequences  += int(char_correct.all(dim=1).sum())
            total_characters   += B * L
            correct_characters += int(char_correct.sum())

    seq_acc  = correct_sequences  / total_sequences  if total_sequences  > 0 else 0.0
    char_acc = correct_characters / total_characters if total_characters > 0 else 0.0

    return {
        "sequence_accuracy":  round(seq_acc,  6),
        "character_accuracy": round(char_acc, 6),
        "total_samples":      total_sequences,
        "total_characters":   total_characters,
        "correct_sequences":  correct_sequences,
        "correct_characters": correct_characters,
    }


# ---------------------------------------------------------------------------
# C. export_predictions_csv
# ---------------------------------------------------------------------------

def export_predictions_csv(model, dataloader, device, idx_to_char, save_path):
    """
    Write one row per sample to a CSV file.

    Columns:
        sample_index, filename, ground_truth, prediction,
        num_wrong_characters, exact_match
    """
    from src.training.engine import unpack_batch

    model.eval()
    rows       = []
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels, filenames = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).argmax(dim=-1)   # [B, L]

            for i in range(labels.size(0)):
                gt    = decode_sequence(labels[i], idx_to_char)
                pred  = decode_sequence(preds[i],  idx_to_char)
                fname = filenames[i] if filenames is not None else ""
                rows.append({
                    "sample_index":         sample_idx,
                    "filename":             fname,
                    "ground_truth":         gt,
                    "prediction":           pred,
                    "num_wrong_characters": sum(g != p for g, p in zip(gt, pred)),
                    "exact_match":          int(gt == pred),
                })
                sample_idx += 1

    save_path = Path(save_path)
    fieldnames = [
        "sample_index", "filename", "ground_truth",
        "prediction", "num_wrong_characters", "exact_match",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"    [saved] {save_path.name}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# D. compute_and_plot_confusion_matrix
# ---------------------------------------------------------------------------

def compute_and_plot_confusion_matrix(model, dataloader, device, idx_to_char, save_path):
    """
    Character-level confusion matrix flattened over all label positions.
    Row-normalised (recall per true class).  Saved as PNG.
    """
    from src.training.engine import unpack_batch

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).argmax(dim=-1)   # [B, L]
            all_preds.append(preds.cpu().reshape(-1))
            all_labels.append(labels.cpu().reshape(-1))

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    num_classes  = len(idx_to_char)
    class_labels = [idx_to_char[i] for i in range(num_classes)]

    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    except ImportError:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(all_labels, all_preds):
            cm[int(t), int(p)] += 1

    # Row-normalise so each row sums to 1.0 (recall-style)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1).astype(float)
    cm_norm  = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_labels, fontsize=7, rotation=90)
    ax.set_yticklabels(class_labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Character-Level Confusion Matrix (row-normalised recall)", fontsize=12)

    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {save_path.name}")


# ---------------------------------------------------------------------------
# E. compute_error_breakdown
# ---------------------------------------------------------------------------

def compute_error_breakdown(model, dataloader, device, idx_to_char):
    """
    Bucket each sequence by how many characters are wrong (0, 1, 2, 3+).

    Returns
    -------
    dict with counts and proportions for:
        exact_match, one_error, two_errors, three_or_more_errors
    """
    from src.training.engine import unpack_batch

    model.eval()
    buckets = {0: 0, 1: 0, 2: 0, "3+": 0}
    total   = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            preds   = model(images).argmax(dim=-1)   # [B, L]
            n_wrong = (preds != labels).sum(dim=1)   # [B] int

            for w in n_wrong.cpu().tolist():
                w = int(w)
                buckets[w if w <= 2 else "3+"] += 1
                total += 1

    def prop(k):
        return round(buckets[k] / total, 6) if total > 0 else 0.0

    return {
        "total_samples":            total,
        "exact_match":              buckets[0],
        "one_error":                buckets[1],
        "two_errors":               buckets[2],
        "three_or_more_errors":     buckets["3+"],
        "exact_match_proportion":   prop(0),
        "one_error_proportion":     prop(1),
        "two_errors_proportion":    prop(2),
        "three_or_more_proportion": prop("3+"),
    }


# ---------------------------------------------------------------------------
# F. generate_qualitative_examples
# ---------------------------------------------------------------------------

def generate_qualitative_examples(
    model,
    dataloader,
    device,
    idx_to_char,
    save_path,
    num_correct=5,
    num_incorrect=5,
    max_batches=100,
):
    """
    Save a report-friendly grid of example predictions.
    Correct predictions are titled in green, incorrect in red.
    Scans up to max_batches to find the requested number of each type.
    """
    from src.training.engine import unpack_batch

    model.eval()
    correct_ex   = []
    incorrect_ex = []

    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            if batch_i >= max_batches:
                break
            if len(correct_ex) >= num_correct and len(incorrect_ex) >= num_incorrect:
                break

            images, labels, _ = unpack_batch(batch)
            images_dev = images.to(device)
            labels_dev = labels.to(device)

            preds = model(images_dev).argmax(dim=-1)

            for i in range(labels.size(0)):
                gt    = decode_sequence(labels_dev[i], idx_to_char)
                pred  = decode_sequence(preds[i],      idx_to_char)
                img_np = images[i].squeeze(0).cpu().numpy()   # [H, W]

                if gt == pred and len(correct_ex) < num_correct:
                    correct_ex.append((img_np, gt, pred))
                elif gt != pred and len(incorrect_ex) < num_incorrect:
                    incorrect_ex.append((img_np, gt, pred))

    examples = correct_ex + incorrect_ex
    if not examples:
        print("    [skip] No examples found for qualitative grid.")
        return

    n     = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(max(3 * n, 6), 3))
    axes  = [axes] if n == 1 else list(axes)

    for ax, (img_np, gt, pred) in zip(axes, examples):
        ax.imshow(img_np, cmap="gray", aspect="auto")
        ax.axis("off")
        colour = "green" if gt == pred else "red"
        ax.set_title(f"GT:   {gt}\nPred: {pred}", fontsize=8, color=colour, pad=4)

    plt.suptitle(
        "Qualitative Examples  (green = correct,  red = incorrect)",
        fontsize=9,
    )
    plt.tight_layout()

    save_path = Path(save_path)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(
        f"    [saved] {save_path.name}"
        f"  ({len(correct_ex)} correct, {len(incorrect_ex)} incorrect)"
    )


# ---------------------------------------------------------------------------
# G. generate_saliency_maps
# ---------------------------------------------------------------------------

def generate_saliency_maps(
    model,
    dataloader,
    device,
    idx_to_char,
    save_path,
    num_samples=5,
):
    """
    Vanilla gradient saliency: backprop from predicted-class logits to input.

    Each sample is processed independently (one forward + one backward per
    sample) so no retain_graph tricks are needed.

    Saves a 2-row grid: top row = original image, bottom row = saliency map.
    """
    from src.training.engine import unpack_batch

    model.eval()
    samples = []

    for batch in dataloader:
        if len(samples) >= num_samples:
            break

        images, labels, _ = unpack_batch(batch)

        for i in range(images.size(0)):
            if len(samples) >= num_samples:
                break

            # Single-sample tensor — grad tracking on input
            img_in = images[i:i+1].to(device).detach().requires_grad_(True)
            label  = labels[i:i+1].to(device)

            output = model(img_in)              # [1, L, C]
            pred   = output.argmax(dim=-1)      # [1, L]

            # Scalar: sum of logits at predicted class for each position
            L = output.size(1)
            score = output[0, torch.arange(L, device=device), pred[0]].sum()
            score.backward()

            # Absolute gradient as saliency [H, W]
            saliency = img_in.grad[0].abs().squeeze(0).detach().cpu().numpy()
            s_min, s_max = saliency.min(), saliency.max()
            if s_max > s_min:
                saliency = (saliency - s_min) / (s_max - s_min)

            gt     = decode_sequence(label[0], idx_to_char)
            pred_s = decode_sequence(pred[0],  idx_to_char)
            img_np = images[i].squeeze(0).cpu().numpy()   # [H, W]

            samples.append((img_np, saliency, gt, pred_s))

    if not samples:
        print("    [skip] No samples available for saliency maps.")
        return

    n    = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(max(3 * n, 6), 5))
    axes = axes.reshape(2, n)   # ensure 2-D even when n=1

    for col, (img_np, saliency, gt, pred_s) in enumerate(samples):
        colour = "green" if gt == pred_s else "red"

        axes[0, col].imshow(img_np, cmap="gray", aspect="auto")
        axes[0, col].axis("off")
        axes[0, col].set_title(
            f"GT:   {gt}\nPred: {pred_s}", fontsize=7, color=colour
        )

        axes[1, col].imshow(saliency, cmap="hot", aspect="auto")
        axes[1, col].axis("off")

    # Row labels via figure text (axis("off") hides set_ylabel)
    fig.text(0.01, 0.73, "Image",    fontsize=8, va="center", rotation=90)
    fig.text(0.01, 0.27, "Saliency", fontsize=8, va="center", rotation=90)

    plt.suptitle("Vanilla Gradient Saliency Maps", fontsize=10)
    plt.tight_layout()

    save_path = Path(save_path)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {save_path.name}  ({n} samples)")


# ===========================================================================
# Report-enhancement functions (added alongside the originals above)
# ===========================================================================

# ---------------------------------------------------------------------------
# H1. collect_all_predictions  — single inference pass for all report outputs
# ---------------------------------------------------------------------------

def collect_all_predictions(model, dataloader, device, idx_to_char, max_images=500):
    """
    Run inference once and collect everything needed by the report-enhancement
    functions below.  Avoids re-running the model for each analysis.

    Returns a dict with two groups:

    Full-set arrays  (all N samples in the loader):
        labels_2d       np.ndarray [N, L]  — ground-truth class indices
        preds_2d        np.ndarray [N, L]  — predicted class indices
        seq_correct     np.ndarray [N]  bool
        seq_confidence  np.ndarray [N]  float  — min(max-softmax) over positions
        n_wrong         np.ndarray [N]  int    — wrong chars per sequence

    Capped image list  (up to max_images, for qualitative / hard-example plots):
        images          list of [H, W] float32 arrays
        gt_strings      list of str
        pred_strings    list of str
        img_correct     list of bool
        img_confidence  list of float
        img_n_wrong     list of int
        img_indices     list of int   — original sample index in the loader
    """
    import torch.nn.functional as F
    from src.training.engine import unpack_batch

    model.eval()

    all_labels = []
    all_preds  = []
    all_conf   = []

    images_out = []
    gt_out     = []
    pred_out   = []
    correct_out = []
    conf_out   = []
    nwrong_out = []
    idx_out    = []

    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = unpack_batch(batch)
            images_dev = images.to(device)
            labels_dev = labels.to(device)

            logits = model(images_dev)                              # [B, L, C]
            preds  = logits.argmax(dim=-1)                         # [B, L]
            probs  = F.softmax(logits, dim=-1)                     # [B, L, C]
            conf   = probs.max(dim=-1).values.min(dim=-1).values   # [B]

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_conf.append(conf.cpu())

            for i in range(labels.size(0)):
                if len(images_out) < max_images:
                    gt   = decode_sequence(labels_dev[i], idx_to_char)
                    pred = decode_sequence(preds[i],      idx_to_char)
                    images_out.append(images[i].squeeze(0).cpu().numpy())
                    gt_out.append(gt)
                    pred_out.append(pred)
                    correct_out.append(gt == pred)
                    conf_out.append(float(conf[i].item()))
                    nwrong_out.append(sum(g != p for g, p in zip(gt, pred)))
                    idx_out.append(sample_idx)
                sample_idx += 1

    labels_2d = torch.cat(all_labels).numpy()   # [N, L]
    preds_2d  = torch.cat(all_preds).numpy()    # [N, L]
    conf_all  = torch.cat(all_conf).numpy()     # [N]

    return {
        "n_total":        sample_idx,
        "labels_2d":      labels_2d,
        "preds_2d":       preds_2d,
        "seq_correct":    (labels_2d == preds_2d).all(axis=1),
        "seq_confidence": conf_all,
        "n_wrong":        (labels_2d != preds_2d).sum(axis=1),
        "images":         images_out,
        "gt_strings":     gt_out,
        "pred_strings":   pred_out,
        "img_correct":    correct_out,
        "img_confidence": conf_out,
        "img_n_wrong":    nwrong_out,
        "img_indices":    idx_out,
    }


# ---------------------------------------------------------------------------
# H2. generate_split_qualitative_examples
# ---------------------------------------------------------------------------

def generate_split_qualitative_examples(collected, out_dir, split, num_per_fig=4):
    """
    Save correct and incorrect examples as two separate, LaTeX-ready figures.

    Outputs:
        qualitative_correct_examples_{split}.png
        qualitative_incorrect_examples_{split}.png
    """
    out_dir = Path(out_dir)

    correct_ex = [
        (img, gt, pred)
        for img, gt, pred, ok in zip(
            collected["images"], collected["gt_strings"],
            collected["pred_strings"], collected["img_correct"],
        )
        if ok
    ][:num_per_fig]

    incorrect_ex = [
        (img, gt, pred)
        for img, gt, pred, ok in zip(
            collected["images"], collected["gt_strings"],
            collected["pred_strings"], collected["img_correct"],
        )
        if not ok
    ][:num_per_fig]

    def _save_grid(examples, save_path, colour):
        if not examples:
            print(f"    [skip] {Path(save_path).name} — no examples found.")
            return
        n   = len(examples)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.5))
        axes = [axes] if n == 1 else list(axes)
        for ax, (img, gt, pred) in zip(axes, examples):
            ax.imshow(img, cmap="gray", aspect="auto")
            ax.axis("off")
            ax.set_title(f"GT:   {gt}\nPred: {pred}", fontsize=8, color=colour, pad=3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    [saved] {Path(save_path).name}")

    _save_grid(correct_ex,   out_dir / f"qualitative_correct_examples_{split}.png",   "darkgreen")
    _save_grid(incorrect_ex, out_dir / f"qualitative_incorrect_examples_{split}.png", "crimson")


# ---------------------------------------------------------------------------
# H3. generate_saliency_figures  — compact, per-figure saliency output
# ---------------------------------------------------------------------------

def _collect_saliency_samples(model, dataloader, device, idx_to_char, num_samples):
    """Collect (img_np, saliency, gt, pred_str) tuples via vanilla gradients."""
    from src.training.engine import unpack_batch

    model.eval()
    samples = []

    for batch in dataloader:
        if len(samples) >= num_samples:
            break
        images, labels, _ = unpack_batch(batch)
        for i in range(images.size(0)):
            if len(samples) >= num_samples:
                break
            img_in = images[i:i+1].to(device).detach().requires_grad_(True)
            label  = labels[i:i+1].to(device)
            output = model(img_in)
            pred   = output.argmax(dim=-1)
            L      = output.size(1)
            score  = output[0, torch.arange(L, device=device), pred[0]].sum()
            score.backward()
            saliency = img_in.grad[0].abs().squeeze(0).detach().cpu().numpy()
            s_min, s_max = saliency.min(), saliency.max()
            if s_max > s_min:
                saliency = (saliency - s_min) / (s_max - s_min)
            gt     = decode_sequence(label[0], idx_to_char)
            pred_s = decode_sequence(pred[0],  idx_to_char)
            samples.append((images[i].squeeze(0).cpu().numpy(), saliency, gt, pred_s))

    return samples


def generate_saliency_figures(
    model, dataloader, device, idx_to_char, out_dir, split,
    samples_per_fig=2, total_samples=4,
):
    """
    Split saliency maps into compact 2-row figures (image + saliency side-by-side).

    Outputs:
        saliency_examples_{split}_01.png
        saliency_examples_{split}_02.png   (and more if total_samples > samples_per_fig)
    """
    out_dir = Path(out_dir)
    samples = _collect_saliency_samples(model, dataloader, device, idx_to_char, total_samples)

    if not samples:
        print(f"    [skip] No saliency samples collected for {split}.")
        return

    chunks = [samples[i:i+samples_per_fig] for i in range(0, len(samples), samples_per_fig)]

    for fig_idx, chunk in enumerate(chunks, start=1):
        n    = len(chunk)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 4))
        axes = axes.reshape(2, n)

        for col, (img_np, saliency, gt, pred_s) in enumerate(chunk):
            colour = "darkgreen" if gt == pred_s else "crimson"
            axes[0, col].imshow(img_np, cmap="gray", aspect="auto")
            axes[0, col].axis("off")
            axes[0, col].set_title(f"GT: {gt}\nPred: {pred_s}", fontsize=7, color=colour)
            axes[1, col].imshow(saliency, cmap="hot", aspect="auto")
            axes[1, col].axis("off")

        fig.text(0.01, 0.73, "Image",    fontsize=7, va="center", rotation=90)
        fig.text(0.01, 0.27, "Saliency", fontsize=7, va="center", rotation=90)
        plt.tight_layout()

        save_path = out_dir / f"saliency_examples_{split}_{fig_idx:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    [saved] {save_path.name}")


# ---------------------------------------------------------------------------
# H4. compute_and_plot_per_position_accuracy
# ---------------------------------------------------------------------------

def compute_and_plot_per_position_accuracy(collected, idx_to_char, out_dir, split):
    """
    Per-position character accuracy from pre-collected predictions.

    Outputs:
        per_position_accuracy_{split}.json
        per_position_accuracy_{split}.png
    """
    out_dir   = Path(out_dir)
    labels_2d = collected["labels_2d"]   # [N, L]
    preds_2d  = collected["preds_2d"]    # [N, L]
    N, L      = labels_2d.shape

    positions = []
    for pos in range(L):
        correct = int((preds_2d[:, pos] == labels_2d[:, pos]).sum())
        positions.append({
            "position":       pos,
            "position_label": f"Position {pos + 1}",
            "correct":        correct,
            "total":          N,
            "accuracy":       round(correct / N, 6) if N > 0 else 0.0,
        })

    result = {"total_samples": N, "label_length": L, "positions": positions}
    json_path = out_dir / f"per_position_accuracy_{split}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"    [saved] {json_path.name}")

    x_labels = [p["position_label"] for p in positions]
    accs     = [p["accuracy"] for p in positions]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(x_labels, accs, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlabel("Character Position", fontsize=10)
    ax.set_title(f"Per-Position Character Accuracy ({split})", fontsize=11)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=8,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    png_path = out_dir / f"per_position_accuracy_{split}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {png_path.name}")


# ---------------------------------------------------------------------------
# H5. plot_sequence_error_distribution
# ---------------------------------------------------------------------------

def plot_sequence_error_distribution(error_breakdown, out_dir, split):
    """
    Bar chart of sequence error distribution from an existing error_breakdown dict.
    Consistent naming with compute_error_breakdown output.

    Outputs:
        sequence_error_distribution_{split}.json
        sequence_error_distribution_{split}.png
    """
    out_dir = Path(out_dir)

    json_path = out_dir / f"sequence_error_distribution_{split}.json"
    with open(json_path, "w") as f:
        json.dump(error_breakdown, f, indent=4)
    print(f"    [saved] {json_path.name}")

    categories = ["exact_match", "one_error", "two_errors", "three_or_more_errors"]
    x_labels   = ["0 errors\n(exact)", "1 error", "2 errors", "≥ 3 errors"]
    counts     = [error_breakdown.get(k, 0) for k in categories]
    total      = error_breakdown.get("total_samples", sum(counts)) or 1
    props      = [c / total for c in counts]
    colours    = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(x_labels, props, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Proportion of sequences", fontsize=10)
    ax.set_title(f"Sequence Error Distribution ({split})", fontsize=11)
    for bar, prop, count in zip(bars, props, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prop:.1%}\n({count})", ha="center", va="bottom", fontsize=8,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    png_path = out_dir / f"sequence_error_distribution_{split}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {png_path.name}")


# ---------------------------------------------------------------------------
# H6. compute_and_plot_top_confusions  +  focused confusion matrix
# ---------------------------------------------------------------------------

def compute_and_plot_top_confusions(collected, idx_to_char, out_dir, split, top_k=10):
    """
    Extract the most frequent off-diagonal confusion pairs and plot them.

    Outputs:
        top_confusions_{split}.csv
        top_confusions_{split}.png
        confusion_matrix_{split}_top_classes.png
    """
    out_dir    = Path(out_dir)
    labels_1d  = collected["labels_2d"].ravel()
    preds_1d   = collected["preds_2d"].ravel()
    num_classes = len(idx_to_char)

    # Build raw (unnormalised) confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels_1d, preds_1d, labels=list(range(num_classes)))
    except ImportError:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(labels_1d, preds_1d):
            cm[int(t), int(p)] += 1

    # Top-k off-diagonal pairs
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_idx          = np.argsort(cm_off.ravel())[::-1][:top_k]
    row_idx, col_idx  = np.unravel_index(flat_idx, cm_off.shape)

    top_pairs = [
        {
            "true_char":      idx_to_char[int(r)],
            "predicted_char": idx_to_char[int(c)],
            "count":          int(cm_off[r, c]),
        }
        for r, c in zip(row_idx, col_idx)
        if cm_off[r, c] > 0
    ]

    if not top_pairs:
        print(f"    [skip] top_confusions_{split} — no misclassifications found.")
        return

    # CSV
    csv_path = out_dir / f"top_confusions_{split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true_char", "predicted_char", "count"])
        writer.writeheader()
        writer.writerows(top_pairs)
    print(f"    [saved] {csv_path.name}")

    # Horizontal bar chart (most frequent at top)
    bar_labels = [f"{p['true_char']} → {p['predicted_char']}" for p in top_pairs]
    bar_counts = [p["count"] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(6, max(3, len(top_pairs) * 0.42)))
    y_pos = range(len(top_pairs))
    ax.barh(list(y_pos), bar_counts[::-1], color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(bar_labels[::-1], fontsize=9)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title(f"Top-{len(top_pairs)} Confusion Pairs ({split})", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    png_path = out_dir / f"top_confusions_{split}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {png_path.name}")

    # Focused confusion matrix: only classes involved in top confusions
    char_to_idx = {v: k for k, v in idx_to_char.items()}
    involved    = sorted(
        {p["true_char"] for p in top_pairs} | {p["predicted_char"] for p in top_pairs}
    )
    inv_idx  = [char_to_idx[c] for c in involved]
    sub_cm   = cm[np.ix_(inv_idx, inv_idx)]
    row_sums = sub_cm.sum(axis=1, keepdims=True).clip(min=1).astype(float)
    sub_norm = sub_cm / row_sums

    n_sub = len(involved)
    fig, ax = plt.subplots(figsize=(max(5, n_sub * 0.65), max(4, n_sub * 0.55)))
    im = ax.imshow(sub_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_sub))
    ax.set_yticks(range(n_sub))
    ax.set_xticklabels(involved, fontsize=9)
    ax.set_yticklabels(involved, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"Confusion Matrix — Top Confused Classes ({split})", fontsize=11)
    plt.tight_layout()
    mat_path = out_dir / f"confusion_matrix_{split}_top_classes.png"
    plt.savefig(mat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {mat_path.name}")


# ---------------------------------------------------------------------------
# H7. plot_confidence_analysis
# ---------------------------------------------------------------------------

def plot_confidence_analysis(collected, out_dir, split):
    """
    Overlapping histograms of sequence confidence for correct vs incorrect predictions.
    Confidence = min(max-softmax) over the 5 positions (weakest-link).

    Output:
        confidence_correct_vs_incorrect_{split}.png
    """
    out_dir = Path(out_dir)

    conf    = collected["seq_confidence"]   # [N]
    correct = collected["seq_correct"]      # [N] bool

    conf_correct   = conf[correct]
    conf_incorrect = conf[~correct]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, 1, 26)
    if len(conf_correct) > 0:
        ax.hist(
            conf_correct, bins=bins, alpha=0.6,
            color="steelblue", label=f"Correct (n={len(conf_correct):,})",
        )
    if len(conf_incorrect) > 0:
        ax.hist(
            conf_incorrect, bins=bins, alpha=0.6,
            color="crimson", label=f"Incorrect (n={len(conf_incorrect):,})",
        )
    ax.set_xlabel("Sequence confidence  (min max-softmax over positions)", fontsize=9)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Confidence: Correct vs Incorrect ({split})", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = out_dir / f"confidence_correct_vs_incorrect_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {save_path.name}")


# ---------------------------------------------------------------------------
# H8. generate_hard_examples
# ---------------------------------------------------------------------------

def generate_hard_examples(collected, out_dir, split, num=5):
    """
    Show the lowest-confidence incorrect predictions.
    Useful for qualitative failure analysis in the report.

    Output:
        hardest_examples_{split}.png
    """
    out_dir = Path(out_dir)

    # Gather incorrect entries from the capped image list
    incorrect = [
        i for i, ok in enumerate(collected["img_correct"]) if not ok
    ]

    if not incorrect:
        print(f"    [skip] hardest_examples_{split} — no incorrect examples in image cache.")
        return

    # Sort by confidence ascending (lowest confidence first)
    incorrect.sort(key=lambda i: collected["img_confidence"][i])
    hard_idx = incorrect[:num]

    n    = len(hard_idx)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.8))
    axes = [axes] if n == 1 else list(axes)

    for ax, i in zip(axes, hard_idx):
        ax.imshow(collected["images"][i], cmap="gray", aspect="auto")
        ax.axis("off")
        ax.set_title(
            f"GT:   {collected['gt_strings'][i]}\n"
            f"Pred: {collected['pred_strings'][i]}\n"
            f"Conf: {collected['img_confidence'][i]:.2f}",
            fontsize=7, color="crimson", pad=3,
        )

    plt.suptitle(
        f"Hardest Examples — Lowest-Confidence Incorrect ({split})", fontsize=9
    )
    plt.tight_layout()
    save_path = out_dir / f"hardest_examples_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {save_path.name}  ({n} samples)")


# ---------------------------------------------------------------------------
# I. Latent-space visualization
# ---------------------------------------------------------------------------

def collect_character_embeddings(model, dataloader, device, idx_to_char, max_chars=5000):
    """
    Extract per-character-slot embeddings from model.extract_features().

    Iterates batches until max_chars individual character embeddings have
    been collected (one per slot per image), then stops.

    Returns
    -------
    embeddings  : np.ndarray  [N, D]
    char_labels : list[str]   length N  — true character at each slot
    positions   : list[int]   length N  — slot index (0..label_length-1)

    Raises
    ------
    AttributeError  if the model does not implement extract_features().
    """
    from src.training.engine import unpack_batch

    if not hasattr(model, "extract_features"):
        raise AttributeError(
            f"{type(model).__name__} does not implement extract_features(). "
            "Add the method or skip latent-space visualization for this model."
        )

    model.eval()
    all_embeddings = []
    all_labels     = []
    all_positions  = []
    total_chars    = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_chars >= max_chars:
                break
            images, labels, _ = unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            feats = model.extract_features(images)          # [B, L, D]
            feats_np  = feats.contiguous().cpu().numpy()    # [B, L, D]
            labels_np = labels.cpu().numpy()                # [B, L]

            B, L, D = feats_np.shape
            for b in range(B):
                for pos in range(L):
                    if total_chars >= max_chars:
                        break
                    all_embeddings.append(feats_np[b, pos])
                    all_labels.append(idx_to_char[int(labels_np[b, pos])])
                    all_positions.append(pos)
                    total_chars += 1

    return np.array(all_embeddings), all_labels, all_positions


def _plot_2d_scatter(coords, char_labels, save_path, title, xlabel, ylabel):
    """Scatter plot colored by character class with a compact legend."""
    unique_chars = sorted(set(char_labels))
    n_classes    = len(unique_chars)

    # Perceptually distinct colors for any class count
    color_fn      = plt.cm.turbo
    char_to_color = {c: color_fn(i / max(n_classes - 1, 1))
                     for i, c in enumerate(unique_chars)}
    point_colors  = [char_to_color[c] for c in char_labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1],
               c=point_colors, s=6, alpha=0.55, linewidths=0)

    # Legend: always include but compact for large alphabets
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=char_to_color[c], markersize=7, label=c)
        for c in unique_chars
    ]
    ncol = 2 if n_classes > 13 else 1
    ax.legend(handles=handles, title="Character", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=7, title_fontsize=8,
              ncol=ncol, framealpha=0.8)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def generate_latent_space_plots(
    model, dataloader, device, idx_to_char, out_dir, split,
    max_chars=8000, run_tsne=True, tsne_sizes=(2000, 5000, 8000),
):
    """
    Collect character embeddings and save PCA (always) and t-SNE (optional) plots.

    Outputs (in out_dir)
    --------------------
    latent_space_characters_pca_{split}.png
    latent_space_characters_tsne_{size}_{split}.png   one per entry in tsne_sizes

    max_chars controls how many embeddings are collected; it should be at
    least as large as the biggest value in tsne_sizes so that each plot
    draws from a full pool of that size rather than being forced to reuse
    points.  If the loader has fewer samples than a requested size, all
    available embeddings are used and the filename reflects the actual count.

    The function calls model.extract_features() which must return [B, L, D].
    For CaptchaCNN this yields true per-slot spatial features (D=256).
    For SmallCaptchaViT this yields the global sequence embedding broadcast
    across slots (D=embed_dim) — noted in the plot subtitle.
    """
    from sklearn.decomposition import PCA

    out_dir    = Path(out_dir)
    model_name = type(model).__name__

    print(f"    collecting embeddings (max {max_chars:,} chars) ...")
    embeddings, char_labels, positions = collect_character_embeddings(
        model, dataloader, device, idx_to_char, max_chars=max_chars,
    )

    N, D = embeddings.shape
    print(f"    {N} embeddings  D={D}  classes={len(set(char_labels))}")

    is_vit = model_name == "SmallCaptchaViT"
    embed_note = " (sequence-level, broadcast to all slots)" if is_vit else ""

    # --- PCA (uses all collected embeddings) ---
    pca        = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(embeddings)
    var        = pca.explained_variance_ratio_
    title_pca  = (
        f"{model_name} — Latent Space (PCA)  [{split}]{embed_note}\n"
        f"n={N:,}  D={D}  classes={len(set(char_labels))}"
    )
    save_pca = out_dir / f"latent_space_characters_pca_{split}.png"
    _plot_2d_scatter(
        coords_pca, char_labels, save_pca, title_pca,
        xlabel=f"PC1 ({var[0]*100:.1f}%)",
        ylabel=f"PC2 ({var[1]*100:.1f}%)",
    )
    print(f"    [saved] {save_pca.name}")

    # --- t-SNE — one plot per requested size ---
    if run_tsne:
        from sklearn.manifold import TSNE

        rng = np.random.default_rng(42)
        for size in tsne_sizes:
            if N > size:
                idx     = rng.choice(N, size, replace=False)
                emb_sub = embeddings[idx]
                lbl_sub = [char_labels[i] for i in idx]
            else:
                emb_sub, lbl_sub = embeddings, char_labels

            n_sub = len(emb_sub)
            tsne  = TSNE(n_components=2, random_state=42,
                         perplexity=min(30, n_sub - 1), n_iter=500)
            coords_tsne = tsne.fit_transform(emb_sub)
            label_suffix = "(subsampled)" if N > size else "(all)"
            title_tsne   = (
                f"{model_name} — Latent Space (t-SNE n={n_sub:,})  [{split}]{embed_note}\n"
                f"n={n_sub:,} {label_suffix}  D={D}  classes={len(set(lbl_sub))}"
            )
            save_tsne = out_dir / f"latent_space_characters_tsne_{size}_{split}.png"
            _plot_2d_scatter(
                coords_tsne, lbl_sub, save_tsne, title_tsne,
                xlabel="t-SNE 1", ylabel="t-SNE 2",
            )
            print(f"    [saved] {save_tsne.name}")
