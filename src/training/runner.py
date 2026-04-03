import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.dataset.dataloader import create_dataloaders
from src.models.CaptchaCNN import CaptchaCNN
from src.models.CaptchaViT import SmallCaptchaViT
from src.training.engine import train_one_epoch, validate_one_epoch
from src.training.evaluate import plot_training_curves
from src.training.setup import initialize_training_run


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def make_run_dir(base_dir, run_name):
    """
    Create a unique run directory under base_dir.

    - Uses run_name as-is if the folder does not exist.
    - Appends _01, _02, ... if the name is already taken.
    - Falls back to 'unspecified' when run_name is empty or None.
    """
    name = run_name.strip() if run_name else "unspecified"
    base_dir = Path(base_dir)

    candidate = base_dir / name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    for i in range(1, 100):
        candidate = base_dir / f"{name}_{i:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(
        f"Could not create a unique directory for '{name}' under {base_dir}. "
        "Too many runs with this name?"
    )


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_one_experiment(cfg, experiments_dir, verbose=True, is_smoke=False):
    """
    Execute a single training experiment defined by cfg.

    cfg keys (all others are ignored):
        run_name, model_name, data_dir, charset, label_length,
        batch_size, num_epochs, learning_rate, weight_decay, dropout,
        train_ratio, val_ratio, test_ratio, random_seed,
        num_workers, pin_memory, drop_last, subset_fraction,
        use_scheduler, scheduler_step_size, scheduler_gamma,
        val_loss_stop_threshold,
        img_size, patch_size, embed_dim, depth, num_heads  (ViT only)

    Returns a metadata dict summarising the run.
    """
    start_time = time.time()

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --- Run directory ---
    run_name = cfg.get("run_name") or "unspecified"
    if is_smoke:
        run_name = "smoke_test"
    run_dir = make_run_dir(Path(experiments_dir), run_name)

    # Resolve epoch / subset overrides for smoke mode
    num_epochs = 2 if is_smoke else cfg["num_epochs"]
    subset_fraction = 0.005 if is_smoke else cfg.get("subset_fraction", 1.0)

    print(f"\n{'='*60}")
    print(f"{'SMOKE TEST' if is_smoke else 'RUN'}: {run_dir.name}")
    print(f"  model={cfg['model_name']}  device={device}  epochs={num_epochs}  subset={subset_fraction}")
    print(f"  data={cfg['data_dir']}")
    print(f"{'='*60}")

    # --- Dataloaders ---
    train_loader, val_loader, _, char_to_idx, idx_to_char, label_length = create_dataloaders(
        data_dir=cfg["data_dir"],
        charset=cfg["charset"],
        batch_size=cfg["batch_size"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        random_seed=cfg["random_seed"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=cfg["drop_last"],
        subset_fraction=subset_fraction,
    )

    # Dataset is authoritative for num_char_classes and label_length
    num_char_classes = len(char_to_idx)

    # --- Model ---
    model_name = cfg["model_name"]
    dropout = cfg.get("dropout", 0.0)

    if model_name == "CNN":
        model_class = CaptchaCNN
        model_kwargs = {
            "num_char_classes": num_char_classes,
            "label_length": label_length,
            "dropout": dropout,
        }
    elif model_name == "ViT":
        model_class = SmallCaptchaViT
        model_kwargs = {
            "img_size": cfg.get("img_size", (64, 192)),
            "patch_size": cfg.get("patch_size", (8, 16)),
            "embed_dim": cfg.get("embed_dim", 128),
            "depth": cfg.get("depth", 4),
            "num_heads": cfg.get("num_heads", 4),
            "num_classes": num_char_classes,
            "label_length": label_length,
            "dropout": dropout,
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}. Expected 'CNN' or 'ViT'.")

    model, criterion, optimizer, scheduler, history = initialize_training_run(
        model_class=model_class,
        model_kwargs=model_kwargs,
        device=device,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        use_scheduler=cfg.get("use_scheduler", False),
        scheduler_step_size=cfg.get("scheduler_step_size", 5),
        scheduler_gamma=cfg.get("scheduler_gamma", 0.5),
    )

    # --- File paths ---
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    history_path    = run_dir / "training_history.json"
    config_path     = run_dir / "config.json"
    metadata_path   = run_dir / "metadata.json"

    # --- Training loop ---
    best_val_seq_acc = -1.0
    best_val_loss    = float("inf")
    best_epoch       = 0
    stop_reason      = "completed"
    threshold        = cfg.get("val_loss_stop_threshold")

    for epoch in range(num_epochs):

        train_loss, train_char_acc, train_seq_acc, current_lr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_char_classes=num_char_classes,
            verbose=verbose,
        )

        val_loss, val_char_acc, val_seq_acc, val_pos_accs = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_char_classes=num_char_classes,
            label_length=label_length,
        )

        if scheduler is not None:
            scheduler.step()

        # Update history
        history["learning_rate"].append(current_lr)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_char_acc"].append(train_char_acc)
        history["val_char_acc"].append(val_char_acc)
        history["train_seq_acc"].append(train_seq_acc)
        history["val_seq_acc"].append(val_seq_acc)
        for i in range(label_length):
            history[f"val_pos_acc_{i}"].append(val_pos_accs[i])

        # Epoch-level summary (always printed)
        print(f"\nEpoch [{epoch+1}/{num_epochs}]  LR: {current_lr:.6f}")
        print(f"  Train  loss={train_loss:.4f}  char={train_char_acc:.4f}  seq={train_seq_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  char={val_char_acc:.4f}  seq={val_seq_acc:.4f}")
        print(f"  Val pos: {[f'{a:.3f}' for a in val_pos_accs]}")

        # Best model checkpoint (tracked by val_seq_acc)
        if val_seq_acc > best_val_seq_acc:
            best_val_seq_acc = val_seq_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  [checkpoint] seq_acc improved → {best_val_seq_acc:.4f}")

        # Early stopping check (uses previous best_val_loss, before this epoch updates it)
        if threshold is not None and best_val_loss < float("inf"):
            stop_at = best_val_loss * (1.0 + threshold)
            if val_loss > stop_at:
                stop_reason = (
                    f"early_stop: val_loss {val_loss:.4f} > {stop_at:.4f} "
                    f"(best {best_val_loss:.4f} + {threshold*100:.0f}%)"
                )
                print(f"\n  [early stop] {stop_reason}")
                break

        # Update best val loss after the stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    epochs_completed = len(history["train_loss"])

    # --- Persist outputs ---
    torch.save(model.state_dict(), last_model_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    plot_training_curves(
        history,
        num_epochs=epochs_completed,
        label_length=label_length,
        save_dir=run_dir,
        show=False,
    )

    # Config: everything that was used (Path values → str for JSON)
    config_record = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in cfg.items()
    }
    config_record.update({
        "num_char_classes": num_char_classes,
        "label_length": label_length,
        "device": str(device),
        "is_smoke": is_smoke,
        "num_epochs_actual": num_epochs,
        "subset_fraction_actual": subset_fraction,
    })
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=4)

    # Metadata: compact run summary
    total_time = time.time() - start_time
    metadata = {
        "run_name": run_dir.name,
        "model_type": model_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_training_time_s": round(total_time, 2),
        "epochs_completed": epochs_completed,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_seq_acc": round(best_val_seq_acc, 6),
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "hyperparameters": {
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg.get("weight_decay", 0.0),
            "dropout": cfg.get("dropout", 0.0),
            "num_epochs": cfg["num_epochs"],
            "subset_fraction": subset_fraction,
            "val_loss_stop_threshold": threshold,
            "use_scheduler": cfg.get("use_scheduler", False),
        },
        "data_dir": cfg["data_dir"],
        "charset": cfg["charset"],
        "num_char_classes": num_char_classes,
        "label_length": label_length,
        "device": str(device),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[done] {run_dir.name}  {stop_reason}  ({total_time:.1f}s)")
    print(f"  → {run_dir}")

    return metadata


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def run_experiment_sequence(experiments, defaults, experiments_dir):
    """
    1. Run a smoke test (first experiment config, 2 epochs, tiny subset).
    2. If it passes, run all experiments sequentially.
    3. Print a final summary table.

    Failures in individual full runs are logged and skipped; the sequence
    continues. A smoke test failure aborts immediately.
    """
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    if not experiments:
        print("No experiments defined.")
        return []

    # --- Smoke test ---
    print("\n" + "="*60)
    print("SMOKE TEST")
    print("="*60)

    smoke_cfg = {**defaults, **experiments[0]}
    try:
        run_one_experiment(smoke_cfg, experiments_dir, verbose=False, is_smoke=True)
        print("\n[OK] Smoke test passed — starting full sequence.\n")
    except Exception as exc:
        print(f"\n[FAIL] Smoke test failed: {exc}")
        print("Fix the issue above before launching the full sequence.")
        raise

    # --- Full runs ---
    print(f"{'='*60}")
    print(f"FULL SEQUENCE: {len(experiments)} experiment(s)")
    print(f"{'='*60}")

    all_metadata = []
    for idx, experiment in enumerate(experiments):
        cfg = {**defaults, **experiment}
        label = cfg.get("run_name") or "unspecified"
        print(f"\n[{idx+1}/{len(experiments)}] {label}")
        try:
            meta = run_one_experiment(cfg, experiments_dir, verbose=False)
            all_metadata.append(meta)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            print("  Skipping to next experiment.")

    # --- Summary ---
    print("\n" + "="*60)
    print("SEQUENCE COMPLETE")
    print("="*60)
    for meta in all_metadata:
        print(
            f"  {meta['run_name']:<40}  "
            f"best_loss={meta['best_val_loss']:.4f}  "
            f"best_seq={meta['best_val_seq_acc']:.4f}  "
            f"epochs={meta['epochs_completed']}  "
            f"({meta['stop_reason']})"
        )

    return all_metadata
