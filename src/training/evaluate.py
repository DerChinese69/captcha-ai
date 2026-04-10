from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_curves(history, num_epochs, label_length, save_dir=None, show=True):
    """
    Plot loss, character accuracy, sequence accuracy, and per-position accuracy curves.

    Args:
        history:      dict returned by initialize_training_run / the training loop
        num_epochs:   number of epochs trained
        label_length: number of character positions (used for per-position plot)
        save_dir:     Path to save PNGs into, or None to skip saving
    """
    epochs = range(1, num_epochs + 1)

    _plot(
        epochs,
        curves=[
            (history["train_loss"], "Train Loss"),
            (history["val_loss"],   "Val Loss"),
        ],
        ylabel="Loss",
        title="Training and Validation Loss",
        save_path=Path(save_dir) / "loss_curves.png" if save_dir else None,
        show=show,
    )

    _plot(
        epochs,
        curves=[
            (history["train_char_acc"], "Train Char Acc"),
            (history["val_char_acc"],   "Val Char Acc"),
        ],
        ylabel="Accuracy",
        title="Training and Validation Character Accuracy",
        save_path=Path(save_dir) / "char_acc_curves.png" if save_dir else None,
        show=show,
    )

    _plot(
        epochs,
        curves=[
            (history["train_seq_acc"], "Train Seq Acc"),
            (history["val_seq_acc"],   "Val Seq Acc"),
        ],
        ylabel="Accuracy",
        title="Training and Validation Sequence Accuracy",
        save_path=Path(save_dir) / "seq_acc_curves.png" if save_dir else None,
        show=show,
    )

    _plot(
        epochs,
        curves=[
            (history[f"val_pos_acc_{i}"], f"Val Pos {i}")
            for i in range(label_length)
        ],
        ylabel="Accuracy",
        title="Validation Per-Position Accuracy",
        save_path=Path(save_dir) / "val_pos_acc_curves.png" if save_dir else None,
        show=show,
    )

    # --- Log-scale additions (additive — existing PNGs are unchanged) ---
    plot_log_training_curves(history, save_dir=save_dir, show=show)


def plot_log_training_curves(history, save_dir=None, show=False):
    """
    Generate log-scale training-history plots and save them to save_dir.

    Derives the epoch count from the length of the loss series so it can
    be called independently of plot_training_curves() — useful for
    backfilling old experiment folders that predate this feature.

    Outputs (when save_dir is set):
        log_loss_curves.png   — train/val loss on log y-axis
        log_error_curves.png  — (1 − char_acc) and (1 − seq_acc) on log y-axis

    Missing metrics are skipped gracefully; a missing loss series skips
    log_loss_curves.png without raising.
    """
    # Derive epoch count from whichever loss series is present
    n = len(history.get("train_loss") or history.get("val_loss") or [])
    if n == 0:
        return
    epochs = range(1, n + 1)

    # Log loss
    loss_curves = []
    for key, label in [("train_loss", "Train Loss"), ("val_loss", "Val Loss")]:
        if history.get(key):
            loss_curves.append((history[key], label))
    if loss_curves:
        _plot(
            epochs,
            curves=loss_curves,
            ylabel="Loss (log scale)",
            title="Training and Validation Loss  [log scale]",
            yscale="log",
            save_path=Path(save_dir) / "log_loss_curves.png" if save_dir else None,
            show=show,
        )

    # Log error rate (1 − accuracy): reveals fine-grained improvement that
    # linear plots compress near zero.  Zero-error points are undefined on
    # log scale and silently omitted by matplotlib.
    err_curves = []
    for key, label in [
        ("train_char_acc", "Train Char Error"),
        ("val_char_acc",   "Val Char Error"),
        ("train_seq_acc",  "Train Seq Error"),
        ("val_seq_acc",    "Val Seq Error"),
    ]:
        if history.get(key):
            err_curves.append(([1.0 - v for v in history[key]], label))
    if err_curves:
        _plot(
            epochs,
            curves=err_curves,
            ylabel="Error Rate  (1 − accuracy, log scale)",
            title="Training and Validation Error Rate  [log scale]",
            yscale="log",
            save_path=Path(save_dir) / "log_error_curves.png" if save_dir else None,
            show=show,
        )

    # Separate log-scale char-accuracy error plot
    char_err_curves = []
    eps = 1e-6
    for key, label in [
        ("train_char_acc", "Train Char Error"),
        ("val_char_acc",   "Val Char Error"),
    ]:
        if history.get(key):
            char_err_curves.append(([max(1.0 - v, eps) for v in history[key]], label))
    if char_err_curves:
        char_save = Path(save_dir) / "training_curves_log_char_accuracy.png" if save_dir else None
        if char_save is None or not char_save.exists():
            _plot(
                epochs,
                curves=char_err_curves,
                ylabel="Char Error Rate  (1 − char_acc, log scale)",
                title="Character Accuracy Error Rate  [log scale]",
                yscale="log",
                save_path=char_save,
                show=show,
            )


def plot_log_validation_per_position_accuracy(
    history,
    label_length,
    save_dir=None,
    show=False,
    filename="training_curves_log_val_pos_accuracy.png",
):
    """
    Plot validation per-position error rate on a log scale.

    Uses (1 - accuracy) for each position so improvements near perfect
    accuracy remain visible on a log axis.
    """
    n = len(history.get("val_loss") or [])
    if n == 0:
        return

    epochs = range(1, n + 1)
    eps = 1e-6
    curves = []

    for i in range(label_length):
        key = f"val_pos_acc_{i}"
        if history.get(key):
            curves.append(([max(1.0 - v, eps) for v in history[key]], f"Val Pos {i} Error"))

    if not curves:
        return

    _plot(
        epochs,
        curves=curves,
        ylabel="Error Rate  (1 - accuracy, log scale)",
        title="Validation Per-Position Error Rate  [log scale]",
        yscale="log",
        save_path=Path(save_dir) / filename if save_dir else None,
        show=show,
    )


def _plot(epochs, curves, ylabel, title, save_path=None, show=True, yscale="linear"):
    plt.figure(figsize=(8, 5))
    for values, label in curves:
        plt.plot(epochs, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
