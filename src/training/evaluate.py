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


def _plot(epochs, curves, ylabel, title, save_path=None, show=True):
    plt.figure(figsize=(8, 5))
    for values, label in curves:
        plt.plot(epochs, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
