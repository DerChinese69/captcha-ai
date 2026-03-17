import torch
import torch.nn as nn

from src.models.FiveCharCNN import FiveCharCaptchaCNN


def build_training_state(
    num_char_classes,
    label_length,
    device,
    learning_rate,
    weight_decay=0.0,
):
    """
    Create a fresh model, loss function, optimizer, and empty history dict
    for a new training run.
    """
    model = FiveCharCaptchaCNN(
        num_char_classes=num_char_classes,
        label_length=label_length,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_char_acc": [],
        "val_char_acc": [],
        "train_seq_acc": [],
        "val_seq_acc": [],
        **{f"val_pos_acc_{i}": [] for i in range(label_length)},
    }

    return model, criterion, optimizer, history