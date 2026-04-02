import torch
import torch.nn as nn

def initialize_training_run(
    model_class,
    model_kwargs,
    device,
    learning_rate,
    weight_decay=0.0,
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs=None,
    use_scheduler=True,
    scheduler_step_size=5,
    scheduler_gamma=0.5,
):
    """
    Build a fresh training run:
    - model
    - criterion
    - optimizer
    - scheduler
    - empty history
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    model = model_class(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma,
        )

    label_length = model_kwargs["label_length"]

    history = {
        "learning_rate": [],
        "train_loss": [],
        "val_loss": [],
        "train_char_acc": [],
        "val_char_acc": [],
        "train_seq_acc": [],
        "val_seq_acc": [],
        **{f"val_pos_acc_{i}": [] for i in range(label_length)},
    }

    return model, criterion, optimizer, scheduler, history