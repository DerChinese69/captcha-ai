#imports
import torch

def compute_metrics(outputs, labels):
    """
    Compute character accuracy, full-sequence accuracy, and per-position accuracy.

    Args:
        outputs: Tensor of shape [B, 5, num_classes]
        labels: Tensor of shape [B, 5]

    Returns:
        char_acc: float
        seq_acc: float
        pos_accs: list[float]
    """
    preds = outputs.argmax(dim=-1)

    char_acc = (preds == labels).float().mean().item()
    seq_acc = (preds == labels).all(dim=1).float().mean().item()

    pos_accs = []
    for pos in range(labels.size(1)):
        pos_acc = (preds[:, pos] == labels[:, pos]).float().mean().item()
        pos_accs.append(pos_acc)

    return char_acc, seq_acc, pos_accs

def unpack_batch(batch):
    """
    Support both:
    - (images, labels)
    - (images, labels, filenames)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, labels = batch
            filenames = None
        elif len(batch) == 3:
            images, labels, filenames = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    return images, labels, filenames

def train_one_epoch(model, loader, optimizer, criterion, device, num_char_classes):
    """
    Train the model for one epoch.

    Returns:
        epoch_loss, epoch_char_acc, epoch_seq_acc
    """
    model.train()

    running_loss = 0.0
    running_char_acc = 0.0
    running_seq_acc = 0.0
    num_batches = 0

    for batch in loader:
        images, labels, _ = unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.view(-1, num_char_classes), labels.view(-1))

        loss.backward()
        optimizer.step()

        char_acc, seq_acc, _ = compute_metrics(outputs, labels)

        running_loss += loss.item()
        running_char_acc += char_acc
        running_seq_acc += seq_acc
        num_batches += 1

    epoch_loss = running_loss / num_batches
    epoch_char_acc = running_char_acc / num_batches
    epoch_seq_acc = running_seq_acc / num_batches

    return epoch_loss, epoch_char_acc, epoch_seq_acc

def validate_one_epoch(model, loader, criterion, device, num_char_classes, label_length):
    """
    Evaluate model on validation set.

    Returns:
        epoch_loss
        epoch_char_acc
        epoch_seq_acc
        epoch_pos_accs
    """
    model.eval()

    running_loss = 0.0
    running_char_acc = 0.0
    running_seq_acc = 0.0
    running_pos_accs = [0.0] * label_length
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images, labels, _ = unpack_batch(batch)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs.view(-1, num_char_classes), labels.view(-1))

            char_acc, seq_acc, pos_accs = compute_metrics(outputs, labels)

            running_loss += loss.item()
            running_char_acc += char_acc
            running_seq_acc += seq_acc

            for i in range(label_length):
                running_pos_accs[i] += pos_accs[i]

            num_batches += 1

    epoch_loss = running_loss / num_batches
    epoch_char_acc = running_char_acc / num_batches
    epoch_seq_acc = running_seq_acc / num_batches
    epoch_pos_accs = [x / num_batches for x in running_pos_accs]

    return epoch_loss, epoch_char_acc, epoch_seq_acc, epoch_pos_accs