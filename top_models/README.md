# Top Models

This folder contains the training metadata and evaluation results for the best
CNN and ViT runs on the **5-character alphanumeric CAPTCHA task**
(36 classes: `0–9`, `A–Z`; 360k training images).

Model checkpoints (`best_model.pt`) are not committed to git.
To reproduce the checkpoint, re-train using the config in each `training/` folder:

```bash
# Edit run_name and add the config values to EXPERIMENTS in run_experiments.py, then:
python run_experiments.py
```

---

## Results summary

| Model | Dataset       | Test seq acc | Test char acc | Training time |
|-------|---------------|:------------:|:-------------:|:-------------:|
| CNN   | 5Char_360k_AlpNum | **91.51%**   | **98.02%**    | ~4.9 h (MPS)  |
| ViT   | 5Char_360k_AlpNum | **91.60%**   | **98.07%**    | ~2.8 h (CUDA) |

Both models were evaluated on a held-out 10% test split (36 000 samples) with
the same random seed (`69`) used during training, ensuring the test fold is
identical to the one seen during model selection.

---

## Folder structure

```
top_models/
├── cnn/
│   ├── training/
│   │   ├── config.json           Full training configuration
│   │   ├── metadata.json         Compact run summary (loss, acc, epoch)
│   │   └── training_history.json Per-epoch metrics
│   └── evaluation/
│       ├── metrics_test.json        Sequence & character accuracy
│       ├── error_breakdown_test.json Error type analysis
│       └── per_position_accuracy_test.json Accuracy at each character slot
└── vit/
    ├── training/
    │   ├── config.json
    │   ├── metadata.json
    │   └── training_history.json
    └── evaluation/
        ├── metrics_test.json
        ├── error_breakdown_test.json
        └── per_position_accuracy_test.json
```

Full evaluation outputs (confusion matrices, saliency maps, qualitative
examples, latent-space t-SNE plots) are generated locally by
`run_evaluation.py` and written to `evaluation/<run_name>/`.

---

## CNN — key hyperparameters

| Parameter         | Value          |
|-------------------|----------------|
| Architecture      | CaptchaCNN     |
| Learning rate     | 3e-4           |
| Batch size        | 64             |
| Epochs            | 40 (all ran)   |
| Weight decay      | 1e-5           |
| Dropout           | 0.3            |
| LR scheduler      | StepLR (step=5, γ=0.5) |
| Dataset           | 5Char_360k_AlpNum_grayscale |

## ViT — key hyperparameters

| Parameter         | Value          |
|-------------------|----------------|
| Architecture      | SmallCaptchaViT (slot-aware pooling) |
| Learning rate     | 1e-4           |
| Batch size        | 16             |
| Epochs            | 20 (all ran)   |
| Weight decay      | 5e-5           |
| Dropout           | 0.0            |
| LR scheduler      | StepLR (step=5, γ=0.5) |
| Embed dim         | 128            |
| Depth             | 4              |
| Num heads         | 4              |
| Patch size        | 8×16           |
| Dataset           | 5Char_360k_AlpNum_grayscale |
