# Alphanumerical CAPTCHA Solver (CNN + ViT)

A deep learning project exploring Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for solving distorted alphanumerical CAPTCHA images.

---

## Overview

This project investigates how different deep learning architectures perform on fixed-length CAPTCHA recognition tasks.

The main objectives are:
- build a working baseline CNN model  
- establish a reproducible training pipeline  
- experiment with alternative architectures (ViT, modified CNNs)  
- compare performance across models  

The focus is on **practical experimentation and iterative improvement**, not production deployment.

---

## Requirements

- Python 3.11 recommended  

---

## Setup

Clone the repository and create a virtual environment:

```bash
git clone <https://github.com/DerChinese69/captcha-ai.git>
cd <captcha-ai>

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Verify Installation

Run the environment check:

```bash
python check_setup.py
```

All packages should return `[OK]`.

---

## Project Structure

```
├── src/                # Core model and training code
├── data/               # Dataset (not included)
├── notebooks/          # Experiments and exploration
├── requirements.txt
├── check_setup.py
├── README.md
```

---

## Data

The dataset consists of synthetic CAPTCHA images generated using:  
https://github.com/Gregwar/Captcha  

Expected structure:

```
data/
└── processed/
    └── <dataset_name>/
        ├── images...
        └── ground_truth_index.csv
```

### Preprocessing
- Grayscale conversion  
- Resizing to consistent resolution  
- Normalization  

Optional augmentations:
- noise injection  
- geometric distortions  
- compression artifacts  

---

## Approach

### Baseline Model: CNN

The initial model uses a convolutional architecture to extract spatial features from CAPTCHA images.

Pipeline:
1. Image preprocessing  
2. Convolutional feature extraction  
3. Feature compression  
4. Character classification  

---

## Project Goals

- Build a working CNN-based CAPTCHA solver  
- Create a reproducible training and evaluation pipeline  
- Experiment with:
  - CNN architecture improvements  
  - Small Vision Transformer (ViT)  
  - Fine-tuned pretrained ViT  

- Compare models based on:
  - accuracy  
  - training stability  
  - generalization  

---

## Notes

AI-assisted tools were used to support development (debugging, refactoring, iteration speed).  
All architectural and methodological decisions were made by the project author.

---

## License

This repository is intended for educational and research purposes.