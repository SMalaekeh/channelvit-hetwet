# ChannelViT-HetWet

Train and evaluate a **multi-channel Vision Transformer (ChannelViT)** for geospatial or scientific imagery where inputs have more than 3 channels (e.g., RGB + NIR). This repo is a clean, job-ready scaffold with:

- Reproducible env (`environment.yml`, `requirements.txt`)
- Training script with custom `in_chans` (PyTorch + timm + Lightning)
- Notebook: `notebooks/Hetwet_ChannelVIT.ipynb` (project work)
- CI (lint + tests) via GitHub Actions
- Pre-commit (`ruff`, `black`)
- Model card, contribution guide, and MIT license

> Author: **Sayedmorteza Malaekeh (Ali)**

---

## Quickstart

### 1) Clone & env
```bash
git clone https://github.com/<your-username>/channelvit-hetwet.git
cd channelvit-hetwet
conda env create -f environment.yml
conda activate channelvit
```

### 2) Data
Place your training data under `data/`. For geospatial stacks, replace the dummy dataset with a real `rasterio`-based loader that returns tensors shaped `(C, H, W)`.

### 3) Train (smoke test)
```bash
python -m channelvit_hetwet.train
```
This runs a 1-epoch sanity check on a synthetic multi-band dataset. To log to Weights & Biases:
```bash
WANDB_API_KEY=... python -m channelvit_hetwet.train
```
and set `use_wandb=True` inside `Config` or wire up Hydra to read `configs/default.yaml`.

### 4) Notebook
Open `notebooks/Hetwet_ChannelVIT.ipynb` for the original end-to-end workflow (ChannelViT training, GPU monitoring, metrics, CV).

---

## Repository structure
```
channelvit-hetwet/
├── channelvit_hetwet/        # package (train/infer stubs)
│   ├── __init__.py
│   ├── train.py
│   └── infer.py
├── configs/                  # YAML config examples
│   └── default.yaml
├── data/                     # put raw/processed data here (ignored)
├── docs/
│   └── model_card.md
├── experiments/              # logs, checkpoints (ignored)
├── notebooks/
│   └── Hetwet_ChannelVIT.ipynb
├── outputs/                  # figures, tables (ignored)
├── tests/
│   └── test_import.py
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── CITATION.cff
├── CONTRIBUTING.md
├── environment.yml
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## What this demonstrates for hiring managers

- **Vision Transformers beyond RGB:** Configurable `in_chans` for multiband inputs (e.g., 4, 8, 13 channels).
- **Experiment hygiene:** Config files, CI, linting, tests, and a deterministic seed.
- **Reproducibility & tracking:** Conda + `requirements.txt`, optional W&B, clean logs/outputs separation.
- **Clarity:** Model card, contribution guide, and a polished README.

Add your actual metrics, confusion matrices, PR/ROC curves, or regression plots under `outputs/` and link them here.

---

## Replace the dummy dataloader with real data

1. Implement a dataset that reads multi-band imagery with `rasterio` or your source of truth (e.g., multiband TIFF stacks):
```python
import rasterio
import torch

class RasterStackDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, y = self.items[idx]
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
        x = torch.from_numpy(arr).float() / 10000.0  # example scaling
        return x, y
```
2. Swap `DummyMultibandDataset` in `channelvit_hetwet/train.py` with your `RasterStackDataset`.

---

## Evaluation
- Classification: accuracy, F1, AUROC, confusion matrix
- Regression: RMSE, MAE, R², residual plots
- Calibration & fairness: reliability diagrams, subgroup metrics

---

## License
MIT © 2025 Sayedmorteza Malaekeh
