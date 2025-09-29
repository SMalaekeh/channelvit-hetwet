import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import timm

@dataclass
class Config:
    data_dir: str = "data/"
    channels: int = 4  # number of spectral channels
    img_size: int = 224
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 1
    lr: float = 3e-4
    model_name: str = "vit_base_patch16_224"
    num_classes: int = 2
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42
    use_wandb: bool = False
    project: str = "channelvit-hetwet"
    run_name: Optional[str] = None

class DummyMultibandDataset(Dataset):
    """Replace this with your real multiband loader (e.g., rasterio stacks)."""
    def __init__(self, root: str, size: int = 224, n: int = 64, channels: int = 4):
        self.root = root
        self.n = n
        self.tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.channels = channels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Fake multi-channel image: (C, H, W)
        x = torch.randn(self.channels, 224, 224)
        y = torch.randint(0, 2, (1,)).item()
        return x, y

class LitChannelViT(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        # create a ViT that accepts arbitrary in_chans
        self.model = timm.create_model(cfg.model_name, pretrained=False, in_chans=cfg.channels, num_classes=cfg.num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.cfg.lr)

def main():
    cfg = Config()
    L.seed_everything(cfg.seed)
    ds = DummyMultibandDataset(cfg.data_dir, size=cfg.img_size, channels=cfg.channels)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    model = LitChannelViT(cfg)

    logger = WandbLogger(project=cfg.project, name=cfg.run_name) if cfg.use_wandb else None
    trainer = L.Trainer(max_epochs=cfg.max_epochs, accelerator=cfg.accelerator, devices=cfg.devices, logger=logger)
    trainer.fit(model, dl)

if __name__ == "__main__":
    main()
