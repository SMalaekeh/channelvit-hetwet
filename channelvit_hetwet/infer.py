import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm

def load_model(ckpt_path: str, in_chans: int = 4, num_classes: int = 2, model_name: str = "vit_base_patch16_224"):
    model = timm.create_model(model_name, pretrained=False, in_chans=in_chans, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def predict_random(model, n=4):
    with torch.no_grad():
        x = torch.randn(n, model.default_cfg.get("in_chans", 3), 224, 224)
        logits = model(x)
        return logits.softmax(dim=1)

if __name__ == "__main__":
    print("Prediction stub - provide a checkpoint to use real weights.")
