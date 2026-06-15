import os
import torch
import timm
from torchvision import transforms
from PIL import Image

MODEL_PATH = os.path.join("models", "best_efficientnet_b3.pth")

# NOTE: keep this list in the SAME order that dataset.classes had during training
CLASS_NAMES = [
    "healthy_maize",
    "healthy_rice",
    "healthy_sugarcane",
    "healthy_wheat",
    "maize_fallArmyWorm",
    "maize_greyLeafSpot",
    "maize_headBlight",
    "maize_leafBlight",
    "maize_mosaicVirus",
    "maize_nitrogenDeficiency",
    "rice_blast",
    "rice_brownLeafSpot",
    "rice_leafBlight",
    "rice_leafStreak",
    "rice_panicleBlight",
    "sugarcane_bacterialBlight",
    "sugarcane_bandedChlorosis",
    "sugarcane_brownSpot",
    "sugarcane_eyeSpot",
    "sugarcane_grassyShoot",
    "sugarcane_leafRust",
    "sugarcane_mosaicVirus",
    "sugarcane_redRot",
    "sugarcane_ringSpots",
    "sugarcane_yellowLeafVirus",
    "wheat_bacterialBlight",
    "wheat_leafBlight",
    "wheat_leafBlotch",
    "wheat_leafRust",
    "wheat_looseSmut",
    "wheat_powderyMildew",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_model():
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Download or place best_efficientnet_b3.pth in models/."
        )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = _build_model()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    conf, idx = torch.max(probs, dim=1)
    return {
        "prediction": CLASS_NAMES[idx.item()],
        "confidence": conf.item(),
        "class_index": idx.item(),
    }
