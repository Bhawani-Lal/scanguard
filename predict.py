import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import build_model

CLASSES = ["def_front", "ok_front"]
DEF = 0

# Same preprocessing as training (must match exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path="scanguard_model.pth"):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def load_threshold(path="threshold.txt"):
    with open(path) as f:
        return float(f.read().strip())

def looks_like_casting(image):
    """Heuristic check: training images are square and greyscale.
    Rejects clearly different inputs (colour photos, portrait documents).
    Not a guarantee: a square greyscale photo of something else would still pass."""
    import numpy as np
    width, height = image.size
    if not 0.8 <= width / height <= 1.25:
        return False
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    colourfulness = np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2]))
    return colourfulness < 15


def predict(image, model, threshold):
    """Take a PIL image, return (label, probability of defective)."""
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)   # add batch dimension: [1, 3, 224, 224]
    with torch.no_grad():
        outputs = model(tensor)
        p_def = F.softmax(outputs, dim=1)[0, DEF].item()
    label = "DEFECTIVE" if p_def >= threshold else "OK"
    return label, p_def


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    model = load_model()
    threshold = load_threshold()
    image = Image.open(sys.argv[1])
    label, p_def = predict(image, model, threshold)

    print(f"Prediction: {label}")
    print(f"Probability of defect: {p_def:.4f}  (threshold {threshold})")