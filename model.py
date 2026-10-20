import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True):
    # pretrained=True: start from ImageNet weights (for training)
    # pretrained=False: empty network (for loading our own saved weights)
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all pretrained layers so they are not changed at first
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer: 512 features -> 2 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total)
    print("Trainable parameters:", trainable)
    print(model.fc)