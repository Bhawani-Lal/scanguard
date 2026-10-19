import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

base_path = r"C:\Users\bhawa\.cache\kagglehub\datasets\ravirajsinh45\real-life-industrial-dataset-of-casting-product\versions\2\casting_data\casting_data"
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Hold out 10% of training images as a validation set.
# A fixed seed means the split is identical every time we run.
val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print("Classes:", full_train_dataset.classes)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))