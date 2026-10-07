import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

base_path = r"C:\Users\bhawa\.cache\kagglehub\datasets\ravirajsinh45\real-life-industrial-dataset-of-casting-product\versions\2\casting_data\casting_data"
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print("Classes:", train_dataset.classes)
    print("Class to index:", train_dataset.class_to_idx)
    print("Train images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)