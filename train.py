import torch
import torch.nn as nn
from data_loader import train_loader, val_loader
from model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on device:", device)

model = build_model().to(device)

# Loss function: how wrong the model is
criterion = nn.CrossEntropyLoss()

# Optimiser: only updates the new final layer (the rest is frozen)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_num, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if batch_num % 50 == 0:
            print(f"  epoch {epoch + 1}, batch {batch_num}/{len(train_loader)}")

    print(f"Epoch {epoch + 1}/{EPOCHS} - "
          f"loss: {running_loss / total:.4f}, "
          f"train accuracy: {correct / total:.4f}")

# Check on the VALIDATION set (the test set stays untouched until the end)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

print(f"Validation accuracy: {correct / total:.4f}")

torch.save(model.state_dict(), "scanguard_model.pth")
print("Model saved to scanguard_model.pth")