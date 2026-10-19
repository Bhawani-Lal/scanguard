import os
import torch
import matplotlib.pyplot as plt
from data_loader import test_loader, test_dataset
from model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained model (no retraining)
model = build_model()
model.load_state_dict(torch.load("scanguard_model.pth", map_location=device))
model.to(device)
model.eval()

# Run the model on every test image
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

classes = test_dataset.classes   # ['def_front', 'ok_front']
DEF, OK = 0, 1                   # defective is class 0

# Count the four outcomes
tp = sum(1 for pred, label in zip(all_preds, all_labels) if label == DEF and pred == DEF)  # defect caught
fn = sum(1 for pred, label in zip(all_preds, all_labels) if label == DEF and pred == OK)   # defect MISSED
fp = sum(1 for pred, label in zip(all_preds, all_labels) if label == OK and pred == DEF)   # false alarm
tn = sum(1 for pred, label in zip(all_preds, all_labels) if label == OK and pred == OK)    # good part passed

total = tp + fn + fp + tn
accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("=== Confusion matrix ===")
print("                     Predicted DEF   Predicted OK")
print(f"Actual DEF (defect)       {tp:5d}          {fn:5d}   <- missed defects")
print(f"Actual OK  (good)         {fp:5d}          {tn:5d}")
print()
print("=== Metrics (defective = positive class) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 score : {f1:.4f}")
print()

# List every image the model got wrong
print("=== Misclassified images ===")
for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
    if pred != label:
        filename = os.path.basename(test_dataset.samples[i][0])
        print(f"{filename}: actual={classes[label]}, predicted={classes[pred]}")

# Save the confusion matrix as an image
matrix = [[tp, fn], [fp, tn]]
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(matrix, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted DEF", "Predicted OK"])
ax.set_yticklabels(["Actual DEF", "Actual OK"])
for r in range(2):
    for c in range(2):
        ax.text(c, r, matrix[r][c], ha="center", va="center", fontsize=14)
ax.set_title("ScanGuard confusion matrix (test set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved to confusion_matrix.png")