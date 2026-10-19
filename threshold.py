import torch
import torch.nn.functional as F
from data_loader import val_loader, test_loader
from model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model()
model.load_state_dict(torch.load("scanguard_model.pth", map_location=device))
model.to(device)
model.eval()

DEF = 0  # defective is class 0


def get_defect_probs(loader):
    """Return the model's probability of 'defective' for each image, plus true labels."""
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            outputs = model(images.to(device))
            p_def = F.softmax(outputs, dim=1)[:, DEF]
            probs.extend(p_def.cpu().tolist())
            labels.extend(y.tolist())
    return probs, labels


def metrics(probs, labels, threshold):
    """Flag as defective if P(defective) >= threshold."""
    tp = fn = fp = tn = 0
    for p, label in zip(probs, labels):
        predicted_def = p >= threshold
        actual_def = label == DEF
        if actual_def and predicted_def:
            tp += 1
        elif actual_def and not predicted_def:
            fn += 1
        elif not actual_def and predicted_def:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    accuracy = (tp + tn) / len(labels)
    return tp, fn, fp, tn, precision, recall, accuracy


# 1. Sweep thresholds on the VALIDATION set
val_probs, val_labels = get_defect_probs(val_loader)
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

print("=== Validation set: threshold sweep ===")
print("Threshold  Missed  FalseAlarm  Precision  Recall")
results = []
for t in thresholds:
    tp, fn, fp, tn, prec, rec, acc = metrics(val_probs, val_labels, t)
    results.append((t, prec, rec))
    print(f"  {t:.2f}      {fn:4d}     {fp:4d}       {prec:.4f}    {rec:.4f}")

# 2. Pick: best recall with precision >= 0.97 (ties -> higher threshold)
MIN_PRECISION = 0.97
candidates = [r for r in results if r[1] >= MIN_PRECISION]
best_t = max(candidates, key=lambda r: (r[2], r[0]))[0]
print(f"\nChosen threshold (best recall with precision >= {MIN_PRECISION}): {best_t:.2f}")

# 3. Final evaluation on the TEST set, used once
test_probs, test_labels = get_defect_probs(test_loader)
print("\n=== Test set: final result ===")
for name, t in [("Default 0.50", 0.50), (f"Tuned {best_t:.2f}", best_t)]:
    tp, fn, fp, tn, prec, rec, acc = metrics(test_probs, test_labels, t)
    print(f"{name}: missed={fn}, false alarms={fp}, "
          f"accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}")

# 4. Save for the app
with open("threshold.txt", "w") as f:
    f.write(str(best_t))
print(f"\nSaved threshold {best_t:.2f} to threshold.txt")