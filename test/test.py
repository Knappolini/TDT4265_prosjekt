from monai.data import Dataset, DataLoader
import torch
import pandas as pd
from monai.networks.nets import DenseNet121
from Preprocessing.transforms import test_ds, test_loader

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Model (MATCH TRAINING)
# ======================
model = DenseNet121(
    spatial_dims=3,
    in_channels=5,
    out_channels=3
).to(device)

# ======================
# Load trained weights
# ======================
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ======================
# Inference
# ======================
results = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)

        # 👇 IMPORTANT: your dataset must include "id"
        ids = batch["id"]

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        probs = probs.cpu().numpy()

        for i in range(len(ids)):
            results.append({
                "ID": ids[i],
                "normal": float(probs[i][0]),
                "benign": float(probs[i][1]),
                "malignant": float(probs[i][2]),
            })

# ======================
# Save CSV
# ======================
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv")