import torch
import torch.nn as nn
import torch.optim as optim
import time

from monai.networks.nets import DenseNet121
from Preprocessing.transforms import train_loader, val_loader, small_train_loader, small_val_loader

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Model
# ======================
model = DenseNet121(
    spatial_dims=3,
    in_channels=5,
    out_channels=3
).to(device)

# ======================
# Loss + Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Mixed precision
scaler = torch.amp.GradScaler("cuda")

# ======================
# Training params
# ======================
max_epochs = 30
best_val_acc = 0

start_time = time.time()

# ======================
# Training loop
# ======================
for epoch in range(max_epochs):

    epoch_start = time.time()

    # ===== TRAIN =====
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    # ===== SAVE BEST MODEL =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model")

    # ===== TIMING =====
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = epoch_time * (max_epochs - epoch - 1)

    # ===== LOGGING =====
    print(f"\nEpoch [{epoch+1}/{max_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"Epoch time: {epoch_time:.1f}s | "
          f"Elapsed: {elapsed/60:.1f} min | "
          f"ETA: {remaining/60:.1f} min")