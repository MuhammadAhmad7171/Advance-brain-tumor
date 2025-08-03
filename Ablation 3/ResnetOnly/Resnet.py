import os
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

# ─── Reproducibility & Device Setup ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != 'cuda':
    print("No GPU available. Exiting.")
    exit(1)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ─── Hyperparameters & Paths ────────────────────────────────────────────────────
dataset_dir        = '/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/Tumor'
resnet_ckpt_path   = "/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/pretrain/Resnet/resnet50.pth"
batch_size         = 128
accumulation_steps = 4
learning_rate      = 1e-5
num_epochs         = 20
patience           = 3
image_size         = 224
num_folds          = 5
output_dir         = "/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/Paper code/ResNetOnly"
os.makedirs(output_dir, exist_ok=True)

# ─── Data Transforms & Dataset ──────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

full_dataset = datasets.ImageFolder(root=dataset_dir)
num_classes  = len(full_dataset.classes)
targets      = np.array(full_dataset.targets)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, base, idxs, transform):
        self.base = base
        self.idxs = idxs
        self.transform = transform
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        img, label = self.base[self.idxs[i]]
        if self.transform:
            img = self.transform(img)
        return img, label

# ─── Load Pretrained ResNet50 ──────────────────────────────────────────────────
resnet = models.resnet50()
resnet.load_state_dict(torch.load(resnet_ckpt_path, map_location=device))
print("Loaded ResNet50 weights")
# Replace final FC to match our num_classes
resnet.fc = nn.Linear(2048, num_classes)

# ─── Training/Evaluation Setup ─────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
skf       = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_metrics = []
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
    print(f"\n=== Fold {fold}/{num_folds} ===")
    # Split into train/val
    tr_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.1111,
        stratify=targets[trainval_idx],
        random_state=42
    )

    train_ds = CustomDataset(full_dataset, tr_idx, train_transform)
    val_ds   = CustomDataset(full_dataset, val_idx, val_transform)
    test_ds  = CustomDataset(full_dataset, test_idx, val_transform)

    # Weighted sampler for imbalance
    counts  = np.bincount(targets[tr_idx], minlength=num_classes)
    weights = counts.sum() / (counts + 1e-8)
    sample_w = weights[targets[tr_idx]]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(sample_w, len(tr_idx), True)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Fresh model copy for this fold
    model = models.resnet50()
    model.load_state_dict(torch.load(resnet_ckpt_path, map_location=device))
    model.fc = nn.Linear(2048, num_classes)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float('inf')
    early_stop_ctr = 0

    # Per-epoch logs
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    # ─── Train & Validate ──────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = corr = total = 0
        optimizer.zero_grad()

        for i, (imgs, lbls) in enumerate(train_loader, 1):
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            loss = criterion(outs, lbls) / accumulation_steps
            loss.backward()

            if i % accumulation_steps == 0 or i == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * imgs.size(0) * accumulation_steps
            preds = outs.argmax(dim=1)
            corr += (preds == lbls).sum().item()
            total += lbls.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc  = 100 * corr / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        v_loss = v_corr = v_tot = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                v_loss += criterion(outs, lbls).item() * imgs.size(0)
                v_corr += (outs.argmax(1) == lbls).sum().item()
                v_tot += lbls.size(0)

        epoch_val_loss = v_loss / v_tot
        epoch_val_acc  = 100 * v_corr / v_tot
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        scheduler.step()

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss {epoch_train_loss:.4f}, Train Acc {epoch_train_acc:.2f}% | "
            f"Val Loss {epoch_val_loss:.4f}, Val Acc {epoch_val_acc:.2f}%"
        )

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_resnet_fold{fold}.pth"))
            early_stop_ctr = 0
        else:
            early_stop_ctr += 1
            if early_stop_ctr >= patience:
                print("Early stopping.")
                break

    # ─── Test ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_resnet_fold{fold}.pth")))
    model.eval()

    all_preds, all_lbls, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            probs = torch.softmax(outs, dim=1).cpu().numpy()
            preds = outs.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds)
            all_lbls.extend(lbls.cpu().numpy())

    all_probs = np.vstack(all_probs)
    test_acc = 100 * (np.array(all_preds) == np.array(all_lbls)).mean()
    test_auc = roc_auc_score(all_lbls, all_probs, multi_class='ovr')

    fold_metrics.append({
        "fold": fold,
        "train_losses": train_losses,
        "train_accuracies": train_accs,
        "val_losses": val_losses,
        "val_accuracies": val_accs,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "classification_report": classification_report(
            all_lbls, all_preds, target_names=full_dataset.classes, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(all_lbls, all_preds).tolist()
    })

    print(f"Fold {fold} → Test Acc: {test_acc:.2f}%, AUC: {test_auc:.4f}")

# ─── Save JSONs ────────────────────────────────────────────────────────────────
metrics_path = os.path.join(output_dir, f"resnet_metrics_{timestamp}.json")
with open(metrics_path, 'w') as fp:
    json.dump(fold_metrics, fp, indent=4)

accuracies = {f"fold_{m['fold']}": m["test_accuracy"] for m in fold_metrics}
acc_path = os.path.join(output_dir, f"resnet_accuracies_{timestamp}.json")
with open(acc_path, 'w') as fp:
    json.dump(accuracies, fp, indent=4)

print(f"\nSaved full metrics to: {metrics_path}")
print(f"Saved test accuracies to: {acc_path}")
