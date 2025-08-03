import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from transformers import ViTModel, ViTConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import random
import json
from datetime import datetime

# ─── Reproducibility & Device Setup ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Force standard attention kernels if adaptive SDPA causes issues
os.environ["TRANSFORMERS_NO_ADAPTIVE_ATTENTION"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != 'cuda':
    print("No GPUs available. Exiting.")
    exit(1)
n_gpus = torch.cuda.device_count()
print(f"Using {n_gpus} GPU(s): {torch.cuda.get_device_name(0)}")

# ─── Hyperparameters & Paths ────────────────────────────────────────────────────
dataset_dir        = '/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/Tumor'
batch_size         = 128
accumulation_steps = 4
learning_rate      = 1e-5
num_epochs         = 20       # reduced from 30
aligned_dim        = 512
patience           = 3        # reduced from 5
image_size         = 224
num_folds          = 5
output_dir         = "/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/Paper code/Ablation 1"
os.makedirs(output_dir, exist_ok=True)

# ─── Data Transforms & Dataset ──────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomPerspective(0.2, p=0.5),
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
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        img, label = self.base[self.idxs[i]]
        if self.transform:
            img = self.transform(img)
        return img, label

# ─── Cross-Gated Multi-Path Attention Fusion (CG-MAF) Module ─────────────────────
class CGMAttention(nn.Module):
    def __init__(self, dim, num_classes):
        super(CGMAttention, self).__init__()
        # Gate to attend to features from ResNet and ViT
        self.resnet_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.vit_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # Fully connected layers to fuse features
        self.fc = nn.Sequential(
            nn.Linear(dim*2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, resnet_features, vit_features):
        # Apply gates to each modality
        resnet_attention = self.resnet_gate(resnet_features)
        vit_attention = self.vit_gate(vit_features)

        # Compute cross-attention
        resnet_attention = resnet_attention * vit_features
        vit_attention = vit_attention * resnet_features

        # Fuse the attention-modulated features
        fused = torch.cat([resnet_attention, vit_attention], dim=1)
        return self.fc(fused)

# ─── HybridNaiveFusionModel (Updated with CG-MAF) ─────────────────────────────────
class HybridCGMAFModel(nn.Module):
    def __init__(self, resnet, vit, fusion_module, aligned_dim):
        super().__init__()
        resnet.fc = nn.Identity()  # Removing the final FC layer from ResNet
        self.resnet = resnet
        self.vit    = vit
        self.resnet_proj = nn.Sequential(
            nn.Linear(2048, aligned_dim),
            nn.Dropout(0.3),  # increased dropout
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(768, aligned_dim),
            nn.Dropout(0.3),  # increased dropout
        )
        self.fusion = fusion_module

    def forward(self, x):
        resnet_feats = self.resnet_proj(self.resnet(x))
        vit_feats = self.vit_proj(self.vit(x).last_hidden_state[:,0,:])
        return self.fusion(resnet_feats, vit_feats)

# ─── Load Pretrained Backbones ──────────────────────────────────────────────────
resnet = models.resnet50()
resnet.load_state_dict(torch.load(
    "/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/pretrain/Resnet/resnet50.pth", map_location=device))
print("Loaded ResNet50")

vit_dir = "/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/pretrain/Vit"
vit_config = ViTConfig.from_pretrained(vit_dir, local_files_only=True)
vit        = ViTModel.from_pretrained(vit_dir, local_files_only=True)
print("Loaded ViT")

# ─── Training/Evaluation Setup ──────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
skf       = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_metrics = []
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
    print(f"\n=== Fold {fold}/{num_folds} ===")
    tr_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.1111,
        stratify=targets[trainval_idx], random_state=42
    )

    train_ds = CustomDataset(full_dataset, tr_idx, train_transform)
    val_ds   = CustomDataset(full_dataset, val_idx,   val_transform)
    test_ds  = CustomDataset(full_dataset, test_idx,  val_transform)

    counts  = np.bincount(targets[tr_idx], minlength=num_classes)
    weights = counts.sum() / (counts + 1e-8)
    samp_w  = weights[targets[tr_idx]]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=WeightedRandomSampler(samp_w, len(tr_idx), True)
    )
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    fusion_layer = CGMAttention(aligned_dim, num_classes)
    model = HybridCGMAFModel(resnet, vit, fusion_layer, aligned_dim).to(device)

    # Multi-GPU
    if n_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float('inf')
    patience_ctr  = 0

    # ─── Train & Validate ──────────────────────────────────────────────────────
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss, total = 0., 0
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
            total += lbls.size(0)

        # Validation
        model.eval()
        v_loss, v_tot = 0., 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                v_loss += criterion(outs, lbls).item() * imgs.size(0)
                v_tot  += lbls.size(0)

        scheduler.step()
        val_loss = v_loss / v_tot
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss {running_loss/total:.4f} | Val Loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_naive_fold{fold}.pth")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping.")
                break

    # ─── Test ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(f"best_naive_fold{fold}.pth"))
    model.eval()

    all_preds, all_lbls, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            probs = F.softmax(outs, dim=1).cpu().numpy()
            preds = outs.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds)
            all_lbls.extend(lbls.cpu().numpy())
    all_probs = np.vstack(all_probs)

    test_acc = 100 * (np.array(all_preds) == np.array(all_lbls)).mean()
    test_auc = roc_auc_score(all_lbls, all_probs, multi_class='ovr')

    fold_metrics.append({
        "fold": fold,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "classification_report": classification_report(
            all_lbls, all_preds, target_names=full_dataset.classes, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(all_lbls, all_preds).tolist()
    })
    print(f"Fold {fold} → Test Acc: {test_acc:.2f}%, AUC: {test_auc:.4f}")

# ─── Save JSONs ────────────────────────────────────────────────────────────────
metrics_path = os.path.join(output_dir, f"metrics_all_folds_{timestamp}.json")
with open(metrics_path, 'w') as fp:
    json.dump(fold_metrics, fp, indent=4)

accuracies = {f"fold_{m['fold']}": m["test_accuracy"] for m in fold_metrics}
acc_path = os.path.join(output_dir, f"test_accuracies_{timestamp}.json")
with open(acc_path, 'w') as fp:
    json.dump(accuracies, fp, indent=4)

print(f"\nSaved full metrics: {metrics_path}")
print(f"Saved test accuracies: {acc_path}")
