import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from transformers import ViTModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from datetime import datetime
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.cuda.amp import autocast, GradScaler

# ─── Reproducibility & Device Setup ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,6"

device = torch.device("cuda")
n_gpus = torch.cuda.device_count()
print(f"Using {n_gpus} GPU(s): {torch.cuda.get_device_name(0)}")

# ─── Hyperparameters & Paths ────────────────────────────────────────────────────
dataset_dir        = '/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/Tumor'
batch_size         = 256
accumulation_steps = 4
learning_rate      = 1e-4
num_epochs         = 30
aligned_dim        = 512
patience           = 5
image_size         = 224
num_folds          = 5
num_gradcam_images = 5
lambda_gcl         = 0.5
output_dir         = "/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/Paper code/Ablation2/R+V+CGMAF+Loss"
os.makedirs(output_dir, exist_ok=True)

# ─── Data Augmentation & Transforms ─────────────────────────────────────────────
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
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─── Load Dataset ───────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(root=dataset_dir)
num_classes  = len(full_dataset.classes)
targets      = np.array(full_dataset.targets)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label

# ─── CG-MAF Fusion Layer ────────────────────────────────────────────────────────
class CGMAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_r_to_v = nn.Linear(dim, dim)
        self.gate_v_to_r = nn.Linear(dim, dim)
    def forward(self, r, v, return_gates=False):
        g_rv = torch.sigmoid(self.gate_r_to_v(r))
        g_vr = torch.sigmoid(self.gate_v_to_r(v))
        v2 = g_rv * v
        r2 = g_vr * r
        out = v2 + r2
        if return_gates:
            return out, g_rv.mean(dim=1), g_vr.mean(dim=1)
        return out

# ─── Hybrid Model ───────────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, resnet, vit, fusion, num_classes, dim):
        super().__init__()
        resnet.fc = nn.Identity()
        self.resnet   = resnet
        self.vit      = vit
        self.resproj  = nn.Linear(2048, dim)
        self.vitproj  = nn.Linear(768, dim)
        self.fusion   = fusion
        self.fc       = nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Dropout(0.6), nn.Linear(256,num_classes))
        self.vit_head = nn.Linear(dim,num_classes)
        self.res_head = nn.Linear(dim,num_classes)

    def forward(self, x, return_gates=False, return_aux=False):
        r = self.resproj(self.resnet(x))
        v = self.vitproj(self.vit(x).last_hidden_state[:,0,:])
        if return_gates or return_aux:
            fused, gv, gr = self.fusion(r, v, return_gates=True)
        else:
            fused = self.fusion(r, v)
        main = self.fc(fused)
        if return_aux:
            return main, self.vit_head(v), self.res_head(r), gv, gr
        if return_gates:
            return main, gv, gr
        return main

# ─── Load Pretrained Backbones ──────────────────────────────────────────────────
resnet = models.resnet50()
resnet.load_state_dict(torch.load("/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/pretrain/Resnet/resnet50.pth", map_location=device))
vit    = ViTModel.from_pretrained("/home/qitam/sdb2/home/qiteam_project/ahmad/BrainTumor/pretrain/Vit", local_files_only=True)

# ─── Training/Evaluation Setup ──────────────────────────────────────────────────
criterion    = nn.CrossEntropyLoss()
skf          = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
scaler       = GradScaler()
fold_metrics = []
ts           = datetime.now().strftime("%Y%m%d_%H%M%S")

for fold, (train_idx, test_idx) in enumerate(skf.split(targets, targets), 1):
    print(f"\n--- Fold {fold}/{num_folds} ---")
    tr, val = train_test_split(train_idx, test_size=0.1111, stratify=targets[train_idx], random_state=42)
    dl_tr = DataLoader(CustomDataset(full_dataset, tr, train_transform),
                       batch_size=batch_size,
                       sampler=WeightedRandomSampler(
                           weights=(np.bincount(targets[tr], minlength=num_classes).sum()/
                                    (np.bincount(targets[tr], minlength=num_classes)+1e-8))[targets[tr]],
                           num_samples=len(tr), replacement=True))
    dl_val = DataLoader(CustomDataset(full_dataset, val, test_transform),
                        batch_size=batch_size, shuffle=False)
    dl_te  = DataLoader(CustomDataset(full_dataset, test_idx, test_transform),
                        batch_size=batch_size, shuffle=False)

    model = HybridModel(resnet, vit, CGMAF(aligned_dim), num_classes, aligned_dim).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)

    opt       = optim.AdamW(model.parameters(), lr=learning_rate)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    best_val  = float('inf')
    patience  = 0
    history   = {"tr_loss":[], "tr_acc":[], "val_loss":[], "val_acc":[]}

    # ── Train & Validate ──────────────────────────────────────────────────────
    for ep in range(num_epochs):
        model.train()
        rloss, rcorr, rtot = 0., 0, 0
        opt.zero_grad()
        for i, (x, y) in enumerate(dl_tr, 1):
            x, y = x.to(device), y.to(device)
            with autocast():
                out, vl, rl, gv, gr = model(x, return_aux=True)
                ce    = criterion(out, y)
                tprob = F.softmax(out, dim=1)
                klv   = gv.unsqueeze(1) * F.kl_div(F.log_softmax(vl,1), tprob, reduction='batchmean')
                klr   = gr.unsqueeze(1) * F.kl_div(F.log_softmax(rl,1), tprob, reduction='batchmean')
                loss  = (ce + lambda_gcl * (klv+klr).mean()) / accumulation_steps
            scaler.scale(loss).backward()
            if i % accumulation_steps == 0 or i == len(dl_tr):
                scaler.step(opt); scaler.update(); opt.zero_grad()
            rloss += ce.item() * y.size(0)
            preds   = out.argmax(1); rcorr += (preds==y).sum().item(); rtot += y.size(0)

        history["tr_loss"].append(rloss/rtot)
        history["tr_acc"].append(100*rcorr/rtot)

        model.eval()
        vloss, vcorr, vtot = 0., 0, 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                out = model(x)
                l   = criterion(out, y)
                vloss += l.item() * y.size(0)
                p     = out.argmax(1); vcorr += (p==y).sum().item(); vtot += y.size(0)

        history["val_loss"].append(vloss/vtot)
        history["val_acc"].append(100*vcorr/vtot)
        sched.step()

        print(f"Epoch {ep+1}/{num_epochs} | "
              f"Tr {history['tr_loss'][-1]:.4f}-{history['tr_acc'][-1]:.1f}% | "
              f"Va {history['val_loss'][-1]:.4f}-{history['val_acc'][-1]:.1f}%")

        if history["val_loss"][-1] < best_val:
            best_val = history["val_loss"][-1]
            torch.save(model.state_dict(), f"best_{fold}.pth")
            patience = 0
        else:
            patience += 1
            if patience >= patience:
                print("Early stopping.")
                break

    # ── Test & Metrics ────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(f"best_{fold}.pth"))
    model.eval()
    AP, AA, PM, GV, GR = [], [], [], [], []
    with torch.no_grad():
        for x, y in dl_te:
            x, y = x.to(device), y.to(device)
            out, gv, gr = model(x, return_gates=True)
            pr = F.softmax(out, dim=1).cpu().numpy()
            p  = out.argmax(1).cpu().numpy()
            AA.extend(p); AP.extend(y.cpu().numpy()); PM.extend(pr)
            GV.extend(gv.cpu().numpy()); GR.extend(gr.cpu().numpy())

    AA, AP, PM = np.array(AA), np.array(AP), np.vstack(PM)
    acc = 100*(AA==AP).mean()
    auc = roc_auc_score(AP, PM, multi_class='ovr')
    cm  = confusion_matrix(AP, AA)

    fold_metrics.append({
        "fold": fold,
        "train_losses": history["tr_loss"],
        "val_losses": history["val_loss"],
        "train_accuracies": history["tr_acc"],
        "val_accuracies": history["val_acc"],
        "test_accuracy": acc,
        "test_auc": auc,
        "classification_report": classification_report(AP, AA, output_dict=True),
        "confusion_matrix": cm.tolist()
    })

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    def visualize_gradcam(mdl, loader, classes, n, outd, fd):
        real = mdl.module if isinstance(mdl, nn.DataParallel) else mdl
        layer = real.resnet.layer4[-1].conv3
        cam   = GradCAM(model=real, target_layers=[layer])
        inv   = transforms.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
                                      std =[1/0.229,1/0.224,1/0.225])
        gdir  = os.path.join(outd, f"gradcam_{fd}"); os.makedirs(gdir, exist_ok=True)

        for ci, name in enumerate(classes):
            cnt = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                for i in range(len(y)):
                    if y[i]==ci and cnt<n:
                        img = inv(x[i]).cpu().numpy().transpose(1,2,0)
                        img = np.clip(img,0,1)
                        gray=cam(input_tensor=x[i].unsqueeze(0), targets=[ClassifierOutputTarget(ci)])[0]
                        vis = show_cam_on_image(img, gray, use_rgb=True)
                        plt.imsave(os.path.join(gdir, f"{name}_{cnt}.png"), vis)
                        cnt+=1
                if cnt>=n: break

    visualize_gradcam(model, dl_te, full_dataset.classes, num_gradcam_images, output_dir, fold)

# ─── Save JSON ───────────────────────────────────────────────────────────────
with open(os.path.join(output_dir, f"metrics_{ts}.json"), "w") as f:
    json.dump(fold_metrics, f, indent=2)

print("Done. Outputs saved to", output_dir)
