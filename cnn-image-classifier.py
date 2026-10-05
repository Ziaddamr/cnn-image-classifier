import os
import random
import glob
import gc
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ══════════════════════════════════════════════════════════════════
#  PREMIUM CONFIGURATION
# ══════════════════════════════════════════════════════════════════
SEED = 42
SCENE_ROOT = "/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors"
TRAIN_DIR = os.path.join(SCENE_ROOT, "train")
TEST_DIR = os.path.join(SCENE_ROOT, "test")

# YOUR SPECIFIC DATASET PATH
WEIGHTS_PATH = "/kaggle/input/datasets/yyoussefshawky/shawky/eva02_large_weights.pt"

IMG_SIZE = 224
BATCH_SIZE = 8
ACC_STEPS = 4    # Effective batch of 32 for high stability
EPOCHS = 16


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
#  ELITE PREPROCESSING (BICUBIC + ANTI-OVERFIT)
# ══════════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    # Bicubic preserves texture details better than standard Bilinear
    transforms.RandomResizedCrop(IMG_SIZE, scale=(
        0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # High defense against memorization
    transforms.RandomErasing(p=0.3, value='random'),
])

test_transform = transforms.Compose([
    transforms.Resize(
        IMG_SIZE + 32, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════
#  EVA-02 LARGE MODEL SETUP
# ══════════════════════════════════════════════════════════════════


class UltimateEVAModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # drop_path_rate=0.3 adds Stochastic Depth to stop overfitting
        self.model = timm.create_model('eva02_large_patch14_224',
                                       pretrained=False,
                                       num_classes=num_classes,
                                       drop_path_rate=0.3)

        if os.path.exists(WEIGHTS_PATH):
            print(f"💎 Loading Weights from: {WEIGHTS_PATH}")
            sd = torch.load(WEIGHTS_PATH, map_location='cpu')
            if 'model' in sd:
                sd = sd['model']
            if 'state_dict' in sd:
                sd = sd['state_dict']
            self.model.load_state_dict(sd, strict=False)
            print("✅ Weights successfully integrated.")
        else:
            print("⚠️ Weights not found! Downloading official weights...")
            self.model = timm.create_model(
                'eva02_large_patch14_224', pretrained=True, num_classes=num_classes)

    def forward(self, x): return self.model(x)


# ══════════════════════════════════════════════════════════════════
#  TRAINING INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════
full_dataset = datasets.ImageFolder(TRAIN_DIR)
num_classes = len(full_dataset.classes)
train_idx, val_idx = train_test_split(np.arange(len(
    full_dataset)), test_size=0.12, stratify=full_dataset.targets, random_state=SEED)

train_loader = DataLoader(Subset(datasets.ImageFolder(TRAIN_DIR, train_transform), train_idx),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(Subset(datasets.ImageFolder(TRAIN_DIR, test_transform), val_idx),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = UltimateEVAModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# LLRD: Delicate backbone, aggressive head
optimizer = torch.optim.AdamW([
    {'params': model.model.patch_embed.parameters(), 'lr': 1e-6},
    {'params': model.model.blocks.parameters(), 'lr': 1e-5},
    {'params': model.model.head.parameters(), 'lr': 4e-4},
], weight_decay=0.05)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
scaler = GradScaler('cuda')

print("\n🚀 MISSION START: TRAINING EVA-02 LARGE")
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for i, (imgs, lbls) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        imgs, lbls = imgs.to(device), lbls.to(device)
        with autocast('cuda'):
            preds = model(imgs)
            loss = criterion(preds, lbls) / ACC_STEPS

        scaler.scale(loss).backward()
        if (i + 1) % ACC_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            correct += (out.argmax(1) == lbls).sum().item()
            total += lbls.size(0)

    v_acc = correct / total
    print(f"✨ Validation Accuracy: {v_acc:.4f}")
    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), "best_eva_final.pth")
        print("⭐ BEST SCORE UPDATED!")
    scheduler.step()

# ══════════════════════════════════════════════════════════════════
#  SAFE INFERENCE (CRASH-PROOF)
# ══════════════════════════════════════════════════════════════════
print("\n📸 GENERATING FINAL SUBMISSION")
if os.path.exists("best_eva_final.pth"):
    model.load_state_dict(torch.load("best_eva_final.pth"))

model.eval()
test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.*")))
results = []

with torch.no_grad():
    for path in tqdm(test_paths):
        fname = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
            img_t = test_transform(img).unsqueeze(0).to(device)
            with autocast('cuda'):
                out = model(img_t)
            results.append({"ImageName": fname, "label": out.argmax(1).item()})
        except:
            results.append({"ImageName": fname, "label": 0})

pd.DataFrame(results).to_csv("submission.csv", index=False)
print("✅ SUBMISSION.CSV CREATED SUCCESSFULLY.")
