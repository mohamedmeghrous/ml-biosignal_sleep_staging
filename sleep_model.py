# -*- coding: utf-8 -*-
# BEACON BIOSIGNALS 2025 
# Architecture: ResNet18 (Stride Hacked) + BiLSTM
# Inference: ENSEMBLE SIMPLE + ARGMAX 


import os
import sys
import glob
import random
import copy
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# ================= 1. INSTALLATION =================
print(">>> INSTALLATION TIMM...")
os.system('pip install -q timm')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
import timm

warnings.filterwarnings("ignore")

# ================= 2. CONFIGURATION KAGGLE =================
CONFIG = {
    # Chemin Kaggle standard
    "root_dir": "/kaggle/input/beacon-biosignals-sleep-staging-2025",
    "epochs": 20,
    "folds": 5,               # 5 Folds pour la robustesse
    "batch_size": 16,         
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "num_workers": 2,
    "seed": 2025,
    "n_classes": 5,
    "n_channels": 5,
    "fs": 100.0,
    "n_mels": 64,
    "spec_size": (64, 64),
    "context": 10,
    "mixup_alpha": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ================= 3. PROCESSING =================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=int(CONFIG['fs']), n_fft=400, hop_length=50, 
    n_mels=CONFIG['n_mels'], f_min=0.0, f_max=40.0
).to(CONFIG['device'])

amp_to_db = torchaudio.transforms.AmplitudeToDB().to(CONFIG['device'])

def to_spectrogram_batch(sig_tensor):
    with torch.no_grad():
        spec = mel_transform(sig_tensor)
        spec = amp_to_db(spec)
        spec = F.interpolate(spec, size=CONFIG['spec_size'], mode='bilinear', align_corners=False)
        mean = spec.mean(dim=(2, 3), keepdim=True)
        std = spec.std(dim=(2, 3), keepdim=True)
        spec = (spec - mean) / (std + 1e-6)
    return spec

class SleepDataset(Dataset):
    def __init__(self, record_ids, sig_map, tgt_map=None, mode="train"):
        self.mode = mode
        self.records = []
        self.index_map = []
        
        for rid in tqdm(record_ids, desc=f"Loading {mode}"):
            try:
                sig = np.load(sig_map[rid])
                if sig.shape[0] != CONFIG['n_channels']: sig = sig.T
                sig = np.clip(sig, -150, 150)
                n_epochs = sig.shape[1] // 3000
                # On stocke le brut pour économiser la RAM sur Kaggle
                raw_data = sig[:, :n_epochs*3000] 
                
                targets = None
                if mode == "train":
                    targets = np.load(tgt_map[rid]).astype(np.int64)
                    targets = targets[:n_epochs]
                    
                self.records.append({"raw": raw_data, "targets": targets, "rid": rid, "n_epochs": n_epochs})
                for i in range(n_epochs):
                    if mode == "train" and targets[i] == -1: continue
                    self.index_map.append((len(self.records)-1, i))
            except: pass
                
    def __len__(self): return len(self.index_map)
    
    def __getitem__(self, idx):
        rec_idx, center = self.index_map[idx]
        rec = self.records[rec_idx]
        c_size = CONFIG['context']
        start_ep = max(0, center - c_size)
        end_ep = min(rec['n_epochs'], center + c_size + 1)
        
        sig_seq = rec['raw'][:, start_ep*3000 : end_ep*3000]
        sig_tensor = torch.tensor(sig_seq).float()
        n_actual = sig_tensor.shape[1] // 3000
        epochs = sig_tensor.view(CONFIG['n_channels'], n_actual, 3000).permute(1, 0, 2)
        
        pad_left = (c_size - (center - start_ep))
        pad_right = (c_size - (end_ep - center - 1))
        if pad_left > 0:
            epochs = torch.cat([epochs[0:1].repeat(pad_left, 1, 1), epochs], dim=0)
        if pad_right > 0:
            epochs = torch.cat([epochs, epochs[-1:].repeat(pad_right, 1, 1)], dim=0)
            
        y = rec["targets"][center] if self.mode == "train" else -1
        return epochs, y, (rec["rid"], center)

# ================= 4. MODEL =================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SleepModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')
        
        w_rgb = self.cnn.conv1.weight.clone()
        self.cnn.conv1 = nn.Conv2d(CONFIG['n_channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.cnn.conv1.weight.copy_(w_rgb.mean(dim=1, keepdim=True).repeat(1, CONFIG['n_channels'], 1, 1))
            
        self.cnn.layer4[0].conv1.stride = (1, 1)
        self.cnn.layer4[0].downsample[0].stride = (1, 1)
        
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        b, s, c, f, t = x.shape
        x = x.view(b * s, c, f, t)
        feats = self.cnn(x)       
        feats = self.se(feats)
        feats = self.pool(feats).flatten(1) 
        feats = feats.view(b, s, -1)
        out, _ = self.lstm(feats)
        center = out[:, s // 2, :]
        return self.fc(self.dropout(center))

# ================= 5. UTILS =================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=5, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = -1
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ================= 6. TRAINING =================
def find_files():
    train_sig, train_tgt, test_sig = {}, {}, {}
    print(f"Scanning: {CONFIG['root_dir']}")
    # Scan robuste pour Kaggle
    for root, _, files in os.walk(CONFIG['root_dir']):
        for f in files:
            if not f.endswith(".npy"): continue
            path = os.path.join(root, f)
            try:
                rid = int(f.split("_")[1].split(".")[0])
                if "signal" in f:
                    if "train" in path or "train" in root: train_sig[rid] = path
                    else: test_sig[rid] = path
                elif "target" in f:
                    train_tgt[rid] = path
            except: pass
    t_ids = sorted(list(set(train_sig.keys()) & set(train_tgt.keys())))
    test_ids = sorted(list(test_sig.keys()))
    print(f"Dataset: {len(t_ids)} Train | {len(test_ids)} Test")
    return t_ids, test_ids, train_sig, train_tgt, test_sig

def train_one_fold(train_ds, val_ds, fold):
    loader_t = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], drop_last=True)
    loader_v = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    model = SleepModel().to(CONFIG['device'])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['lr'], steps_per_epoch=len(loader_t), epochs=CONFIG['epochs'], pct_start=0.3)
    crit = LabelSmoothingLoss()
    scaler = GradScaler()
    
    best_acc, best_state = 0.0, None
    print(f"\n=== Fold {fold} ===")
    
    for ep in range(CONFIG['epochs']):
        model.train()
        l_sum = 0
        pbar = tqdm(loader_t, desc=f"Ep {ep+1}", leave=False)
        for raw, y, _ in pbar:
            raw, y = raw.to(CONFIG['device']), y.to(CONFIG['device'])
            b, s, c, t = raw.shape
            specs = to_spectrogram_batch(raw.view(b*s, c, t))
            X = specs.view(b, s, c, CONFIG['spec_size'][0], CONFIG['spec_size'][1])
            
            opt.zero_grad()
            with autocast():
                if random.random() < 0.6:
                    X_mix, ya, yb, lam = mixup_data(X, y, CONFIG['mixup_alpha'])
                    out = model(X_mix)
                    loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
                else:
                    out = model(X)
                    loss = crit(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            l_sum += loss.item()
            
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for raw, y, _ in loader_v:
                raw, y = raw.to(CONFIG['device']), y.to(CONFIG['device'])
                b, s, c, t = raw.shape
                specs = to_spectrogram_batch(raw.view(b*s, c, t))
                X = specs.view(b, s, c, CONFIG['spec_size'][0], CONFIG['spec_size'][1])
                with autocast(): out = model(X)
                pred = out.argmax(1)
                corr += (pred == y).sum().item()
                tot += y.size(0)
        
        acc = corr / tot
        print(f"Ep {ep+1} | Loss: {l_sum/len(loader_t):.4f} | Val: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            
    return best_state

# ================= 7. EXECUTION (NO HMM) =================
t_ids, test_ids, sig_map, tgt_map, test_sig_map = find_files()

if len(t_ids) == 0:
    print("ERREUR: Dataset non trouvé. Vérifie le chemin 'root_dir'.")
    sys.exit()

kf = KFold(n_splits=CONFIG['folds'], shuffle=True, random_state=CONFIG['seed'])
models = []

for f, (t_idx, v_idx) in enumerate(kf.split(t_ids)):
    ds_t = SleepDataset([t_ids[i] for i in t_idx], sig_map, tgt_map, "train")
    ds_v = SleepDataset([t_ids[i] for i in v_idx], sig_map, tgt_map, "train")
    
    st = train_one_fold(ds_t, ds_v, f+1)
    m = SleepModel().to(CONFIG['device'])
    m.load_state_dict(st)
    m.eval()
    models.append(m)
    
    del ds_t, ds_v, st
    gc.collect()
    torch.cuda.empty_cache()

# ================= 8. INFERENCE SIMPLE (SANS VITERBI) =================
print("\n>>> INFERENCE PURE (ARGMAX)...")
ds_test = SleepDataset(test_ids, test_sig_map, mode="test")
loader_test = DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

logits_dict = {}

for model in models:
    model.to(CONFIG['device'])
    with torch.no_grad():
        for raw_epochs, _, info in tqdm(loader_test):
            raw_epochs = raw_epochs.to(CONFIG['device'])
            b, s, c, t = raw_epochs.shape
            flat = raw_epochs.view(b*s, c, t)
            specs = to_spectrogram_batch(flat)
            X = specs.view(b, s, c, CONFIG['spec_size'][0], CONFIG['spec_size'][1])
            
            with autocast():
                # Softmax pour avoir des probas propres
                out = torch.softmax(model(X), dim=1).cpu().numpy()
            
            rids, centers = info
            for i in range(len(rids)):
                k = (int(rids[i]), int(centers[i]))
                if k not in logits_dict: logits_dict[k] = []
                logits_dict[k].append(out[i])
    model.cpu()

# Construction soumission
sample_files = glob.glob(f"{CONFIG['root_dir']}/**/sample*.csv", recursive=True)
# Fallback si le sample est directement dans le dossier courant
if not sample_files: sample_files = glob.glob("sample*.csv") 
sample_path = sample_files[0] if sample_files else None

if sample_path:
    sample = pd.read_csv(sample_path)
    if "identifier" in sample.columns:
        sample["record_id"] = (sample["identifier"] // 10000).astype(int)
        sample["epoch_index"] = (sample["identifier"] % 10000).astype(int)
    else:
        s_split = sample["row_id"].str.split("_", expand=True)
        sample["record_id"] = s_split[0].astype(int)
        sample["epoch_index"] = s_split[1].astype(int)

    submission = []
    for rid, grp in tqdm(sample.groupby("record_id", sort=True)):
        grp = grp.sort_values("epoch_index")
        epochs = grp["epoch_index"].values
        
        for ep in epochs:
            k = (rid, ep)
            if k in logits_dict:
                # 1. Moyenne des probas
                avg_probs = np.mean(logits_dict[k], axis=0)
                # 2. ARGMAX (Decision locale pure)
                pred = np.argmax(avg_probs)
            else:
                pred = 0 
                
            row = grp[grp["epoch_index"] == ep]
            id_val = row["identifier"].values[0] if "identifier" in row else row["row_id"].values[0]
            submission.append((id_val, pred))

    col_id = "identifier" if "identifier" in sample.columns else "row_id"
    # Sauvegarde dans le dossier de sortie Kaggle
    pd.DataFrame(submission, columns=[col_id, "target"]).to_csv("submission.csv", index=False)
    print(">>> SUCCESS: 'submission.csv' généré dans Output.")
else:
    print("ERREUR: Sample submission introuvable. Impossible de générer le CSV.")
