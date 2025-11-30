# Training_torch.py
import os
import glob
import json
import time
import math
import numpy as np
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import csv
from datetime import datetime

def set_seed(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARNING] {msg}")

# helper for plot
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def make_run_dir(tag: str, plot_root: str = "./plot_data") -> str:
    run_dir = os.path.join(plot_root, f"{tag}_{_now_tag()}")
    _ensure_dir(run_dir)
    return run_dir

def csv_logger(run_dir: str):
    """Create a CSV writer for epoch metrics."""
    _ensure_dir(run_dir)
    path = os.path.join(run_dir, "metrics.csv")
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr","secs"])
    return f, writer, path

def save_test_result(run_dir: str, loss: float, acc: float, meta: dict = None):
    meta = {} if meta is None else dict(meta)
    meta.update({"test_loss": float(loss), "test_acc": float(acc)})
    with open(os.path.join(run_dir, "test.json"), "w", encoding="utf-8") as g:
        json.dump(meta, g, ensure_ascii=False, indent=2)

# Dataset
class SkeletonNPZDataset(Dataset):
    """
    Load .npz files
    X: (T,H,21,3) float32, mask: (T,H,21) bool
    mode:
      - "mlp":  (D,) flattened
      - "rnn":  (T,F)
      - "cnn":  (C,T)
    """
    def __init__(self, files: List[str], label2id: Dict[str, int], mode: str = "mlp"):
        self.files = files
        self.label2id = label2id
        self.mode = mode
        assert mode in {"mlp", "rnn", "cnn"}

    @staticmethod
    def _infer_label_from_path(path: str) -> str:
        name = os.path.splitext(os.path.basename(path))[0]
        return name.split("_", 1)[0] if "_" in name else name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        dat = np.load(path, allow_pickle=False)
        X = dat["X"].astype(np.float32)      # (T,H,21,3)

        T, H, J, C = X.shape
        F = H * J * C
        if self.mode == "mlp":
            feat = X.reshape(-1)                            # (T*H*21*3,)
            feat = torch.from_numpy(feat)
        elif self.mode == "rnn":
            seq = X.reshape(T, F)                           # (T,F)
            feat = torch.from_numpy(seq)
        else:  # cnn
            seq = X.reshape(T, F).transpose(1, 0)           # (F,T)
            feat = torch.from_numpy(seq)

        label_name = self._infer_label_from_path(path)
        y = self.label2id[label_name]
        return feat, torch.tensor(y, dtype=torch.long)

def collect_files(root="./npz_fixed") -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz found in {root}")
    return files

def build_label_map(files: List[str]) -> Dict[str, int]:
    labels = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        labels.append(name.split("_", 1)[0] if "_" in name else name)
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}

# =============== Robust 7:2:1 split ===============
def split_7_2_1(files: List[str], label2id: Dict[str, int], seed: int = 42):
    """
    Robust 7:2:1 split with rare-class handling.
    """
    from collections import defaultdict
    from sklearn.model_selection import train_test_split

    per_cls = defaultdict(list)
    for i, p in enumerate(files):
        name = os.path.splitext(os.path.basename(p))[0]
        cls = name.split("_", 1)[0] if "_" in name else name
        per_cls[label2id[cls]].append(i)

    singletons = []   # classes with count==1
    rest_indices = []
    for cls_id, idxs in per_cls.items():
        if len(idxs) == 1:
            singletons.extend(idxs)  # force to train
        else:
            rest_indices.extend(idxs)

    singletons = np.array(singletons, dtype=int)
    rest_indices = np.array(rest_indices, dtype=int)

    # If nothing to stratify, just random split the rest
    def naive_split(idx_all: np.ndarray):
        if len(idx_all) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
        # 10% test
        idx_tmp, idx_test = train_test_split(
            idx_all, test_size=0.1, random_state=seed, shuffle=True, stratify=None
        )
        # 20% of remaining for val
        idx_train, idx_val = train_test_split(
            idx_tmp, test_size=(0.2/0.9), random_state=seed, shuffle=True, stratify=None
        )
        return idx_train, idx_val, idx_test

    if len(rest_indices) == 0:
        # Only singletons exist
        warn("All classes are singletons; putting everything into TRAIN by design.")
        return singletons, np.array([], dtype=int), np.array([], dtype=int)

    # Prepare y for rest_indices
    y_rest = []
    for i in rest_indices:
        name = os.path.splitext(os.path.basename(files[i]))[0]
        cls = name.split("_", 1)[0] if "_" in name else name
        y_rest.append(label2id[cls])
    y_rest = np.array(y_rest)

    # Try stratified split on the 'rest'
    try:
        idx_tmp, idx_test = train_test_split(
            rest_indices, test_size=0.1, random_state=seed, shuffle=True, stratify=y_rest
        )
        y_tmp = []
        for i in idx_tmp:
            name = os.path.splitext(os.path.basename(files[i]))[0]
            cls = name.split("_", 1)[0] if "_" in name else name
            y_tmp.append(label2id[cls])
        y_tmp = np.array(y_tmp)

        idx_train, idx_val = train_test_split(
            idx_tmp, test_size=(0.2/0.9), random_state=seed, shuffle=True, stratify=y_tmp
        )
    except ValueError as e:
        warn(f"Stratified split failed on remaining classes ({e}). Falling back to non-stratified for them.")
        idx_train, idx_val, idx_test = naive_split(rest_indices)

    # Merge singletons into train and shuffle
    idx_train = np.concatenate([idx_train, singletons]).astype(int)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx_train)

    return idx_train, idx_val.astype(int), idx_test.astype(int)

def make_loaders(files: List[str], idx_train, idx_val, idx_test, label2id, mode="mlp",
                 batch_size=256, num_workers=4, pin_memory=True):
    ds_tr = SkeletonNPZDataset([files[i] for i in idx_train], label2id, mode=mode)
    ds_va = SkeletonNPZDataset([files[i] for i in idx_val],   label2id, mode=mode)
    ds_te = SkeletonNPZDataset([files[i] for i in idx_test],  label2id, mode=mode)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=(num_workers>0))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=(num_workers>0))
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=(num_workers>0))
    return dl_tr, dl_va, dl_te

# Models
class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):  # (B,D)
        return self.net(x)

class CNN1D(nn.Module):
    """ Input: (B, C, T) where C=H*21*3 """
    def __init__(self, in_channels: int, n_classes: int, T: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, n_classes)
    def forward(self, x):  # (B,C,T)
        z = self.feat(x).squeeze(-1)  # (B,256)
        return self.head(z)

class CNN2D(nn.Module):
    """
    2D CNN over (T x F) skeleton image.
    Accepts (B,T,F) or (B,1,T,F); always returns logits (B,num_classes).
    """
    def __init__(self, in_feat: int, n_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B,T,F) or (B,1,T,F)
        if x.dim() == 3:
            x = x.unsqueeze(1)         # -> (B,1,T,F)
        elif x.dim() == 4:
            if x.size(1) != 1:         # 保证单通道
                x = x[:, :1, ...]
        else:
            raise RuntimeError(f"CNN2D expects (B,T,F) or (B,1,T,F), got {tuple(x.shape)}")

        z = self.backbone(x).flatten(1)  # (B,128)
        logits = self.head(z)            # (B,C)
        return logits

class GRUClassifier(nn.Module):
    """ Input: (B, T, F) where F=H*21*3 """
    def __init__(self, input_size: int, n_classes: int, hidden: int = 256, layers: int = 2, bidir: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=0.1 if layers>1 else 0.0, bidirectional=bidir
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Linear(out_dim, n_classes)
    def forward(self, x):  # (B,T,F)
        y, _ = self.gru(x)
        last = y[:, -1, :]
        return self.head(last)


class SinPositionalEncoding(nn.Module):
    """Standard sinusoidal PE: (B,T,D) -> (B,T,D)"""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T,D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)  # (1,T,D) + (B,T,D)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder over time: input (B,T,F) -> proj -> PE -> Encoder -> pooled -> logits
    """
    def __init__(self, in_feat: int, n_classes: int,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_feat, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,  # (B,T,D)
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = SinPositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        if x.dim() != 3:
            raise RuntimeError(f"TemporalTransformer expects (B,T,F), got {tuple(x.shape)}")
        z = self.proj(x)              # (B,T,D)
        z = self.posenc(z)            # (B,T,D)
        z = self.encoder(z)           # (B,T,D)
        z = self.norm(z)              # (B,T,D)
        z = z.mean(dim=1)             # global average over time -> (B,D)
        logits = self.head(z)         # (B,C)
        return logits

# =============== Train / Eval (CUDA+AMP) ===============
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        pred = logits.argmax(dim=1)
        total_loss += loss.item() * xb.size(0)
        total_acc  += (pred == yb).sum().item()
        total_n    += xb.size(0)
    return total_loss / max(1,total_n), total_acc / max(1,total_n)

def train_torch(model, train_loader, val_loader, epochs=20, lr=1e-3, weight_decay=1e-2,
                grad_clip=1.0, early_stop=5, use_amp=True, ckpt_dir="./ckpt", plot_run_dir: str = None):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device_auto()
    log(f"Device: {device}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    criterion = nn.CrossEntropyLoss()

    # ---- CSV logger ----
    csv_f, csv_w, csv_path = (None, None, None)
    if plot_run_dir is not None:
        csv_f, csv_w, csv_path = csv_logger(plot_run_dir)
        log(f"[Plot] Logging metrics to: {csv_path}")

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pt")
    no_imp = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * xb.size(0)
            run_correct += (logits.argmax(1) == yb).sum().item()
            run_n += xb.size(0)

        train_loss = run_loss / max(1, run_n)
        train_acc = run_correct / max(1, run_n)
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        # 学习率（OneCycleLR 的当前 lr 在 param_groups[0]["lr"]）
        cur_lr = optimizer.param_groups[0]["lr"]

        log(f"Epoch {ep:03d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | {dt:.1f}s")

        # ---- write a CSV row ----
        if csv_w is not None:
            csv_w.writerow([ep, f"{train_loss:.6f}", f"{train_acc:.6f}",
                            f"{val_loss:.6f}", f"{val_acc:.6f}",
                            f"{cur_lr:.8f}", f"{dt:.3f}"])
            csv_f.flush()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_imp = 0
            torch.save({"model": model.state_dict()}, best_path)
            log(f"  -> New best (val_loss={best_val:.4f}) saved to {best_path}")
        else:
            no_imp += 1
            if no_imp >= early_stop:
                log("Early stopping triggered.")
                break

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log("Loaded best checkpoint.")

    if csv_f is not None:
        csv_f.close()
    return model


def main():
    set_seed(42)
    root = "./npz_fixed"
    files = collect_files(root)
    label2id = build_label_map(files)
    n_classes = len(label2id)
    log(f"Found {len(files)} files, #classes={n_classes}")

    # robust split
    idx_tr, idx_va, idx_te = split_7_2_1(files, label2id, seed=42)
    log(f"Split sizes -> train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    # MLP
    dl_tr, dl_va, dl_te = make_loaders(files, idx_tr, idx_va, idx_te, label2id,
                                       mode="mlp", batch_size=512, num_workers=4)
    xb0, y0 = next(iter(dl_tr))
    D = xb0.shape[1]
    log(f"[MLP] Input dim D={D}")
    run_dir_mlp = make_run_dir("MLP")
    mlp = MLP(in_dim=D, n_classes=n_classes, hidden=512)
    mlp = train_torch(mlp, dl_tr, dl_va, epochs=20, lr=1e-3, ckpt_dir="./ckpt_mlp", plot_run_dir=run_dir_mlp)
    te_loss, te_acc = evaluate(mlp, dl_te, device_auto())
    log(f"[MLP] Test loss={te_loss:.4f}, acc={te_acc:.4f}")

    # 1D-CNN
    # dl_tr_cnn, dl_va_cnn, dl_te_cnn = make_loaders(files, idx_tr, idx_va, idx_te, label2id,
    #                                                mode="cnn", batch_size=64, num_workers=4)
    # xb1, _ = next(iter(dl_tr_cnn))         # (B,C,T)
    # C, T = xb1.shape[1], xb1.shape[2]
    # log(f"[CNN1D] Channels={C}, T={T}")
    # run_dir_cnn1 = make_run_dir("CNN")
    # cnn = CNN1D(in_channels=C, n_classes=n_classes, T=T)
    # cnn = train_torch(cnn, dl_tr_cnn, dl_va_cnn, epochs=100, lr=8e-4, ckpt_dir="./ckpt_cnn", plot_run_dir=run_dir_cnn1)
    # te_loss, te_acc = evaluate(cnn, dl_te_cnn, device_auto())
    # log(f"[CNN1D] Test loss={te_loss:.4f}, acc={te_acc:.4f}")

    # RNN (GRU)
    # dl_tr_rnn, dl_va_rnn, dl_te_rnn = make_loaders(files, idx_tr, idx_va, idx_te, label2id,
    #                                                mode="rnn", batch_size=128, num_workers=4)
    # xb2, _ = next(iter(dl_tr_rnn))         # (B,T,F)
    # F = xb2.shape[2]
    # log(f"[GRU] F={F}")
    # run_dir_gru = make_run_dir("GRU")
    # rnn = GRUClassifier(input_size=F, n_classes=n_classes, hidden=256, layers=2, bidir=False)
    # rnn = train_torch(rnn, dl_tr_rnn, dl_va_rnn, epochs=100, lr=1e-3, ckpt_dir="./ckpt_rnn", plot_run_dir=run_dir_gru)
    # te_loss, te_acc = evaluate(rnn, dl_te_rnn, device_auto())
    # log(f"[GRU] Test loss={te_loss:.4f}, acc={te_acc:.4f}")

    # 2D-CNN
    # Reuse "rnn" mode loaders to get tensors shaped (B, T, F)
    # dl_tr_2d, dl_va_2d, dl_te_2d = make_loaders(
    #     files, idx_tr, idx_va, idx_te, label2id,
    #     mode="rnn", batch_size=64, num_workers=4
    # )
    # xb2d, _ = next(iter(dl_tr_2d))
    # T2D, F2D = xb2d.shape[1], xb2d.shape[2]
    # log(f"[CNN2D] Input as image: T={T2D}, F={F2D}")
    # run_dir_cnn = make_run_dir("CNN2D")
    #
    # cnn2d = CNN2D(in_feat=F2D, n_classes=n_classes)
    # cnn2d = train_torch(cnn2d, dl_tr_2d, dl_va_2d, epochs=100, lr=8e-4, ckpt_dir="./ckpt_cnn2d", plot_run_dir=run_dir_cnn)
    # te_loss, te_acc = evaluate(cnn2d, dl_te_2d, device_auto())
    # log(f"[CNN2D] Test loss={te_loss:.4f}, acc={te_acc:.4f}")

    # Transformer
    # Reuse "rnn" mode loaders to get (B,T,F) tensors
    # dl_tr_tx, dl_va_tx, dl_te_tx = make_loaders(
    #     files, idx_tr, idx_va, idx_te, label2id,
    #     mode="rnn", batch_size=64, num_workers=4
    # )
    # xb_tx, _ = next(iter(dl_tr_tx))
    # T_tx, F_tx = xb_tx.shape[1], xb_tx.shape[2]
    # log(f"[Transformer] Using (B,T,F) with T={T_tx}, F={F_tx}")
    # run_dir_tx = make_run_dir("Transformer")
    # tx = TemporalTransformer(in_feat=F_tx, n_classes=n_classes,
    #                          d_model=256, nhead=4, num_layers=2,
    #                          dim_feedforward=512, dropout=0.1)
    # tx = train_torch(tx, dl_tr_tx, dl_va_tx, epochs=30, lr=1e-3, ckpt_dir="./ckpt_tx", plot_run_dir=run_dir_tx)
    # te_loss, te_acc = evaluate(tx, dl_te_tx, device_auto())
    # log(f"[Transformer] Test loss={te_loss:.4f}, acc={te_acc:.4f}")

if __name__ == "__main__":
    main()
