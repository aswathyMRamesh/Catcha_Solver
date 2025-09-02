import os
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np
from PIL import Image
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_csv(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            fname, label = line.split(",", 1)
            rows.append((fname.strip(), label.strip()))
    return rows

# -----------------------------
# Dataset
# -----------------------------
class CaptchaDataset(Dataset):
    def __init__(self, root: Path, rows: List[Tuple[str, str]], img_h: int, img_w: int,
                 alphabet: str, aug: bool):
        self.root = root
        self.rows = rows
        self.img_h = img_h
        self.img_w = img_w
        self.alphabet = alphabet
        self.char2idx = {c: i+1 for i, c in enumerate(alphabet)}  # 0 = CTC blank

        if aug:
            # Slightly gentler defaults for captchas
            self.transform = T.Compose([
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.15),
                T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.35),
                T.RandomAffine(degrees=3, scale=(0.95, 1.05), shear=2, translate=(0.04, 0.04)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.rows)

    def encode_text(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx):
        fname, label = self.rows[idx]
        img_path = self.root / "images" / fname

        # Load grayscale image
        img = Image.open(img_path).convert("L")
        w, h = img.size

        # Resize keeping aspect ratio with padding
        scale = min(self.img_w / w, self.img_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = img.resize((nw, nh), Image.BILINEAR)

        canvas = Image.new("L", (self.img_w, self.img_h), 255)
        x0 = (self.img_w - nw) // 2
        y0 = (self.img_h - nh) // 2
        canvas.paste(resized, (x0, y0))

        canvas = canvas.convert("RGB")
        x = self.transform(canvas)

        target = self.encode_text(label)
        target_len = len(target)

        return {
            "image": x,
            "label": torch.tensor(target, dtype=torch.long),
            "label_str": label,
            "label_len": torch.tensor(target_len, dtype=torch.long),
            "w": torch.tensor(self.img_w),
            "h": torch.tensor(self.img_h),
        }

def ctc_collate(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = [b["label"] for b in batch]
    label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat = torch.cat(labels, dim=0) if label_lens.sum().item() > 0 else torch.empty((0,), dtype=torch.long)
    label_strs = [b["label_str"] for b in batch]
    return images, flat, label_lens, label_strs

# -----------------------------
# Model (CRNN)
# -----------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_h: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True),
        )
        # For img_h divisible by 16, this yields a small residual height; subtract 1 due to last conv
        self.feature_h = img_h // 16 - 1
        self.rnn = nn.LSTM(
            input_size=512 * self.feature_h, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feats = self.cnn(x)              # [B, C, Hp, Wp]
        b, c, hp, wp = feats.size()
        feats = feats.permute(0, 3, 1, 2).contiguous().view(b, wp, c * hp)  # [B, Wp, C*Hp]
        seq, _ = self.rnn(feats)         # [B, Wp, 512]
        out = self.fc(seq)               # [B, Wp, num_classes]
        return out.permute(1, 0, 2)      # [T=Wp, B, C]

# -----------------------------
# Decoding / Metrics
# -----------------------------
def ctc_greedy_decode(logits: torch.Tensor, idx2char: Dict[int, str]) -> List[str]:
    soft = logits.detach().cpu().argmax(-1).numpy()  # [T, B]
    Tt, B = soft.shape
    preds = []
    for b in range(B):
        last = -1
        out_chars = []
        for t in range(Tt):
            k = int(soft[t, b])
            if k != last and k != 0:
                out_chars.append(idx2char.get(k, ""))
            last = k
        preds.append("".join(out_chars))
    return preds

def cer(pred: str, gt: str) -> float:
    dp = np.zeros((len(pred)+1, len(gt)+1), dtype=np.int32)
    for i in range(len(pred)+1): dp[i, 0] = i
    for j in range(len(gt)+1): dp[0, j] = j
    for i in range(1, len(pred)+1):
        for j in range(1, len(gt)+1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)
    return dp[len(pred), len(gt)] / max(1, len(gt))

# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    steps = 0
    for images, flat_targets, target_lens, _ in loader:
        images = images.to(device).float()
        flat_targets = flat_targets.to(device)
        target_lens = target_lens.to(device)

        with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                            dtype=torch.float16, enabled=True):
            logits = model(images)
            Tt, B, C = logits.shape
            input_lens = torch.full(size=(B,), fill_value=Tt, dtype=torch.long, device=device)
            loss = criterion(logits.log_softmax(2), flat_targets, input_lens, target_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)          # <-- optimizer.step first
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1
        steps += 1
    return total_loss / max(1, n_batches), steps

@torch.no_grad()
def evaluate(model, loader, criterion, device, idx2char):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_cer, correct = [], 0
    for images, flat_targets, target_lens, label_strs in loader:
        images = images.to(device).float()
        flat_targets = flat_targets.to(device)
        target_lens = target_lens.to(device)

        logits = model(images)
        Tt, B, C = logits.shape
        input_lens = torch.full(size=(B,), fill_value=Tt, dtype=torch.long, device=device)
        loss = criterion(logits.log_softmax(2), flat_targets, input_lens, target_lens)

        preds = ctc_greedy_decode(logits, idx2char)
        for p, g in zip(preds, label_strs):
            if p == g:
                correct += 1
            all_cer.append(cer(p, g))

        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "acc": correct / len(loader.dataset) if len(loader.dataset) > 0 else 0.0,
        "cer": float(np.mean(all_cer)) if all_cer else 1.0,
    }

@torch.no_grad()
def peek_predictions(model, ds, idx2char, device, n=5):
    model.eval()
    if len(ds) == 0:
        return
    picks = random.sample(range(len(ds)), min(n, len(ds)))
    print("[PEEK] sample predictions:")
    for i in picks:
        sample = ds[i]
        img = sample["image"].unsqueeze(0).to(device).float()
        logits = model(img)
        pred = ctc_greedy_decode(logits, idx2char)[0]
        print(f"  gt='{sample['label_str']}'  pred='{pred}'")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root (parent containing train/ val/ test/)")
    parser.add_argument("--train_root", type=str, default=None,
                        help="Optional explicit train/ path (overrides data_root/train)")
    parser.add_argument("--val_root", type=str, default=None,
                        help="Optional explicit val/ path (overrides data_root/val)")
    parser.add_argument("--test_root", type=str, default=None,
                        help="Optional explicit test/ path (overrides data_root/test)")
    parser.add_argument("--annotations", type=str, default="annotations.csv")
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--alphabet", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=str(Path.home() / "captcha_solver" / "models"))
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    data_root = Path(args.data_root)
    ann_file = args.annotations

    # --- Detect dataset structure (explicit roots take precedence) ---
    if args.train_root and args.val_root:
        train_dir = Path(args.train_root)
        val_dir   = Path(args.val_root)
        test_dir  = Path(args.test_root) if args.test_root else None
        print("[INFO] Using explicit train/val/test roots from CLI")
        train_rows = read_csv(train_dir / ann_file)
        val_rows   = read_csv(val_dir / ann_file)
        test_rows  = read_csv(test_dir / ann_file) if test_dir and (test_dir / ann_file).exists() else []
        alphabet = args.alphabet or "".join(sorted({c for _, t in (train_rows + val_rows) for c in t}))
    else:
        train_dir = data_root / "train"
        val_dir   = data_root / "val"
        test_dir  = data_root / "test"
        if train_dir.exists() and val_dir.exists():
            print("[INFO] Using predefined train/val/test splits")
            train_rows = read_csv(train_dir / ann_file)
            val_rows   = read_csv(val_dir / ann_file)
            test_rows  = read_csv(test_dir / ann_file) if test_dir.exists() else []
            alphabet = args.alphabet or "".join(sorted({c for _, t in train_rows for c in t}))
        else:
            print("[INFO] Single folder dataset: random split (90/10)")
            all_rows = read_csv(data_root / ann_file)
            random.shuffle(all_rows)
            n_val = max(1, int(len(all_rows) * 0.1))
            val_rows, train_rows = all_rows[:n_val], all_rows[n_val:]
            test_rows = []
            alphabet = args.alphabet or "".join(sorted({c for _, t in all_rows for c in t}))

    print(f"[INFO] Alphabet size = {len(alphabet)} | Alphabet = {alphabet}")

    # --- Datasets ---
    train_root_for_ds = train_dir
    val_root_for_ds   = val_dir

    train_ds = CaptchaDataset(train_root_for_ds, train_rows, args.img_h, args.img_w, alphabet, aug=True)
    val_ds   = CaptchaDataset(val_root_for_ds,   val_rows,   args.img_h, args.img_w, alphabet, aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=ctc_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=ctc_collate
    )

    # --- Model / Loss / Optim ---
    num_classes = len(alphabet) + 1  # +1 for CTC blank (index 0)
    idx2char = {i+1: c for i, c in enumerate(alphabet)}

    model = CRNN(num_classes=num_classes, img_h=args.img_h).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Warmup + cosine LR over *steps*; step scheduler after optimizer to avoid warnings ---
    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, args.batch_size)))
    total_steps = max(1, steps_per_epoch * max(1, args.epochs))
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(
        "cuda" if device == "cuda" else "cpu",
        enabled=(device == "cuda")
    )
    print(f"[INFO] Starting training for {args.epochs} epochs on {device}")
    best_cer = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, steps_this_epoch = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device=device, grad_clip=1.0
        )
        # Step the scheduler *after* optimizer has stepped all batches in this epoch
        for _ in range(steps_this_epoch):
            scheduler.step()
            global_step += 1

        metrics = evaluate(model, val_loader, criterion, device, idx2char)
        dt = time.time() - t0

        # Last LR is scheduler.get_last_lr()[0]
        print(f"[E{epoch:03d}] train_loss={train_loss:.4f}  "
              f"val_loss={metrics['loss']:.4f}  acc={metrics['acc']:.4f}  "
              f"cer={metrics['cer']:.4f}  time={dt:.1f}s  lr={scheduler.get_last_lr()[0]:.6f}")

        # quick sanity check
        peek_predictions(model, val_ds, idx2char, device, n=5)

        # Save best by CER
        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            best_path = Path(out_dir) / "crnn_ctc_best.pth"
            torch.save({
                "model": model.state_dict(),
                "alphabet": alphabet,
                "img_h": args.img_h,
                "img_w": args.img_w
            }, best_path)
            print(f"[INFO] Saved best checkpoint to {best_path}")

        # Periodic checkpoints
        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = Path(out_dir) / f"crnn_ctc_epoch{epoch:03d}.pth"
            torch.save({
                "model": model.state_dict(),
                "alphabet": alphabet,
                "img_h": args.img_h,
                "img_w": args.img_w
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint to {ckpt_path}")

    print("[INFO] Training complete.")

if __name__ == "__main__":
    main()
