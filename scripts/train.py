# train_vgg16_lstm_ctc.py

import os
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Union
from collections import Counter

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from final_augmentation import build_augment  # expects (img_h, img_w, strength, rotation_mode, aug_flags)
except Exception as e:
    build_augment = None
    print("[WARN] data_augmentation3.py not found or import failed. External augmentation disabled:", e)


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
            if fname.lower() == "filename" and label.lower() == "label":
                continue  # header
            rows.append((fname.strip(), label.strip()))
    return rows

def read_yolo_labels(root: Path, labels_subdir: str, alphabet: str) -> List[Tuple[str, str]]:
    img_dir = root / "images"
    lab_dir = root / labels_subdir
    rows: List[Tuple[str, str]] = []
    exts = {".png", ".jpg", ".jpeg"}
    alphabet_len = len(alphabet)
    for img_p in img_dir.iterdir():
        if not img_p.is_file() or img_p.suffix.lower() not in exts:
            continue
        lab_p = lab_dir / f"{img_p.stem}.txt"
        if not lab_p.exists():
            continue
        try:
            items = []
            for ln in (x.strip() for x in lab_p.read_text().splitlines() if x.strip()):
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x_center = float(parts[1])  # used only for sorting
                if 0 <= cls < alphabet_len:
                    items.append((x_center, cls))
            items.sort(key=lambda t: t[0])
            label = "".join(alphabet[c] for _, c in items)
            if label:
                rows.append((img_p.name, label))
        except Exception:
            continue
    return rows

def filter_rows(root: Path, rows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    keep, dropped = [], 0
    for fn, lab in rows:
        if not lab:
            dropped += 1; continue
        p = root / "images" / fn
        if not p.exists():
            dropped += 1; continue
        keep.append((fn, lab))
    if dropped:
        print(f"[INFO] Dropped {dropped} rows (empty label / missing image).")
    return keep

def quick_check_split_from_rows(name: str, root: Path, rows: List[Tuple[str, str]], alphabet: str):
    img_dir = root / "images"
    imgs = {p.name for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    csv_files = {fn for fn, _ in rows}
    inter = imgs & csv_files
    non_alpha = [(fn, lab) for fn, lab in rows if any(ch not in alphabet for ch in lab)]
    empty = [(fn, lab) for fn, lab in rows if len(lab) == 0]
    print(f"[CHECK] {name}: images={len(imgs)}  label_rows={len(rows)}  matched={len(inter)}")
    if empty:
        print(f"[CHECK] empty labels: {len(empty)} (showing up to 3): {empty[:3]}")
    if non_alpha:
        print(f"[CHECK] labels with out-of-alphabet chars: {len(non_alpha)} (up to 3): {non_alpha[:3]}")
    cnt = Counter()
    for _, lab in rows:
        for ch in lab:
            if ch in alphabet:
                cnt[ch] += 1
    if cnt:
        total = sum(cnt.values())
        print(f"[CHECK] label histogram (top 10 of {total}): {cnt.most_common(10)}")


# -----------------------------
# Dataset
# -----------------------------
class CaptchaDataset(Dataset):
    """
    Expects:
      root/
        images/
        (either annotations.csv OR labels/<basename>.txt per image)
    """
    def __init__(self, root: Path, rows: List[Tuple[str, str]], img_h: int, img_w: int,
                 alphabet: str, aug: bool, rgb: bool,
                 use_external_aug: bool = False,
                 ext_aug_kwargs: dict = None):
        self.root = root
        self.rows = rows
        self.img_h = img_h
        self.img_w = img_w
        self.alphabet = alphabet
        self.rgb = rgb
        self.char2idx = {c: i+1 for i, c in enumerate(alphabet)}  # 0 = blank for CTC

        # External aug (PIL->PIL) — only if requested AND available
        self.ext_aug = None
        if aug and use_external_aug and build_augment is not None:
            # expected keys: strength, rotation_mode, aug_flags (optional)
            self.ext_aug = build_augment(
                img_h=img_h,
                img_w=img_w,
                strength=(ext_aug_kwargs or {}).get("strength", "medium"),
                rotation_mode=(ext_aug_kwargs or {}).get("rotation_mode", "resize"),
                aug_flags=(ext_aug_kwargs or {}).get("aug_flags", None),
            )

        # Small torchvision affine only if no external aug
        self.torch_geom = None
        if aug and self.ext_aug is None:
            self.torch_geom = T.RandomApply([T.RandomAffine(
                degrees=2, translate=(0.03, 0.03), shear=1.5, fill=255
            )], p=0.25)

        # Final tensorization/normalization
        tail = []
        if rgb:
            tail += [T.Grayscale(num_output_channels=3),
                     T.ToTensor(),
                     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        else:
            tail += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
        self.tail = T.Compose(tail)

    def __len__(self): return len(self.rows)

    def encode_text(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def _resize_letterbox(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.img_w / w, self.img_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (self.img_w, self.img_h), 255)  # white background
        x0 = (self.img_w - nw) // 2
        y0 = (self.img_h - nh) // 2
        canvas.paste(resized, (x0, y0))
        return canvas

    def __getitem__(self, idx):
        fname, label = self.rows[idx]
        img_path = self.root / "images" / fname

        base = Image.open(img_path).convert("L")
        base = self._resize_letterbox(base)

        if self.torch_geom is not None:
            base = self.torch_geom(base)

        if self.ext_aug is not None:
            base = self.ext_aug(base)

        x = self.tail(base)

        target = torch.tensor(self.encode_text(label), dtype=torch.long)
        return {
            "image": x,
            "label": target,
            "label_str": label,
            "label_len": torch.tensor(len(target), dtype=torch.long),
        }

def ctc_collate(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = [b["label"] for b in batch]
    label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat = torch.cat(labels, dim=0) if label_lens.sum().item() > 0 else torch.empty((0,), dtype=torch.long)
    label_strs = [b["label_str"] for b in batch]
    return images, flat, label_lens, label_strs


# -----------------------------
# Model: VGG16 (features) + BiLSTM + CTC
# -----------------------------
class VGG16_LSTM_CTC(nn.Module):
    def __init__(self, num_classes: int, img_h: int, rgb: bool, rnn_hidden: int = 256,
                 rnn_layers: int = 2, rnn_dropout: float = 0.1, blank_bias: float = -3.0,
                 pretrained: bool = False):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_FEATURES if pretrained else None)

        if not rgb:
            first_conv = vgg.features[0]
            new_conv = nn.Conv2d(1, first_conv.out_channels,
                                 kernel_size=first_conv.kernel_size,
                                 stride=first_conv.stride,
                                 padding=first_conv.padding,
                                 bias=first_conv.bias is not None)
            nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)
            vgg.features[0] = new_conv

        self.cnn = vgg.features
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=rnn_hidden, num_layers=rnn_layers,
            batch_first=True, bidirectional=True,
            dropout=(float(rnn_dropout) if rnn_layers > 1 else 0.0)
        )
        self.fc = nn.Linear(2 * rnn_hidden, num_classes)
        with torch.no_grad():
            if self.fc.bias is not None:
                self.fc.bias.zero_()
                self.fc.bias[0] = float(blank_bias)  # bias towards blank early on

    def forward(self, x):
        feats = self.cnn(x)               # [B, 512, H', W']
        feats = self.vertical_pool(feats) # [B, 512, 1, W']
        feats = feats.squeeze(2)          # [B, 512, W']
        feats = feats.permute(0, 2, 1)    # [B, W', 512]
        seq, _ = self.rnn(feats)          # [B, W', 2*hidden]
        out = self.fc(seq)                # [B, W', C]
        return out.permute(1, 0, 2)       # [T=W', B, C]  <-- time-first for CTC


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
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0, debug_once: dict = None):
    model.train()
    total_loss = 0.0; n_batches = 0; steps = 0
    for images, flat_targets, target_lens, _ in loader:
        images = images.to(device).float()
        flat_targets = flat_targets.to(device)
        target_lens = target_lens.to(device)

        if debug_once is not None and not debug_once.get("done", False):
            print("[DEBUG] batch shapes:",
                  "images", tuple(images.shape),
                  "targets_len", int(flat_targets.numel()),
                  "sum(target_lengths)", int(target_lens.sum().item()))
            assert flat_targets.ndim == 1
            assert flat_targets.numel() == int(target_lens.sum().item())
            debug_once["done"] = True

        with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                            dtype=torch.float16, enabled=True):
            logits = model(images)  # [T,B,C]
        Tt, B, C = logits.shape
        input_lens = torch.full((B,), Tt, dtype=torch.long, device=device)
        loss = criterion(logits.float().log_softmax(2), flat_targets, input_lens, target_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item(); n_batches += 1; steps += 1
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

        logits = model(images)  # [T,B,C]
        Tt, B, C = logits.shape
        input_lens = torch.full((B,), Tt, dtype=torch.long, device=device)
        loss = criterion(logits.float().log_softmax(2), flat_targets, input_lens, target_lens)

        preds = ctc_greedy_decode(logits, idx2char)
        for p, g in zip(preds, label_strs):
            if p == g: correct += 1
            all_cer.append(cer(p, g))

        total_loss += loss.item(); n_batches += 1
    return {
        "loss": total_loss / max(1, n_batches),
        "acc": correct / len(loader.dataset) if len(loader.dataset) > 0 else 0.0,
        "cer": float(np.mean(all_cer)) if all_cer else 1.0,
    }

@torch.no_grad()
def peek_predictions(model, ds, idx2char, device, n=5):
    model.eval()
    if len(ds) == 0: return
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
    # ---- core training args ----
    parser.add_argument("--data_root", type=str, required=True, help="Parent folder containing train/ val/ test/")
    parser.add_argument("--annotations", type=str, default="annotations.csv", help="CSV file (if --label_mode csv)")
    parser.add_argument("--label_mode", choices=["csv", "yolo"], default="csv",
                        help="Use 'yolo' to read labels/<img>.txt files; 'csv' uses annotations.csv")
    parser.add_argument("--labels_subdir", type=str, default="labels", help="Subfolder name for YOLO txts")
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=256, help="Use 256 to get ~8 time steps after VGG")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alphabet", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=str(Path.home() / "captcha_solver" / "models"))
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--no_aug", action="store_true", help="Disable augmentation")
    parser.add_argument("--rgb", action="store_true", help="Use RGB (3ch) input; default is 1ch")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet VGG16_bn weights (needs internet)")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="none", help="Cosine or constant LR")
    parser.add_argument("--blank_bias", type=float, default=-3.0)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--balance", action="store_true", help="Enable inverse-freq WeightedRandomSampler")

    # ---- external augmentation toggles ----
    parser.add_argument("--use_external_aug", action="store_true",
                        help="Enable custom augmentations from data_augmentation5.py")

    # overall strength + rotation behavior (add these ONCE)
    parser.add_argument("--aug_strength", choices=["mild", "medium", "strong"], default="mild",
                        help="Set the overall augmentation strength level")
    parser.add_argument("--rotation_mode", choices=["resize", "crop"], default="resize",
                        help="How to handle rotation (resize keeps full image, crop trims borders)")

    # optional per-op intensity knobs (only if your build_augment supports them)
    parser.add_argument("--aug_p_geom", type=float, default=0.15, help="Prob. to apply geometric jitter")
    parser.add_argument("--aug_p_noise", type=float, default=0.10, help="Prob. to apply noise/clutter stack")
    parser.add_argument("--aug_gauss_min", type=float, default=1.0, help="Gaussian noise std min (pixel scale)")
    parser.add_argument("--aug_gauss_max", type=float, default=2.0, help="Gaussian noise std max (pixel scale)")

    # ---- sanity/limits ----
    parser.add_argument("--limit_train", type=int, default=0, help="If >0, limit train rows to this many")
    parser.add_argument("--limit_val", type=int, default=0, help="If >0, limit val rows")
    parser.add_argument("--dry_run", action="store_true", help="Build one batch & exit (sanity check)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir   = data_root / "val"

    if not (train_dir.exists() and val_dir.exists()):
        raise SystemExit("[ERROR] train/ and val/ folders are required.")

    print("[INFO] Using predefined train/val/test splits")

    alphabet = args.alphabet

    if args.label_mode == "csv":
        train_rows = read_csv(train_dir / args.annotations)
        val_rows   = read_csv(val_dir / args.annotations)
    else:
        train_rows = read_yolo_labels(train_dir, args.labels_subdir, alphabet)
        val_rows   = read_yolo_labels(val_dir,   args.labels_subdir, alphabet)

    quick_check_split_from_rows("train", train_dir, train_rows, alphabet)
    quick_check_split_from_rows("val",   val_dir,   val_rows,   alphabet)

    train_rows = filter_rows(train_dir, train_rows)
    val_rows   = filter_rows(val_dir,   val_rows)

    if args.limit_train > 0:
        train_rows = train_rows[:args.limit_train]
        print(f"[INFO] limit_train active: {len(train_rows)} samples")
    if args.limit_val > 0:
        val_rows = val_rows[:args.limit_val]
        print(f"[INFO] limit_val active: {len(val_rows)} samples")

    # Datasets — external aug for train only
    use_aug = not args.no_aug
    ext_kwargs = dict(
        p_geom=args.aug_p_geom,
        p_noise=args.aug_p_noise,
        gauss_std_px=(args.aug_gauss_min, args.aug_gauss_max),
        strength=args.aug_strength,           
        rotation_mode=args.rotation_mode,     
    )
    train_ds = CaptchaDataset(
        train_dir, train_rows, args.img_h, args.img_w, alphabet,
        aug=use_aug, rgb=args.rgb,
        use_external_aug=(args.use_external_aug and build_augment is not None),
        ext_aug_kwargs=ext_kwargs
    )
    val_ds   = CaptchaDataset(
        val_dir, val_rows, args.img_h, args.img_w, alphabet,
        aug=False, rgb=args.rgb,
        use_external_aug=False
    )

    # Dataloaders
    dl_kwargs = dict(collate_fn=ctc_collate, pin_memory=(device == "cuda"))
    if args.num_workers and args.num_workers > 0:
        dl_kwargs.update(dict(num_workers=args.num_workers, persistent_workers=True, prefetch_factor=2))
    else:
        dl_kwargs.update(dict(num_workers=0))

    if args.balance and len(train_rows) > 0:
        char_freq = Counter()
        for _, lab in train_rows:
            for ch in lab: char_freq[ch] += 1
        inv = {ch: 1.0 / max(1, c) for ch, c in char_freq.items()}
        sample_weights = []
        for _, lab in train_rows:
            w = (sum(inv.get(ch, 0.0) for ch in lab) / max(1, len(lab))) if lab else 1e-6
            sample_weights.append(max(w, 1e-6))
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    # Model / loss / optim
    num_classes = len(alphabet) + 1
    idx2char = {i+1: c for i, c in enumerate(alphabet)}
    model = VGG16_LSTM_CTC(
        num_classes=num_classes, img_h=args.img_h, rgb=args.rgb,
        rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, rnn_dropout=args.rnn_dropout,
        blank_bias=args.blank_bias, pretrained=args.pretrained
    ).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Sanity on time-steps
    with torch.no_grad():
        dummy = torch.zeros(1, 3 if args.rgb else 1, args.img_h, args.img_w, device=device)
        Tt = model(dummy).shape[0]
    max_lab_len = max((len(t) for _, t in train_rows), default=0)
    print(f"[INFO] time_steps={Tt}  max_label_len={max_lab_len}")
    if Tt < max_lab_len:
        print(f"[WARN] time_steps ({Tt}) < max_label_len ({max_lab_len}). "
              f"Consider increasing --img_w or reducing label length/alphabet.")

    # Dry run
    if args.dry_run:
        print("[INFO] --dry_run: grabbing one batch to sanity-check shapes…")
        it = iter(train_loader)
        images, flat_targets, target_lens, label_strs = next(it)
        print("[DRY] images", tuple(images.shape),
              "len(targets)", int(flat_targets.numel()),
              "sum(target_lens)", int(target_lens.sum().item()),
              "example_labels", label_strs[:5])
        return

    # Scheduler
    if args.scheduler == "cosine":
        steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, args.batch_size)))
        total_steps = max(1, steps_per_epoch * max(1, args.epochs))
        warmup_steps = max(1, int(0.1 * total_steps))
        def lr_lambda(step):
            if step < warmup_steps: return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda" if device == "cuda" else "cpu", enabled=(device == "cuda"))

    print(f"[INFO] Starting training for {args.epochs} epochs on {device}")
    best_cer = float("inf"); debug_once = {"done": False}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, steps_this_epoch = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, grad_clip=1.0, debug_once=debug_once
        )
        if scheduler is not None:
            for _ in range(steps_this_epoch): scheduler.step()

        metrics = evaluate(model, val_loader, criterion, device, idx2char)
        dt = time.time() - t0

        lr_print = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        print(f"[E{epoch:03d}] train_loss={train_loss:.4f}  val_loss={metrics['loss']:.4f}  "
              f"acc={metrics['acc']:.4f}  cer={metrics['cer']:.4f}  time={dt:.1f}s  lr={lr_print:.6f}")
        peek_predictions(model, val_ds, idx2char, device, n=5)

        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            best_path = Path(args.out_dir) / "vgg16_lstm_ctc_best.pth"
            torch.save({
                "model": model.state_dict(),
                "alphabet": alphabet,
                "img_h": args.img_h,
                "img_w": args.img_w,
                "rgb": args.rgb,
                "rnn_hidden": args.rnn_hidden,
                "rnn_layers": args.rnn_layers,
            }, best_path)
            print(f"[INFO] Saved best checkpoint to {best_path}")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = Path(args.out_dir) / f"vgg16_lstm_ctc_epoch{epoch:03d}.pth"
            torch.save({
                "model": model.state_dict(),
                "alphabet": alphabet,
                "img_h": args.img_h,
                "img_w": args.img_w,
                "rgb": args.rgb,
                "rnn_hidden": args.rnn_hidden,
                "rnn_layers": args.rnn_layers,
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint to {ckpt_path}")

    print("[INFO] Training complete.")

if __name__ == "__main__":
    main()
