from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


_import_ok = False
_errs = []

for modname in ['train']:
    try:
        m = __import__(modname)
        globals().update({k: getattr(m, k) for k in dir(m) if not k.startswith('_')})
        _import_ok = True
        break
    except Exception as e:
        _errs.append(f'{modname}: {e}')
if not _import_ok:
    raise SystemExit('Could not import training module. Tried: ' + '; '.join(_errs))

def load_split_rows(split_dir: Path, split_name: str):
    """Load rows [(filename, label_str or '')]; prefer annotations.csv if present, else unlabeled."""
    csv_p = split_dir / 'annotations.csv'
    rows = []
    if csv_p.exists():
        rows = read_csv(csv_p)
        rows = filter_rows(split_dir, rows)
    else:
        img_dir = split_dir / 'images'
        for p in img_dir.glob('*'):
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                rows.append((p.name, ''))
    return rows

def build_split_dataset(data_root: Path, split: str, img_h: int, img_w: int, alphabet: str, rgb: bool):
    """build_split_dataset.

    Args:
        data_root: 
        split: 
        img_h: 
        img_w: 
        alphabet: 
        rgb: 

    Returns:
        None"""
    split_dir = data_root / split
    if not split_dir.exists():
        raise SystemExit(f"Split folder '{split}' not found at {split_dir}")
    rows = load_split_rows(split_dir, split)
    ds = CaptchaDataset(split_dir, rows, img_h, img_w, alphabet, aug=False, rgb=rgb, use_external_aug=False)
    return ds

def batched(iterable, n):
    """
    
    Args:
        iterable: Argument.
        n: Argument.

    Returns:
        None
    """
    it = iter(iterable)
    while True:
        chunk = list()
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def main():
    """
    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--num_samples', type=int, default=64)
    ap.add_argument('--cols', type=int, default=8)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    ap.add_argument('--out_image', type=str, default='pred_grid.png')
    ap.add_argument('--out_csv', type=str, default='predictions.csv')
    args = ap.parse_args()
    dev = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device if args.device != 'auto' else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=dev)
    
    alphabet = ckpt.get('alphabet', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    img_h = int(ckpt.get('img_h', 64))
    img_w = int(ckpt.get('img_w', 256))
    rgb = bool(ckpt.get('rgb', False))
    
    rnn_hidden = int(ckpt.get('rnn_hidden', 256))
    rnn_layers = int(ckpt.get('rnn_layers', 2))
    num_classes = len(alphabet) + 1
    idx2char = {i + 1: c for i, c in enumerate(alphabet)}

    model = VGG16_LSTM_CTC(num_classes=num_classes, img_h=img_h, rgb=rgb, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers, pretrained=False).to(dev)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    data_root = Path(args.data_root)
    ds = build_split_dataset(data_root, args.split, img_h, img_w, alphabet, rgb)
    N = min(args.num_samples, len(ds))
    if N == 0:
        raise SystemExit(f"No samples found in split '{args.split}'.")
    
    idxs = random.sample(range(len(ds)), N)
    records = []
    images_to_plot = []
    
    with torch.no_grad():
        for chunk in batched(idxs, 64):
            batch_imgs = []
            gts = []
            raw_imgs = []
            fnames = []
            for i in chunk:
                sample = ds[i]
                batch_imgs.append(sample['image'])
                gts.append(sample['label_str'] if 'label_str' in sample else '')
                fname = ds.rows[i][0]
                img_p = ds.root / 'images' / fname
                
                pil = Image.open(img_p).convert('L')
                pil = ds._resize_letterbox(pil)
                raw_imgs.append(pil)
                fnames.append(fname)
            X = torch.stack(batch_imgs, dim=0).to(dev).float()
            logits = model(X)

            preds = ctc_greedy_decode(logits, idx2char)
            
            for fname, pil, gt, pred in zip(fnames, raw_imgs, gts, preds):
                images_to_plot.append((fname, pil, pred, gt))
                records.append({'filename': fname, 'pred': pred, 'gt': gt})
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    cols = max(1, args.cols)
    rows = int(np.ceil(N / cols))
    figsize = (cols * 2.6, rows * 2.2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < N:
                fname, pil, pred, gt = images_to_plot[k]
                ax.imshow(pil, cmap='gray')
                title = f'pred: {pred}'
                if gt:
                    title += f' | gt: {gt}'
                ax.set_title(title, fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')
            k += 1
    fig.tight_layout()
    
    out_img = Path(args.out_image)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=150, bbox_inches='tight')
    
    print(f'Saved image grid to: {out_img}')
    print(f'Saved predictions CSV to: {out_csv}')


if __name__ == '__main__':
    main()