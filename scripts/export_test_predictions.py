from __future__ import annotations

import argparse
from pathlib import Path
import json
import PIL
import torch
from torch.utils.data import DataLoader

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

def load_test_rows(test_dir: Path):
    
    rows = []
    img_dir = test_dir / 'images'
    if not img_dir.exists():
        raise SystemExit(f'[ERROR] Missing folder: {img_dir}')
    for p in img_dir.glob('*'):
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            rows.append((p.name, ''))
    rows.sort()
    return rows

def build_test_dataset(data_root: Path, img_h: int, img_w: int, alphabet: str, rgb: bool):
    
    test_dir = data_root / 'test'
    rows = load_test_rows(test_dir)
    ds = CaptchaDataset(test_dir, rows, img_h, img_w, alphabet, aug=False, rgb=rgb, use_external_aug=False)
    return ds

def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def main():
    """main.    """
    ap = argparse.ArgumentParser(description='Export predictions for all test images to JSON.')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_json', type=str, default='/mnt/data/test_predictions.json')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
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
    ds = build_test_dataset(data_root, img_h, img_w, alphabet, rgb)
    results = []
    
    with torch.no_grad():
        for idx_batch in batched(range(len(ds)), args.batch_size):
            batch_imgs = []
            batch_fnames = []
            batch_hw = []
            for i in idx_batch:
                sample = ds[i]
                batch_imgs.append(sample['image'])
                fname = ds.rows[i][0]
                batch_fnames.append(fname)
                from PIL import Image
                img_path = ds.root / 'images' / fname
                with Image.open(img_path) as pil:
                    w, h = pil.size
                batch_hw.append((h, w))
            X = torch.stack(batch_imgs, dim=0).to(dev).float()
            logits = model(X)
            preds = ctc_greedy_decode(logits, idx2char)
            for fname, (h, w), pred in zip(batch_fnames, batch_hw, preds):
                results.append({'height': int(h), 'width': int(w), 'image_id': Path(fname).stem, 'captcha_string': pred, 'annotations': [{'bbox': [], 'oriented_bbox': [], 'category_id': None}]})
    
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'[OK] Wrote {len(results)} entries to {out_p}')

if __name__ == '__main__':
    main()