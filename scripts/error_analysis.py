
from __future__ import annotations

from pathlib import Path
from collections import Counter

import argparse
import json
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None

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

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def levenshtein_backtrace(a: str, b: str):
    """Return (ops, distance) where ops is list of ('eq'|'sub'|'ins'|'del', a_char, b_char)."""
    n, m = (len(a), len(b))
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.empty((n + 1, m + 1), dtype=object)
    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = ('del', a[i - 1], None)
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = ('ins', None, b[j - 1])
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            if ai == bj:
                dp[i, j] = dp[i - 1, j - 1]
                bt[i, j] = ('eq', ai, bj)
            else:
                sub = dp[i - 1, j - 1] + 1
                ins = dp[i, j - 1] + 1
                dele = dp[i - 1, j] + 1
                best = min(sub, ins, dele)
                dp[i, j] = best
                if best == sub:
                    bt[i, j] = ('sub', ai, bj)
                elif best == ins:
                    bt[i, j] = ('ins', None, bj)
                else:
                    bt[i, j] = ('del', ai, None)
    ops = []
    i, j = (n, m)
    while i > 0 or j > 0:
        op = bt[i, j]
        ops.append(op)
        t = op[0]
        if t in ('eq', 'sub'):
            i -= 1
            j -= 1
        elif t == 'ins':
            j -= 1
        elif t == 'del':
            i -= 1
    ops.reverse()
    return (ops, int(dp[n, m]))

def build_loader_for_split(data_root: Path,
                            split: str,
                            img_h: int,
                            img_w: int,
                            alphabet: str,
                            rgb: bool,
                            batch_size: int=128,
                            num_workers: int=0):
    """
    build_loader_for_split.
    Args:
        data_root: Root folder with train/ val/ test/ subfolders
        split: 'train', 'val' or 'test'
        img_h: Image height
        img_w: Image width
        alphabet: Alphabet string
        rgb: Whether images are RGB (3-channel) or grayscale (1-channel)
        batch_size: Batch size
        num_workers: Number of DataLoader workers
    """
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder '{split}' not found at {split_dir}")
    rows = []
    csv_p = split_dir / 'annotations.csv'
    if csv_p.exists():
        rows = read_csv(csv_p)
        rows = filter_rows(split_dir, rows)
    else:
        img_dir = split_dir / 'images'
        for p in img_dir.glob('*'):
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                rows.append((p.name, ''))
    ds = CaptchaDataset(split_dir, rows, img_h, img_w, alphabet, aug=False, rgb=rgb, use_external_aug=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate, num_workers=num_workers if num_workers > 0 else 0, pin_memory=torch.cuda.is_available())
    return (dl, ds)

def run_eval(data_root: str, ckpt_path: str, batch_size: int=128, num_workers: int=2, device: str='auto', split: str='val', out_prefix: str='/mnt/data/', print_reports: bool=True, top_k: int=15, save_outputs: bool=True):
    """
    Evaluate on a split (val/test). Returns dict of overall metrics and paths to outputs.
    Also writes:
      - per_sample_metrics.csv
      - per_char_error.csv
      - confusion_matrix_raw.csv
      - confusion_matrix_row_normalized.csv
      - substitutions_all.csv (truth, pred, count, truth_support, row_frac)
      - insertions_all.csv (char, count)
      - deletions_all.csv (char, count)
      - error_analysis_summary.json
    """
    dev = 'cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=dev)
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
    data_root = Path(data_root)
    loader, ds = build_loader_for_split(data_root, split, img_h, img_w, alphabet, rgb, batch_size=batch_size, num_workers=num_workers)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss, n_batches = (0.0, 0)
    correct = 0
    cer_values = []
    characters = list(alphabet)
    V = len(characters)
    char_to_idx = {c: i for i, c in enumerate(characters)}
    conf_raw = np.zeros((V, V), dtype=np.int64)
    sub_counter = Counter()
    ins_counter = Counter()
    del_counter = Counter()
    truth_support = Counter()
    correct_support = Counter()
    per_sample_rows = []
    with torch.no_grad():
        for batch in loader:
            images, flat_targets, target_lens, label_strs = batch
            images = images.to(dev).float()
            logits = model(images)
            Tt, B, C = logits.shape
            input_lens = torch.full((B,), Tt, dtype=torch.long, device=dev)
            flat_targets = flat_targets.to(dev)
            target_lens = target_lens.to(dev)
            loss = criterion(logits.float().log_softmax(2), flat_targets, input_lens, target_lens)
            total_loss += float(loss.item())
            n_batches += 1
            preds = ctc_greedy_decode(logits, idx2char)
            for p, g in zip(preds, label_strs):
                if p == g:
                    correct += 1
                _, dist = levenshtein_backtrace(p, g)
                cer_val = dist / max(1, len(g))
                cer_values.append(cer_val)
                subs = ins = dels = 0
                for ch in g:
                    truth_support[ch] += 1
                ops, _ = levenshtein_backtrace(g, p)
                for op, a_ch, b_ch in ops:
                    if op == 'eq':
                        correct_support[a_ch] += 1
                    elif op == 'sub':
                        subs += 1
                        sub_counter[a_ch, b_ch] += 1
                        if a_ch in char_to_idx and b_ch in char_to_idx:
                            conf_raw[char_to_idx[a_ch], char_to_idx[b_ch]] += 1
                    elif op == 'ins':
                        ins += 1
                        if b_ch is not None:
                            ins_counter[b_ch] += 1
                    elif op == 'del':
                        dels += 1
                        if a_ch is not None:
                            del_counter[a_ch] += 1
                per_sample_rows.append({'gt': g, 'pred': p, 'cer': cer_val, 'subs': subs, 'ins': ins, 'dels': dels})
    overall = {'split': split, 'val_loss': total_loss / max(1, n_batches), 'seq_accuracy': correct / max(1, len(ds)), 'cer_mean': float(np.mean(cer_values)) if cer_values else 1.0, 'num_samples': len(ds)}
    rows = []
    for ch in characters:
        subs_from_ch = sum((cnt for (t, p), cnt in sub_counter.items() if t == ch))
        dels_of_ch = del_counter[ch]
        total_err = subs_from_ch + dels_of_ch
        rows.append({'char': ch, 'errors': total_err, 'support': truth_support[ch], 'error_rate': safe_div(total_err, truth_support[ch]), 'subs_from_char': subs_from_ch, 'deletions_of_char': dels_of_ch, 'correct': correct_support[ch]})
    per_char_df = pd.DataFrame(rows).sort_values('error_rate', ascending=False, ignore_index=True)
    conf_df_raw = pd.DataFrame(conf_raw, index=characters, columns=characters)
    row_sums = conf_raw.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf_raw, np.maximum(row_sums, 1), where=row_sums > 0)
    conf_df_norm = pd.DataFrame(conf_norm, index=characters, columns=characters)
    pairs = []
    for i, t_ch in enumerate(characters):
        for j, p_ch in enumerate(characters):
            if i == j:
                continue
            if row_sums[i, 0] > 0 and conf_raw[i, j] > 0:
                pairs.append({'truth': t_ch, 'pred': p_ch, 'frac': float(conf_norm[i, j]), 'count': int(conf_raw[i, j]), 'row_total': int(row_sums[i, 0])})
    worst_pairs = sorted(pairs, key=lambda x: x['frac'], reverse=True)
    subs_rows = []
    for (t, p), cnt in sub_counter.items():
        den = truth_support[t]
        frac = cnt / den if den else 0.0
        subs_rows.append({'truth': t, 'pred': p, 'count': cnt, 'truth_support': den, 'row_frac': frac})
    subs_df_all = pd.DataFrame(subs_rows)
    subs_df_norm = subs_df_all.sort_values(['row_frac', 'count'], ascending=[False, False]).head(top_k)
    subs_df_raw = subs_df_all.sort_values(['count', 'row_frac'], ascending=[False, False]).head(top_k)
    ins_df = pd.DataFrame([{'char': ch, 'count': cnt} for ch, cnt in ins_counter.most_common()]).head(top_k)
    del_df = pd.DataFrame([{'char': ch, 'count': cnt} for ch, cnt in del_counter.most_common()]).head(top_k)
    paths = {}
    if save_outputs:
        out_prefix = Path(out_prefix)
        out_prefix.mkdir(parents=True, exist_ok=True)
        with open(out_prefix / 'error_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump({'overall': overall, 'top_worst_normalized_subs': worst_pairs[:max(15, top_k)]}, f, indent=2)
        paths['summary_json'] = str(out_prefix / 'error_analysis_summary.json')
        pd.DataFrame(per_sample_rows).to_csv(out_prefix / 'per_sample_metrics.csv', index=False)
        paths['per_sample_csv'] = str(out_prefix / 'per_sample_metrics.csv')
        per_char_df.to_csv(out_prefix / 'per_char_error.csv', index=False)
        paths['per_char_csv'] = str(out_prefix / 'per_char_error.csv')
        conf_df_raw.to_csv(out_prefix / 'confusion_matrix_raw.csv')
        paths['conf_matrix_raw_csv'] = str(out_prefix / 'confusion_matrix_raw.csv')
        conf_df_norm.to_csv(out_prefix / 'confusion_matrix_row_normalized.csv')
        paths['conf_matrix_norm_csv'] = str(out_prefix / 'confusion_matrix_row_normalized.csv')
        subs_df_all.to_csv(out_prefix / 'substitutions_all.csv', index=False)
        paths['subs_all_csv'] = str(out_prefix / 'substitutions_all.csv')
        pd.DataFrame([{'char': ch, 'count': cnt} for ch, cnt in ins_counter.most_common()]).to_csv(out_prefix / 'insertions_all.csv', index=False)
        paths['ins_all_csv'] = str(out_prefix / 'insertions_all.csv')
        pd.DataFrame([{'char': ch, 'count': cnt} for ch, cnt in del_counter.most_common()]).to_csv(out_prefix / 'deletions_all.csv', index=False)
        paths['del_all_csv'] = str(out_prefix / 'deletions_all.csv')
    if print_reports:
        print('\n=== Overall ===')
        for k, v in overall.items():
            print(f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}')
        print('\n=== Top {} most error-prone characters (by error_rate) ==='.format(top_k))
        print(per_char_df.head(top_k).to_string(index=False))
        print('\n=== Top {} substitutions (ROW-NORMALIZED by truth support) ==='.format(top_k))
        if len(subs_df_norm) > 0:
            print(subs_df_norm.to_string(index=False))
        else:
            print('(none)')
        print('\n=== Top {} substitutions (RAW counts, NOT normalized) ==='.format(top_k))
        if len(subs_df_raw) > 0:
            print(subs_df_raw.to_string(index=False))
        else:
            print('(none)')
        print('\n=== Top {} insertions (spurious predicted chars) ==='.format(top_k))
        if len(ins_df) > 0:
            print(ins_df.to_string(index=False))
        else:
            print('(none)')
        print('\n=== Top {} deletions (missed truth chars) ==='.format(top_k))
        if len(del_df) > 0:
            print(del_df.to_string(index=False))
        else:
            print('(none)')
        print('\n=== Top {} worst normalized substitutions (truth -> pred) from confusion matrix ==='.format(top_k))
        for row in worst_pairs[:top_k]:
            print(f"{row['truth']} -> {row['pred']}: frac={row['frac']:.3f}  count={row['count']}  (subs from '{row['truth']}' total={row['row_total']})")
    return {'overall': overall, 'paths': paths, 'per_char_df': per_char_df, 'conf_df_raw': conf_df_raw, 'conf_df_norm': conf_df_norm, 'subs_df_all': subs_df_all}

def display_edit_ops_tables(data_root: str, ckpt: str, split: str='val', top_n: int=25, device: str='auto'):
    """
    display_edit_ops_tables.
    """
    dev = 'cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu'
    ckpt_obj = torch.load(ckpt, map_location=dev)
    alphabet = ckpt_obj.get('alphabet', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    img_h = int(ckpt_obj.get('img_h', 64))
    img_w = int(ckpt_obj.get('img_w', 256))
    rgb = bool(ckpt_obj.get('rgb', False))

    rnn_hidden = int(ckpt_obj.get('rnn_hidden', 256))
    rnn_layers = int(ckpt_obj.get('rnn_layers', 2))
    num_classes = len(alphabet) + 1
    idx2char = {i + 1: c for i, c in enumerate(alphabet)}
    
    model = VGG16_LSTM_CTC(num_classes=num_classes, img_h=img_h, rgb=rgb, 
                           rnn_hidden=rnn_hidden, rnn_layers=rnn_layers, pretrained=False).to(dev)
    model.load_state_dict(ckpt_obj['model'])
    model.eval()
    
    data_root = Path(data_root)
    loader, ds = build_loader_for_split(data_root, split, img_h, img_w, alphabet, rgb)
    sub_counter = Counter()
    ins_counter = Counter()
    del_counter = Counter()
    truth_support = Counter()
    
    characters = list(alphabet)
    
    V = len(characters)
    char_to_idx = {c: i for i, c in enumerate(characters)}
    conf_raw = np.zeros((V, V), dtype=np.int64)
    
    with torch.no_grad():
        for batch in loader:
            images, flat_targets, target_lens, label_strs = batch
            images = images.to(dev).float()
            logits = model(images)
            preds = ctc_greedy_decode(logits, idx2char)
            for g, p in zip(label_strs, preds):
                for ch in g:
                    truth_support[ch] += 1
                ops, _ = levenshtein_backtrace(g, p)
                for op, a_ch, b_ch in ops:
                    if op == 'sub':
                        sub_counter[a_ch, b_ch] += 1
                        if a_ch in char_to_idx and b_ch in char_to_idx:
                            conf_raw[char_to_idx[a_ch], char_to_idx[b_ch]] += 1
                    elif op == 'ins':
                        if b_ch is not None:
                            ins_counter[b_ch] += 1
                    elif op == 'del':
                        if a_ch is not None:
                            del_counter[a_ch] += 1
    rows = []
    for (t, p), cnt in sub_counter.items():
        den = truth_support[t]
        frac = cnt / den if den else 0.0
        rows.append({'truth': t, 'pred': p, 'count': cnt, 'truth_support': den, 'row_frac': frac})
    subs_df = pd.DataFrame(rows).sort_values(['row_frac', 'count'], ascending=[False, False]).head(top_n)
    ins_df = pd.DataFrame([{'char': ch, 'count': cnt} for ch, cnt in ins_counter.most_common()]).head(top_n)
    del_df = pd.DataFrame([{'char': ch, 'count': cnt} for ch, cnt in del_counter.most_common()]).head(top_n)
    if display_dataframe_to_user is not None:
        display_dataframe_to_user('Top substitutions (row-normalized by truth char support)', subs_df)
        display_dataframe_to_user('Top insertions (spurious)', ins_df)
        display_dataframe_to_user('Top deletions (missed truth chars)', del_df)
    else:
        print('\nTop substitutions (row-normalized):\n', subs_df.head(top_n).to_string(index=False))
        print('\nTop insertions:\n', ins_df.head(top_n).to_string(index=False))
        print('\nTop deletions:\n', del_df.head(top_n).to_string(index=False))
    return (subs_df, ins_df, del_df)

def display_top15_edit_ops_inline(data_root: str, ckpt: str, split: str='val', device: str='auto'):
    """display_top15_edit_ops_inline.

Args:
    data_root: Argument.
    ckpt: Argument.
    split: Argument.
    device: Argument.

Returns:
    None"""
    return display_edit_ops_tables(data_root=data_root, ckpt=ckpt, split=split, top_n=15, device=device)

def display_substitutions_dual_tables(data_root: str, ckpt: str, split: str='val', top_n: int=25, device: str='auto'):
    """display_substitutions_dual_tables."""

    alphabet, sub_counter, ins_counter, del_counter, truth_support = _collect_ops_counters_for_split(data_root=data_root, ckpt=ckpt, split=split, device=device)
    rows = []
    for (t, p), cnt in sub_counter.items():
        den = truth_support[t]
        frac = cnt / den if den else 0.0
        rows.append({'truth': t, 'pred': p, 'count': cnt, 'truth_support': den, 'row_frac': frac})
    df_all = pd.DataFrame(rows)
    subs_df_norm = df_all.sort_values(['row_frac', 'count'], ascending=[False, False]).head(top_n)
    subs_df_raw = df_all.sort_values(['count', 'row_frac'], ascending=[False, False]).head(top_n)
    
    if display_dataframe_to_user is not None:
        display_dataframe_to_user('Top substitutions (ROW-NORMALIZED by truth support)', subs_df_norm)
        display_dataframe_to_user('Top substitutions (RAW counts, NOT normalized)', subs_df_raw)
    else:
        print('\nTop substitutions (ROW-NORMALIZED):\n', subs_df_norm.to_string(index=False))
        print('\nTop substitutions (RAW counts):\n', subs_df_raw.to_string(index=False))
    
    return (subs_df_norm, subs_df_raw)

def display_top15_substitutions_dual_inline(data_root: str, ckpt: str, split: str='val', device: str='auto'):
    """
    display_top15_substitutions_dual_inline.
    """
    return display_substitutions_dual_tables(data_root=data_root, ckpt=ckpt, split=split, top_n=15, device=device)

def _collect_ops_counters_for_split(data_root: str, ckpt: str, split: str='val', device: str='auto'):
    """Internal: run model on split and collect substitution/ins/del counters + truth support."""
    dev = 'cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu'
    ckpt_obj = torch.load(ckpt, map_location=dev)
    alphabet = ckpt_obj.get('alphabet', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    img_h = int(ckpt_obj.get('img_h', 64))
    img_w = int(ckpt_obj.get('img_w', 256))
    rgb = bool(ckpt_obj.get('rgb', False))
    rnn_hidden = int(ckpt_obj.get('rnn_hidden', 256))
    rnn_layers = int(ckpt_obj.get('rnn_layers', 2))
    num_classes = len(alphabet) + 1
    
    idx2char = {i + 1: c for i, c in enumerate(alphabet)}
    
    model = VGG16_LSTM_CTC(num_classes=num_classes, img_h=img_h, rgb=rgb, rnn_hidden=rnn_hidden, rnn_layers=rnn_layers, pretrained=False).to(dev)
    model.load_state_dict(ckpt_obj['model'])
    model.eval()
    
    data_root = Path(data_root)
    loader, ds = build_loader_for_split(data_root, split, img_h, img_w, alphabet, rgb)
    
    sub_counter = Counter()
    ins_counter = Counter()
    del_counter = Counter()
    truth_support = Counter()
    
    with torch.no_grad():
        for batch in loader:
            images, flat_targets, target_lens, label_strs = batch
            images = images.to(dev).float()
            logits = model(images)
            preds = ctc_greedy_decode(logits, idx2char)
            for g, p in zip(label_strs, preds):
                for ch in g:
                    truth_support[ch] += 1
                ops, _ = levenshtein_backtrace(g, p)
                for op, a_ch, b_ch in ops:
                    if op == 'sub':
                        sub_counter[a_ch, b_ch] += 1
                    elif op == 'ins':
                        if b_ch is not None:
                            ins_counter[b_ch] += 1
                    elif op == 'del':
                        if a_ch is not None:
                            del_counter[a_ch] += 1
    return (alphabet, sub_counter, ins_counter, del_counter, truth_support)

def main():
    """
    main."""
    ap = argparse.ArgumentParser(description='Evaluate captcha recognizer and export error analysis.')
    ap.add_argument('--data_root', type=str, required=True, help='Folder with train/ val/ test/')
    ap.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint saved by training')
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    ap.add_argument('--out_prefix', type=str, default='/mnt/data/')
    ap.add_argument('--top_k', type=int, default=15, help='How many items to print for each top list')
    ap.add_argument('--no_save', action='store_true', help='Print-only mode: do not write any files')
    args = ap.parse_args()
    _ = run_eval(data_root=args.data_root, ckpt_path=args.ckpt, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device, split=args.split, out_prefix=args.out_prefix, print_reports=True, top_k=args.top_k, save_outputs=not args.no_save)


if __name__ == '__main__':
    main()