import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from train_crnn_ctc import CaptchaDataset, read_csv, ctc_collate, CRNN, ctc_greedy_decode, cer

@torch.no_grad()
def evaluate_metrics(model, loader, device, idx2char):
    model.eval()
    all_preds, all_labels = [], []
    all_cer, correct = [], 0

    for images, _, _, label_strs in loader:
        images = images.to(device).float()
        logits = model(images)
        preds = ctc_greedy_decode(logits, idx2char)

        all_preds.extend(preds)
        all_labels.extend(label_strs)

        for p, g in zip(preds, label_strs):
            if p == g: correct += 1
            all_cer.append(cer(p, g))

    acc = correct / len(loader.dataset)
    avg_cer = float(sum(all_cer) / len(all_cer))

    # For precision/recall we compute at character level
    y_true = list("".join(all_labels))
    y_pred = list("".join(all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {"acc": acc, "cer": avg_cer, "precision": precision, "recall": recall, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--annotations", type=str, default="annotations.csv")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load annotations and alphabet
    data_root = Path(args.data_root)
    rows = read_csv(data_root / args.annotations)

    ckpt = torch.load(args.model_path, map_location=device)
    alphabet = ckpt["alphabet"]
    model = CRNN(num_classes=len(alphabet)+1, img_h=args.img_h).to(device)
    model.load_state_dict(ckpt["model"])

    idx2char = {i+1: c for i, c in enumerate(alphabet)}

    ds = CaptchaDataset(data_root, rows, args.img_h, args.img_w, alphabet, aug=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=ctc_collate)

    metrics = evaluate_metrics(model, loader, device, idx2char)
    print("[RESULTS]", metrics)

if __name__ == "__main__":
    main()
