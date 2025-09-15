# CAPTCHA Detection & Recognition (CRNN + CTC)

A production‑ready pipeline for CAPTCHA sequence recognition using **CRNN backbones** (ResNet‑18 / VGG‑16 / Inception‑v1), **BiLSTM** sequence modeling, and **CTC loss**. The repo includes a rich **augmentation suite** tailored for noisy, distorted CAPTCHA images and optional **refinement modules** 

> **Data scale**: trained on **60k** images (train) and evaluated on **20k** images (test) spanning **part2/part3/part4** splits. Validation uses 20k images.

---

## 🚀 TL;DR
- **Task:** Multi‑character CAPTCHA transcription (0–9, A–Z → 36 symbols) with CTC.
- **Models tried:** CRNN with backbones {ResNet‑18, VGG‑16, Inception‑v1, Custom/Vanilla}; + optional **Adaptive Refiner (AFFN)**; + optional **Spatial Transformer (RST‑STN)**; and a combined **AFFN→STN** variant.
- **Augmentations:** Configurable **per‑op toggles** covering rotation, shear, translation, noise, clutter, photometrics, warps, overlaps, inversion, and perspective.
- **Loss/Decode:** CTC (greedy decoding). Metrics: **CER** and **sequence accuracy**.

---

## 📁 Repository Structure (key files)

```
.
├── data_augmentation.py          # CAPTCHA-specific augmentation pipeline
├── backbone_architecture.py      # ResNet18/VGG16/Inception-v1 backbones for grayscale
├── refine_architecture.py        # Adaptive Refiner (AFFN), RST-STN, and combo
├── rnn_architecture.py           # BiLSTM encoder + additive attention utils
├── demo.ipynb                    # code  run demo
├── captcha_av_project_updated.ipynb      # Train/val loops, decoding, eval & error analysis
└── README.md                     # (this file)
└── utils
     └── error_analysis.py        # Accuracy and Error
     └── export_test_predictions.py # Export Test prediction to json 
     └── visualize_predictions.py   # visualize Predictions
└── results                         # contains json and demo 
└── data                            # dataset structure and annotation.csv file
└── model                           # trained model weights
└── requirements.txt                # File to install modules
```

---

## 📦 Environment

- Python ≥ 3.10
- PyTorch ≥ 2.0, Torchvision ≥ 0.15
- NumPy, Pillow, Matplotlib, scikit‑learn

Install:

```bash
pip install -r requirements.txt
```

> CUDA is auto‑detected; cuDNN benchmark is enabled for speed.

---

## 🗂️ Dataset & Labels

- **Characters:** `0–9A–Z` (36 tokens) + **CTC blank** (= 36) → **num_classes = 37**.
- **Layout:**
  - `train/images/*.png|jpg` and `train/labels/*.txt`
  - `val/images/*` and `val/labels/*`
  - label files contain one index per character, e.g. a line list of class IDs `[0..35]` in sequence order
- **Image size:** resized to **(H,W) = (160, 640)**.
- **Color:** converted to **grayscale** and normalized with mean/std = **0.5/0.5**.

---

## 🔧 Augmentation (train‑time)

The augmentation pipeline (`data_augmentation.py`) is **toggle‑based** with per‑op control via `DEFAULT_AUG_FLAGS`. Each transform operates on grayscale PIL images before conversion to tensors.

### Geometric
- **RandomSafeRotate / RandomSafeRotateCrop** — pad + rotate, then resize/center‑crop (±18°, p=0.7).
- **RandomSafeShear** — shear with safe padding (±8°, p=0.5).
- **RandomSafeTranslate** — shift up to 8% width/height (p=0.6).
- **RandomPerspectiveSafe** — padded perspective warp with mild jitter (p=0.35).

### Noise & Clutter
- **AddGaussianNoise** — Gaussian noise (std=8.0, p=0.6).
- **AddLines** — 1–3 random lines (p=0.7).
- **AddBlobs** — random rect/circle/triangle clutter (p=0.5).
- **AddDots** — 80–180 small speckles (p=0.6).
- **AddSymbolDistractors** — simple geometric clutter (*, #, ?, &, ✓) (p=0.5).
- **AddNonASCIIChars** — overlay glyphs (*#?✓) at 45–70% height (p=0.6).
- **SimulateCharacterOverlap** — patch duplication to mimic overlapping chars (p=0.6).

### Photometric
- **ColorContrastJitter** — brightness ±0.2, contrast ±0.3 (p=0.7).
- **RandomBrightness** — explicit scaling ±0.25 (p=0.5).
- **RandomInvert** — invert intensities (p=0.1).

### Scale/Layout
- **RandomShrinkIntoCanvas** — shrink into padded canvas (scale 0.82–0.95, p=0.45).
- **RandomEnlargeSafely** — zoom with pad + resize back (scale 1.05–1.20, p=0.45).

### Warps
- **RandomElasticDistortion** — elastic grid warp (α≈2.2, σ≈8.0, p=0.3).
- **RandomSinusoidalDistortion** — sinusoidal wobble (amp≈0.02, freq≈1.4, p=0.3).

### Tensor‑stage
- **Resize(160,640) → ToTensor**.
- **Normalize(mean=0.5, std=0.5)**.

## Augmentation Summary (This is augmentation we applied in final model)

| Transform                    | Purpose                                            | Key Args                                 |                   Default p | Stage             | Notes                                                  |
| ---------------------------- | -------------------------------------------------- | ---------------------------------------- | ----------------------------: | ----------------- | ------------------------------------------------------ |
| RandomGaussianNoise        | Add Gaussian noise to emulate sensor noise         | std_px=(1.0, 2.0)                      | 0.10 (mild) / ≥0.15 (medium+) | Noise             | Grayscale-safe; clamps to \[0,255].                    |
| RandomDotsLines            | Sparse dots + thin lines clutter                   | n_dots, n_lines                      | 0.10 (mild) / ≥0.12 (medium+) | Noise             | Uses PIL drawing; black artifacts by default.          |
| RandomEraser               | Small white rectangular erase (occlusion)          | area=(0.01, 0.03)                      |                     0.04–0.05 | Noise             | “Cutout”-style; mild preset only (and optional later). |
| RandomSafeRotate           | Rotation with pre-padding and safe resize-back     | degrees=18                             |                           0.7 | Geometric         | Avoids border cropping; keeps final size.              |
| RandomSafeShear            | Shear with safe padding + resize-back              | shear=8                                |                           0.5 | Geometric         | Uses TF.affine; gray-filled borders.                 |
| RandomSafeTranslate        | Translate with safe padding + resize-back          | translate=(0.08, 0.08)                 |                           0.6 | Geometric         | Pixel-accurate integer paste after pad.                |
| ColorContrastJitter        | Brightness/contrast jitter (grayscale)             | brightness=0.2, contrast=0.3         |                           0.7 | Noise             | Uses ImageEnhance.                                   |
| RandomBrightness           | Brightness-only jitter                             | max_delta=0.25                         |                           0.5 | Noise             | Multiplies image brightness factor.                    |
| RandomShrinkIntoCanvas     | Shrink content and re-center on background         | min_scale=0.82, max_scale=0.95       |                          0.45 | Extra             | Background estimated from corners.                     |
| RandomEnlargeSafely        | Zoom-in (pad → resize → resize-back)               | min_scale=1.05, max_scale=1.20       |                          0.45 | Extra             | Preserves content without cropping.                    |
| AddBlobs                   | Light blob/shape clutter                           | n_blobs=(1,3)                          |                           0.5 | Extra             | Draws rect/circle/tri outlines.                        |
| AddDots                    | Speckle dots (tiny ellipses/points)                | n_dots=(80,180), radius=(0,1)        |                           0.6 | Extra             | Intensity around background ± jitter.                  |
| RandomElasticDistortion    | Elastic warp via grid_sample                     | alpha=2.2, sigma=8.0                 |                          0.30 | Extra             | Smoothed random fields; border padding mode.           |
| RandomSinusoidalDistortion | Gentle sinusoidal wobble                           | amp=0.02, freq=1.4                   |                          0.30 | Extra             | Horizontal & vertical sine displacements.              |
| AddNonASCIIChars           | Font glyph distractors (e.g., * # ? ✓)           | n_chars=(1,3), size_frac             |                           0.6 | Extra (late)      | Renders semi-transparent glyph tiles.                  |
| AddSymbolDistractors       | Font-free geometric symbols (e.g., * # ? & tick) | n_symbols=(2,5)                        |                           0.5 | Extra (late)      | Pure ImageDraw shapes (size ≈ 30 px tiles).          |
| SimulateCharacterOverlap   | Duplicate & shift patches to mimic overlap         | patch_size=(20,40), max_shift=(5,10) |                           0.6 | Extra (late)      | No labels required; copies small regions.              |
| RandomInvert               | Invert grayscale (negative image)                  | —                                        |                          0.10 | Extra (very late) | Useful for robustness to polarity.                     |
| RandomPerspectiveSafe      | Mild perspective warp with padding                 | distortion_scale=0.25                  |                          0.35 | Extra             | Jitters corners; keeps final size.                     |



---

## 🧱 Architecture

### Backbones (grayscale‑aware)
- **ResNet‑18 / VGG‑16 / Inception‑v1** modified to accept **1‑channel** input.
- Effective stride **≈ /16** (de‑striding final stages), then **1×W pooling** → **time sequence**.

### Sequence encoder
- **BiLSTM** (2×hidden 256 by default) over time tokens.
- Optional **additive attention** head for interpretability/visualization.

### Refinement modules (optional)
- **Adaptive Refiner (AFFN)** — nested conv/deconv stages with learnable gated fusion (α in [0,1]).
- **RST‑Spatial Transformer (STN)** — predicts rotation θ, scales sx/sy, translations tx/ty with bounded ranges; composes affine and warps.
- **AFFN→STN** — recommended chain for hard distortions.

### Tried model variants (class names)
- `CRNN_ResNet18_LSTM`, `CRNN_VGG16_LSTM`, `CRNN_InceptionV1_LSTM`, `VanillaCRNN`
- `CRNN_Adaptive_ResNet_LSTM`, `CRNN_STN_ResNet_LSTM`, `CRNN_Adaptive_STN_ResNet_LSTM`

---

## ⚙️ Training

Default settings (see notebook):

- **Optimizer:** Adam (lr=3e-4, weight_decay=0.0)
- **Loss:** `nn.CTCLoss(blank=36, zero_infinity=True)`
- **Scheduler:** `ReduceLROnPlateau(patience=3, factor=0.5)` on val loss
- **Batch size:** 128 (train/val), **epochs:** 20
- **Dataloaders:** `num_workers=2`, `pin_memory=True`
- **Device:** CUDA if available; else CPU

**Run (example):**

```python
from data_augmentation import get_transforms
# pick a model class from your architectures module, e.g. CRNN_Adaptive_STN_ResNet_LSTM
model = CRNN_Adaptive_STN_ResNet_LSTM(num_classes=37, stn_in_channels=3).to(device)
```

Checkpoints are saved on **best val loss** with: epoch, model, optimizer, scheduler, and best_loss.

---

## 🔍 Decoding & Metrics

- **CTC greedy decoding** (collapse repeats, drop blanks).
- **Metrics:**
  - **CER** (character error rate via Levenshtein)
  - **Sequence accuracy** (exact string match)
- **Error analysis utilities:** per‑char error rates; top substitutions/insertions/deletions; normalized confusion.

---

## 📊 Expected Results

> Below are the results of training these models for 20 epochs. Here are there accuracy and CER on validation dataset. 

| Model | Aug Pipeline | CER ↓ | Seq Acc ↑ | Notes |
|---|---|---:|---:|---|
| Vanilla CRNN (NO pretrained weight) | default | 0.1507 | 50.69% | baseline |
| CRNN_ResNet18_LSTM | default | 0.1235 | 57.01% | baseline |
| CRNN_VGG16_LSTM | default | 0.0937 | 68.36% | baseline |
| CRNN_InceptionV1_LSTM | default | 0.2917 | 24.23% | baseline |
| CRNN_Adaptive_ResNet_LSTM | default | 0.1260 | 57.24% | refiner |
| CRNN_STN_ResNet_LSTM | default |  0.1544 | 49.23%  | STN |
| CRNN_Adaptive_STN_ResNet_LSTM | default | 0.1822 | 43.05%   | refiner+STN |

- We proceeded VGG 16 + LSTM as our final model.
---


## Training
The annotations.csv file for train and validation available in `data/part2/train/*` and `data/part2/val/*` folder to use label_mode csv 
```bash
!python train.py \
  --data_root /kaggle/working/part2 \
  --annotations annotations.csv \
  --label_mode csv \
  --img_h 64 --img_w 320 \
  --batch_size 128 --epochs 20 \
  --lr 3e-4 --scheduler cosine \
  --rnn_hidden 320 \
  --use_external_aug \
  --aug_p_geom 0.5 \
  --aug_p_noise 0.5 \
  --aug_gauss_min 2.0 \
  --aug_gauss_max 6.0
```

## 📦 Inference

You can use the helper script `error_analysis.py` to run inference on a folder of images.
Note: we need `train.py` and `final_augmentation.py` in same folder as `error_analysis.py`(and other file) and change your paths
**Error Analysis:**
```bash
!python error_analysis.py \
  --data_root /kaggle/working/part2 \
  --ckpt /kaggle/working/vgg16_lstm_ctc_epoch020.pth \
  --split val \
  --batch_size 128 \
  --num_workers 2 \
  --device auto \
  --out_prefix . 
  ```
**Test Prediction in json**
```bash
!python export_test_predictions.py \
  --data_root /kaggle/working/part2 \
  --ckpt /kaggle/working/vgg16_lstm_ctc_epoch020.pth \
  --out_json test_predictions.json \
  --batch_size 128 \
  --device auto
```

**Visualization**
```bash
!python visualize_predictions.py \
  --data_root /kaggle/working/part2 \
  --ckpt /kaggle/working/vgg16_lstm_ctc_epoch020.pth \
  --split test \
  --num_samples 64 \
  --cols 8 \
  --out_image test_preds_grid.png \
  --out_csv test_preds.csv \
  --device auto
```

- Produces a `test_predictions.json` mapping image paths to `{text, confidence}`.
- Supports visualization with prediction overlays and optional grid.
- Works with all model variants in the repo.

---

## ✍️ Acknowledgements

- Based on standard CRNN + CTC patterns; custom grayscale backbones and CAPTCHA‑specific augmentation.

---

## 📄 License

MIT
