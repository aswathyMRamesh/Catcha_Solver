# ------------------------------------------------------------
# CAPTCHA-friendly augmentations (Grayscale version):
# - Image-only (PIL->PIL) custom ops
# - Resize -> ToTensor -> Normalize (1-channel) -> (optional) Erasing
# ------------------------------------------------------------

import math, random
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageFont

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


__all__ = [
    "get_transforms",
    "build_default_augmentation_pipeline",
    # individual ops

    "RandomSafeRotate", "RandomSafeRotateCrop", "build_rotation", "RandomSafeShear", "RandomSafeTranslate",
    "AddGaussianNoise", "AddLines", "ColorContrastJitter",
    "RandomBrightness",
    "RandomShrinkIntoCanvas", "RandomEnlargeSafely",
    "AddBlobs", "AddDots", "AddSymbolDistractors", "AddNonASCIIChars",
    "RandomElasticDistortion", "RandomSinusoidalDistortion", "SimulateCharacterOverlap",
]

# ---- Aug toggles (defaults) ----
DEFAULT_AUG_FLAGS = {
    "rotate": True,          # RandomSafeRotate / RandomSafeRotateCrop
    "shear": True,           # RandomSafeShear
    "translate": True,       # RandomSafeTranslate
    "gaussian_noise": True,  # AddGaussianNoise
    "lines": True,           # AddLines
    "color_contrast": True,  # ColorContrastJitter
    "brightness": True,      # RandomBrightness
    "shrink": True,          # RandomShrinkIntoCanvas
    "enlarge": True,         # RandomEnlargeSafely
    "blobs": True,           # AddBlobs
    "dots": True,            # AddDots
    "elastic": True,         # RandomElasticDistortion
    "sine": True,            # RandomSinusoidalDistortion
    "symbol_distractors": True, # AddSymbolDistractors used for glyphs only(For size)
    "glyphs": True,          # AddNonASCIIChars (size cannot be changed due basic prompt)
    "overlap_sim": True,     # Overlap charcters with random small patch in the image to simulte overlap confusion

}


# Small helpers (grayscale)


def _bg_gray_from_corners(img: Image.Image) -> int:
    """Estimate background gray level as the average of 4 corner pixels."""
    g = img.convert("L")
    w, h = g.size
    pts = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    vals = [g.getpixel(p) for p in pts]
    return int(sum(vals) / 4)

def _pad(img: Image.Image, px: int) -> Image.Image:
    """Uniform pad with estimated gray background."""
    if px <= 0:
        return img
    return ImageOps.expand(img, border=px, fill=_bg_gray_from_corners(img))

def _resize_back(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize to target (W,H) without cropping (avoids losing border after affine)."""
    w, h = size
    return img.resize((w, h), resample=Image.BILINEAR)

def _pad_needed_for_rotation_shear(w: int, h: int, angle_deg: float, shear_deg: float) -> int:
    """Conservative padding to avoid cropping under rotation+shear."""
    ang = abs(math.radians(angle_deg))
    sh  = abs(math.radians(shear_deg))
    nW = abs(w*math.cos(ang)) + abs(h*math.sin(ang))
    nH = abs(w*math.sin(ang)) + abs(h*math.cos(ang))
    nW *= (1.0 + 0.5*math.tan(sh))
    nH *= (1.0 + 0.5*math.tan(sh))
    pad_w = max(0, int(math.ceil((nW - w)/2)))
    pad_h = max(0, int(math.ceil((nH - h)/2)))
    return max(pad_w, pad_h) + 6  # small safety margin

def _pil_to_torch(img: Image.Image) -> torch.Tensor:
    """PIL (L) -> torch float tensor [1,1,H,W] in [0,1]."""
    g = np.array(img.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(g)[None, None, ...]  # [1,1,H,W]
    return t

def _torch_to_pil(t: torch.Tensor) -> Image.Image:
    """torch [1,1,H,W] -> PIL.Image L."""
    t = t.clamp(0, 1).squeeze(0).squeeze(0)  # [H,W]
    arr = (t.cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Backward-compatible text size (Pillow 10+: textbbox)."""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    return draw.textsize(text, font=font)

def _safe_center_crop_L(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Center-crop to (W,H) in grayscale."""
    w, h = size
    return TF.center_crop(img.convert("L"), [h, w])

# PIL -> PIL transforms (grayscale)

class RandomSafeRotate:
    def __init__(self, degrees: float = 18.0, p: float = 0.7):
        self.degrees = degrees; self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        angle = random.uniform(-self.degrees, self.degrees)
        pad_px = _pad_needed_for_rotation_shear(w, h, angle, 0.0)
        work = _pad(img, pad_px)
        out = TF.affine(
            work, angle=angle, translate=(0, 0), scale=1.0, shear=[0.0, 0.0],
            interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work)
        )
        # changed: resize back instead of center-crop
        return _resize_back(out, (w, h))


class RandomSafeRotateCrop:
    """
    Rotation that avoids border artifacts by padding first,
    then rotates and finally center-crops back to (w,h).
    """
    def __init__(self, degrees: float = 18.0, p: float = 0.7):
        self.degrees = degrees
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img = img.convert("L")
        w, h = img.size
        angle = random.uniform(-self.degrees, self.degrees)
        pad_px = _pad_needed_for_rotation_shear(w, h, angle, 0.0)
        work = _pad(img, pad_px)
        out = TF.affine(
            work, angle=angle, translate=(0, 0), scale=1.0, shear=[0.0, 0.0],
            interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work)
        )
        return _safe_center_crop_L(out, (w, h))


class RandomSafeShear:
    def __init__(self, shear: float = 8.0, p: float = 0.5):
        self.shear = shear; self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        sh = random.uniform(-self.shear, self.shear)
        pad_px = _pad_needed_for_rotation_shear(w, h, 0.0, sh)
        work = _pad(img, pad_px)
        out = TF.affine(
            work, angle=0.0, translate=(0, 0), scale=1.0, shear=[sh, 0.0],
            interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work)
        )
        return _resize_back(out, (w, h))
        

class RandomSafeTranslate:
    def __init__(self, translate=(0.08, 0.08), p: float = 0.6):
        self.translate = translate; self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        tx = random.uniform(-self.translate[0] * w, self.translate[0] * w)
        ty = random.uniform(-self.translate[1] * h, self.translate[1] * h)
        pad_px = int(max(abs(tx), abs(ty))) + 6
        work = _pad(img, pad_px)
        out = TF.affine(
            work, angle=0.0, translate=(int(tx), int(ty)), scale=1.0, shear=[0.0, 0.0],
            interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work)
        )
        return _resize_back(out, (w, h))
        

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=8.0, p: float = 0.6):
        self.mean = mean; self.std = std; self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        g = np.array(img.convert("L")).astype(np.float32)
        noise = np.random.normal(self.mean, self.std, g.shape)
        g = np.clip(g + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(g, mode="L")
        

class AddLines:
    def __init__(self, n_lines=(1, 3), width=(1, 2), p: float = 0.7):
        self.n_lines = n_lines; self.width = width; self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        draw = ImageDraw.Draw(img)
        w, h = img.size; bg = _bg_gray_from_corners(img)
        for _ in range(random.randint(*self.n_lines)):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            col = int(max(0, min(255, bg + random.randint(-60, 60))))
            draw.line((x1, y1, x2, y2), fill=col, width=random.randint(*self.width))
        return img

class ColorContrastJitter:
    """Works on grayscale: brightness + contrast."""
    def __init__(self, brightness=0.2, contrast=0.3, p: float = 0.7):
        self.brightness=brightness; self.contrast=contrast; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        if self.brightness > 0:
            img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-self.contrast, self.contrast))
        return img
        

class RandomBrightness:
    """Explicit brightness-only jitter (grayscale)."""
    def __init__(self, max_delta: float = 0.25, p: float = 0.5):
        self.max_delta = max_delta; self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        factor = 1.0 + random.uniform(-self.max_delta, self.max_delta)
        return ImageEnhance.Brightness(img).enhance(factor)
        

class RandomShrinkIntoCanvas:
    def __init__(self, min_scale=0.82, max_scale=0.95, p: float = 0.45):
        self.min_scale=min_scale; self.max_scale=max_scale; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        nw, nh = max(1, int(w*s)), max(1, int(h*s))
        small = img.resize((nw, nh), resample=Image.BILINEAR)
        bg = Image.new("L", (w, h), _bg_gray_from_corners(img))
        x = random.randint(0, w - nw); y = random.randint(0, h - nh)
        bg.paste(small, (x, y))
        return bg
        

class RandomEnlargeSafely:
    """Zoom-in without cropping text: pad -> resize -> resize back (no crop)."""
    def __init__(self, min_scale=1.05, max_scale=1.20, p: float = 0.45):
        self.min_scale=min_scale; self.max_scale=max_scale; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        pad = int(math.ceil((max(w, h) * (s - 1.0)) / 2)) + 10
        work = _pad(img, pad)
        nw, nh = int(work.size[0] * s), int(work.size[1] * s)
        zoom = work.resize((nw, nh), resample=Image.BILINEAR)
        return _resize_back(zoom, (w, h))
        

class AddBlobs:
    def __init__(self, n_blobs=(1, 3), p: float = 0.5):
        self.n_blobs=n_blobs; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        draw = ImageDraw.Draw(img)
        w, h = img.size; bg = _bg_gray_from_corners(img)
        for _ in range(random.randint(*self.n_blobs)):
            shape = random.choice(["rect", "circle", "tri"])
            col = int(max(0, min(255, bg + random.randint(-80, 80))))
            x1, y1 = random.randint(0, w-5), random.randint(0, h-5)
            x2, y2 = random.randint(x1+3, min(w, x1 + w//3)), random.randint(y1+3, min(h, y1 + h//3))
            if shape == "rect":
                draw.rectangle([x1, y1, x2, y2], outline=col, width=1)
            elif shape == "circle":
                draw.ellipse([x1, y1, x2, y2], outline=col, width=1)
            else:
                draw.polygon([(x1, y2), ((x1+x2)//2, y1), (x2, y2)], outline=col)
        return img


class AddDots:
    """Add many tiny grayscale dots (speckle)."""
    def __init__(self, n_dots: Tuple[int,int]=(80, 180), radius: Tuple[int,int]=(0, 1), p: float = 0.6):
        self.n_dots = n_dots
        self.radius = radius
        self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: 
            return img
        img = img.convert("L")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        bg = _bg_gray_from_corners(img)
        nd = random.randint(*self.n_dots)
        rmin, rmax = self.radius
        for _ in range(nd):
            r = random.randint(max(0, rmin), max(0, rmax))
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            col = int(max(0, min(255, bg + random.randint(-70, 70))))
            if r <= 0:
                draw.point((x, y), fill=col)
            else:
                draw.ellipse((x-r, y-r, x+r, y+r), fill=col, outline=None)
        return img
        

class RandomElasticDistortion:
    """Moderate elastic warp using torch.grid_sample (no SciPy)."""
    def __init__(self, p: float = 0.30, alpha: float = 2.0, sigma: float = 8.0):
        self.p=p; self.alpha=alpha; self.sigma=sigma
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        t = _pil_to_torch(img)  # [1,1,H,W]
        _, _, H, W = t.shape
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None]  # [1,H,W,2]

        dx, dy = torch.randn((1,1,H,W)), torch.randn((1,1,H,W))
        k = max(3, int(2*round(self.sigma)+1))
        coords = torch.arange(k) - k//2
        g1d = torch.exp(-(coords**2)/(2*self.sigma**2)); g1d = (g1d/g1d.sum()).view(1,1,k,1)
        dx = F.conv2d(dx, g1d, padding=(k//2,0)); dx = F.conv2d(dx, g1d.transpose(2,3), padding=(0,k//2))
        dy = F.conv2d(dy, g1d, padding=(k//2,0)); dy = F.conv2d(dy, g1d.transpose(2,3), padding=(0,k//2))

        disp_x = (dx.squeeze(1) * self.alpha) / W
        disp_y = (dy.squeeze(1) * self.alpha) / H
        disp   = torch.stack((disp_x.squeeze(0), disp_y.squeeze(0)), dim=-1).unsqueeze(0)
        warped = F.grid_sample(t, grid + disp, mode="bilinear", padding_mode="border", align_corners=True)
        return _torch_to_pil(warped)


class RandomSinusoidalDistortion:
    """Gentle sinusoidal wobble."""
    def __init__(self, p: float = 0.30, amp: float = 0.02, freq: float = 1.4):
        self.p=p; self.amp=amp; self.freq=freq
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        t = _pil_to_torch(img)  # [1,1,H,W]
        _, _, H, W = t.shape
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None]
        disp_x = self.amp * torch.sin(2*math.pi*self.freq*yy)
        disp_y = self.amp * torch.sin(2*math.pi*self.freq*xx)
        disp = torch.stack((disp_x, disp_y), dim=-1).unsqueeze(0)
        warped = F.grid_sample(t, grid + disp, mode="bilinear", padding_mode="border", align_corners=True)
        return _torch_to_pil(warped)

# class AddNonASCIIChars:
#     """Distractor glyphs; keep LAST in pipeline (grayscale + alpha)."""
#     def __init__(self, glyphs="*#?✓", n_chars=(1,3), p: float = 0.6, size_frac=(0.45, 0.70)):
#         self.glyphs=list(glyphs); self.n_chars=n_chars; self.p=p; self.size_frac=size_frac
#     def __call__(self, img: Image.Image) -> Image.Image:
#         if random.random() > self.p: return img
#         w, h = img.size
#         base = img.convert("LA")  # grayscale with alpha
#         bg = _bg_gray_from_corners(img)
#         for _ in range(random.randint(*self.n_chars)):
#             g = random.choice(self.glyphs)
#             fsize = max(12, int(random.uniform(*self.size_frac) * h))
#             try: font = ImageFont.truetype(font=None, size=fsize)
#             except: font = ImageFont.load_default()
#             # Render glyph tile
#             dummy = Image.new("LA", (fsize*3, fsize*3), (0,0))
#             d = ImageDraw.Draw(dummy)
#             tw, th = _text_size(d, g, font)
#             tile = Image.new("LA", (tw+12, th+12), (0,0))
#             td = ImageDraw.Draw(tile)
#             col = int(max(0, min(255, bg + random.randint(-100, 100))))
#             td.text((6,6), g, fill=(col, 200), font=font)  # semi-opaque glyph
#             tile = tile.rotate(random.uniform(-25, 25), resample=Image.BICUBIC, expand=True)
#             tw2, th2 = tile.size
#             if tw2 >= w or th2 >= h:
#                 continue
#             x = random.randint(0, w - tw2); y = random.randint(0, h - th2)
#             base.alpha_composite(tile, (x, y))
#         return base.convert("L")

class AddNonASCIIChars:
    """Distractor glyphs; keep LAST in pipeline (grayscale + alpha)."""
    def __init__(self, glyphs="*#?✓", n_chars=(1,3), p: float = 0.6, size_frac=(0.45, 0.70)):
        self.glyphs=list(glyphs); self.n_chars=n_chars; self.p=p; self.size_frac=size_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: 
            return img

        w, h = img.size
        base = img.convert("L")             # <-- L (not LA)
        bg = _bg_gray_from_corners(img)

        for _ in range(random.randint(*self.n_chars)):
            g = random.choice(self.glyphs)
            fsize = max(12, int(random.uniform(*self.size_frac) * h))
            try:
                font = ImageFont.truetype(font=None, size=fsize)
            except:
                font = ImageFont.load_default()

            # make an LA tile
            dummy = Image.new("LA", (fsize*3, fsize*3), (0,0))
            d = ImageDraw.Draw(dummy)
            tw, th = _text_size(d, g, font)
            tile = Image.new("LA", (tw+12, th+12), (0,0))
            td = ImageDraw.Draw(tile)

            gray = int(max(0, min(255, bg + random.randint(-100, 100))))
            alpha = 200
            td.text((6,6), g, fill=(gray, alpha), font=font)

            tile = tile.rotate(random.uniform(-25, 25), resample=Image.BICUBIC, expand=True)
            tw2, th2 = tile.size
            if tw2 >= w or th2 >= h:
                continue

            x = random.randint(0, w - tw2); y = random.randint(0, h - th2)

            # composite in grayscale: split LA -> (L, A) and paste with mask
            tile_L, tile_A = tile.split()
            base.paste(tile_L, (x, y), mask=tile_A)   # <-- no alpha_composite

        return base
    
    
class AddSymbolDistractors:
    """
    Adds synthetic visual clutter using simple geometric symbols
    (like *, #, ?, &, tick) without relying on fonts.
    Can be used in a torchvision-style augmentation pipeline.
    """
    def __init__(self, symbols=None, n_symbols=(2, 5), p: float = 0.5):
        self.symbol_set = symbols or ['*', 'tick', '?', '#', '&']
        self.count_range = n_symbols
        self.probability = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image

        canvas = image.copy()
        pen = ImageDraw.Draw(canvas)
        w, h = canvas.size
        num = random.randint(*self.count_range)

        for _ in range(num):
            shape = random.choice(self.symbol_set)
            ox = random.randint(0, max(1, w - 30))
            oy = random.randint(0, max(1, h - 30))

            if shape == '*':
                mid = 15
                pen.line([ox, oy+mid, ox+30, oy+mid], fill=255, width=2)
                pen.line([ox+mid, oy, ox+mid, oy+30], fill=255, width=2)
                pen.line([ox, oy, ox+30, oy+30], fill=255, width=2)
                pen.line([ox, oy+30, ox+30, oy], fill=255, width=2)

            elif shape == '#':
                pen.line([ox+5, oy, ox+5, oy+30], fill=255, width=2)
                pen.line([ox+15, oy, ox+15, oy+30], fill=255, width=2)
                pen.line([ox, oy+10, ox+25, oy+10], fill=255, width=2)
                pen.line([ox, oy+20, ox+25, oy+20], fill=255, width=2)

            elif shape == '?':
                pen.arc([ox, oy, ox+20, oy+20], start=0, end=180, fill=255)
                pen.line([ox+10, oy+20, ox+10, oy+25], fill=255, width=2)
                pen.ellipse([ox+9, oy+28, ox+11, oy+30], fill=255)

            elif shape == '&':
                pen.ellipse([ox, oy, ox+20, oy+20], outline=255)
                pen.line([ox+10, oy+10, ox+25, oy+25], fill=255, width=2)

            elif shape == 'tick':
                pen.line([ox, oy+15, ox+10, oy+25], fill=255, width=2)
                pen.line([ox+10, oy+25, ox+25, oy], fill=255, width=2)

        return canvas

class SimulateCharacterOverlap:
    """
    Simulates overlapping character regions by duplicating and shifting
    rectangular patches of the image. No labels required.
    """
    def __init__(self, n_overlaps=(1, 3), max_shift=(5, 15), patch_size=(20, 40), p: float = 0.5):
        self.n_overlaps = n_overlaps
        self.max_shift = max_shift
        self.patch_size = patch_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img = img.convert("L")
        w, h = img.size
        draw_img = img.copy()

        for _ in range(random.randint(*self.n_overlaps)):
            pw = random.randint(*self.patch_size)
            ph = random.randint(*self.patch_size)
            x1 = random.randint(0, max(1, w - pw - 1))
            y1 = random.randint(0, max(1, h - ph - 1))

            patch = img.crop((x1, y1, x1 + pw, y1 + ph))

            dx = random.randint(-self.max_shift[0], self.max_shift[0])
            dy = random.randint(-self.max_shift[1], self.max_shift[1])

            x2 = min(max(0, x1 + dx), w - pw)
            y2 = min(max(0, y1 + dy), h - ph)

            draw_img.paste(patch, (x2, y2))

        return draw_img


# Pipeline builders


def build_rotation(mode: str = "resize", degrees: float = 18.0, p: float = 0.7):
    """
    mode: 'resize' -> pad+rotate+resize-back  (keeps all content, may slightly shrink/blur)
          'crop'   -> pad+rotate+center-crop (keeps scale crisper, may trim corners)
    """
    m = mode.lower().strip()
    if m == "resize":
        return RandomSafeRotate(degrees=degrees, p=p)
    elif m == "crop":
        return RandomSafeRotateCrop(degrees=degrees, p=p)
    else:
        raise ValueError("rotation mode must be 'resize' or 'crop'")


def build_default_augmentation_pipeline(
    aug_flags: dict = None,
    rotation_mode: str = "resize",
) -> T.Compose:
    """
    Build PIL->PIL augmentation pipeline with per-op flags.
      - rotation_mode: 'resize' (pad+rotate+resize-back) or 'crop' (pad+rotate+center-crop)
      - aug_flags: dict to toggle each op (see DEFAULT_AUG_FLAGS keys)
    """
    f = dict(DEFAULT_AUG_FLAGS)
    if aug_flags:
        f.update(aug_flags)

    ops = []

    # geometric
    if f["rotate"]:
        ops.append(build_rotation(rotation_mode, degrees=18, p=0.7))
    if f["shear"]:
        ops.append(RandomSafeShear(shear=8, p=0.5))
    if f["translate"]:
        ops.append(RandomSafeTranslate(translate=(0.08, 0.08), p=0.6))

    # photometric/clutter
    if f["gaussian_noise"]:
        ops.append(AddGaussianNoise(std=8.0, p=0.6))
    if f["lines"]:
        ops.append(AddLines(n_lines=(1, 3), width=(1, 2), p=0.7))
    if f["color_contrast"]:
        ops.append(ColorContrastJitter(brightness=0.2, contrast=0.3, p=0.7))
    if f["brightness"]:
        ops.append(RandomBrightness(max_delta=0.25, p=0.5))

    # scale/layout
    if f["shrink"]:
        ops.append(RandomShrinkIntoCanvas(min_scale=0.82, max_scale=0.95, p=0.45))
    if f["enlarge"]:
        ops.append(RandomEnlargeSafely(min_scale=1.05, max_scale=1.20, p=0.45))

    # clutter dots/shapes
    if f["blobs"]:
        ops.append(AddBlobs(n_blobs=(1, 3), p=0.5))
    if f["dots"]:
        ops.append(AddDots(n_dots=(80, 180), radius=(0, 1), p=0.6))

    # mild warps
    if f["elastic"]:
        ops.append(RandomElasticDistortion(p=0.30, alpha=2.2, sigma=8.0))
    if f["sine"]:
        ops.append(RandomSinusoidalDistortion(p=0.30, amp=0.02, freq=1.4))

    # distractor glyphs last
    if f["glyphs"]:
        ops.append(AddNonASCIIChars(glyphs="*#?✓", n_chars=(1, 3), p=0.6, size_frac=(0.45, 0.70)))
    if f["symbol_distractors"]:  # <-- check the flag
        ops.append(AddSymbolDistractors(p=0.5, n_symbols=(2, 5)))
    if f["overlap_sim"]:
        ops.append(SimulateCharacterOverlap(
            n_overlaps=(1, 2),
            max_shift=(5, 10),
            patch_size=(20, 40),
            p=0.6
        ))

    return T.Compose(ops)


# def build_default_augmentation_pipeline() -> T.Compose:
#     """
#     PIL->PIL compose. Order:
#       - geometric (with safe padding & resize-back),
#       - photometric/clutter (incl. brightness),
#       - mild warps,
#       - distractor glyphs last.
#     """
#     return T.Compose([
#         build_rotation(rotation_mode, degrees=18, p=0.7),
#         RandomSafeShear(shear=8, p=0.5),
#         RandomSafeTranslate(translate=(0.08, 0.08), p=0.6),

#         AddGaussianNoise(std=8.0, p=0.6),
#         AddLines(n_lines=(1, 3), width=(1, 2), p=0.7),
#         ColorContrastJitter(brightness=0.2, contrast=0.3, p=0.7),
#         RandomBrightness(max_delta=0.25, p=0.5),  # explicit brightness jitter

#         RandomShrinkIntoCanvas(min_scale=0.82, max_scale=0.95, p=0.45),
#         RandomEnlargeSafely(min_scale=1.05, max_scale=1.20, p=0.45),

#         AddBlobs(n_blobs=(1, 3), p=0.5),
#         AddDots(n_dots=(80, 180), radius=(0, 1), p=0.6),

#         RandomElasticDistortion(p=0.30, alpha=2.2, sigma=8.0),
#         RandomSinusoidalDistortion(p=0.30, amp=0.02, freq=1.4),

#         AddNonASCIIChars(glyphs="*#?✓", n_chars=(1, 3), p=0.6, size_frac=(0.45, 0.70)),
#     ])

def _grayscale_stats():
    # Single-channel normalization (simple default)
    return [0.5], [0.5]

# def get_transforms(
#     train: bool = True,
#     img_size: Tuple[int, int] = (160, 640),
#     random_erasing_p: float = 0.10,
# ) -> T.Compose:
#     """
#     Returns a torchvision Compose that:
#       (train)   Resize -> PIL Augs -> ToTensor -> Normalize(1ch) -> RandomErasing
#       (val/test) Resize -> ToTensor -> Normalize(1ch)
#     """
#     H, W = img_size
#     mean, std = _grayscale_stats()

#     if train:
#         pil_augs = build_default_augmentation_pipeline()
#         return T.Compose([
#             T.Lambda(lambda im: im.convert("L") if isinstance(im, Image.Image) else im),
#             T.Resize((H, W)),
#             build_default_augmentation_pipeline(rotation_mode="resize"), # or "crop"
#             pil_augs,                 # PIL->PIL augmentations (grayscale)
#             T.ToTensor(),             # -> [1,H,W]
#             T.Normalize(mean=mean, std=std),
#             T.RandomErasing(p=random_erasing_p, scale=(0.02, 0.05), ratio=(0.3, 3.3), value="random"),
#         ])
#     else:
#         return T.Compose([
#             T.Lambda(lambda im: im.convert("L") if isinstance(im, Image.Image) else im),
#             T.Resize((H, W)),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])

def get_transforms(
    train: bool = True,
    img_size: Tuple[int, int] = (160, 640),
    random_erasing_p: float = 0.10,
    rotation_mode: str = "resize",    # 'resize' or 'crop'
    aug_flags: dict = None,           # per-op toggles (see DEFAULT_AUG_FLAGS)
) -> T.Compose:
    """
    Returns a torchvision Compose that:
      (train)   Resize -> PIL Augs -> ToTensor -> Normalize(1ch) -> RandomErasing
      (val/test) Resize -> ToTensor -> Normalize(1ch)
    """
    H, W = img_size
    mean, std = _grayscale_stats()

    if train:
        pil_augs = build_default_augmentation_pipeline(
            aug_flags=aug_flags,
            rotation_mode=rotation_mode
        )
        return T.Compose([
            T.Lambda(lambda im: im.convert("L") if isinstance(im, Image.Image) else im),
            T.Resize((H, W)),
            pil_augs,                 # PIL->PIL augmentations (grayscale)
            T.ToTensor(),             # -> [1,H,W]
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=random_erasing_p, scale=(0.02, 0.05), ratio=(0.3, 3.3), value="random"),
        ])
    else:
        return T.Compose([
            T.Lambda(lambda im: im.convert("L") if isinstance(im, Image.Image) else im),
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
