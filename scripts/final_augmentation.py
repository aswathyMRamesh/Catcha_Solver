# Final version Data Augmentation

from torchvision import transforms as T
import random, math
from typing import Tuple, Dict, Optional
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Original Operation

class RandomGaussianNoise:
    """
    Additive i.i.d. Gaussian noise on the grayscale image.

    Args:
        p (float): Probability of applying the transform.
        std_px (Tuple[float,float]): Range for noise STD (pixel units).
    """
    def __init__(self, p=0.10, std_px=(1.0, 2.0)):
        self.p = p
        self.std_px = std_px
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        std = random.uniform(*self.std_px)
        noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

class RandomDotsLines:
    """
    Draw sparse dark dots and (optionally) a few thin lines as clutter.

    Args:
        p (float): Probability to apply.
        n_dots (Tuple[int,int]): Dot count range.
        n_lines (Tuple[int,int]): Line count range.
    """
    def __init__(self, p=0.10, n_dots=(3, 8), n_lines=(0, 2)):
        self.p = p; self.n_dots = n_dots; self.n_lines = n_lines
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        draw = ImageDraw.Draw(img)
        # sparse dots
        for _ in range(random.randint(*self.n_dots)):
            x = random.randint(0, w-1); y = random.randint(0, h-1)
            r = random.randint(0, 1)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=0)
        # rare thin lines
        for _ in range(random.randint(*self.n_lines)):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            draw.line((x1, y1, x2, y2), fill=0, width=1)
        return img

class RandomEraser:
    """
    Erase a small random rectangle to white (background) to simulate occlusion.

    Args:
        p (float): Probability to apply.
        area (Tuple[float,float]): Target area fraction range of the erased patch.
    """
    def __init__(self, p=0.05, area=(0.01, 0.03)):
        self.p = p; self.area = area
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        A = w*h
        a = random.uniform(*self.area) * A
        rw = int(max(1, min(w//8, (a ** 0.5) * random.uniform(0.5, 1.5))))
        rh = int(max(1, min(h//3, a / max(1, rw))))
        x = random.randint(0, max(0, w - rw))
        y = random.randint(0, max(0, h - rh))
        draw = ImageDraw.Draw(img)
        draw.rectangle((x, y, x+rw, y+rh), fill=255)  # erase to white
        return img

# Helpers for advanced ops

def _bg_gray_from_corners(img: Image.Image) -> int:
    """Estimate background gray level as the average of 4 corner pixels."""
    g = img.convert("L")
    w, h = g.size
    pts = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    vals = [g.getpixel(p) for p in pts]
    return int(sum(vals) / 4)

def _pad(img: Image.Image, px: int) -> Image.Image:
    """Uniform pad with estimated gray background."""
    if px <= 0: return img
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
    return max(pad_w, pad_h) + 6

def _pil_to_torch(img: Image.Image) -> torch.Tensor:
    """PIL (L) -> torch float tensor [1,1,H,W] in [0,1]."""
    g = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(g)[None, None, ...]  # [1,1,H,W]

def _torch_to_pil(t: torch.Tensor) -> Image.Image:
    """torch [1,1,H,W] -> PIL.Image L."""
    t = t.clamp(0, 1).squeeze(0).squeeze(0)
    arr = (t.cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Backward-compatible text size (Pillow 10+: textbbox)."""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    return draw.textsize(text, font=font)

# Advanced ops PIL->PIL

class RandomSafeRotate:
    """
    Rotation with background-aware padding; resizes back to original size.

    Args:
        degrees (float): Max absolute rotation angle.
        p (float): Probability to apply.
    """
    def __init__(self, degrees: float = 18.0, p: float = 0.7):
        self.degrees=degrees; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        angle = random.uniform(-self.degrees, self.degrees)
        pad_px = _pad_needed_for_rotation_shear(w, h, angle, 0.0)
        work = _pad(img, pad_px)
        out = TF.affine(work, angle=angle, translate=(0,0), scale=1.0, shear=[0.0,0.0],
                        interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work))
        return _resize_back(out, (w, h))

class RandomSafeShear:
    """
    Horizontal shear with safe padding and resize-back.

    Args:
        shear (float): Max absolute shear in degrees.
        p (float): Probability to apply.
    """
    def __init__(self, shear: float = 8.0, p: float = 0.5):
        self.shear=shear; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        sh = random.uniform(-self.shear, self.shear)
        pad_px = _pad_needed_for_rotation_shear(w, h, 0.0, sh)
        work = _pad(img, pad_px)
        out = TF.affine(work, angle=0.0, translate=(0,0), scale=1.0, shear=[sh,0.0],
                        interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work))
        return _resize_back(out, (w, h))

class RandomSafeTranslate:
    """
    Translation with padding to avoid cropping; resize back after shift.

    Args:
        translate (Tuple[float,float]): Max fractional shift (x,y) of width/height.
        p (float): Probability to apply.
    """
    def __init__(self, translate=(0.08,0.08), p: float=0.6):
        self.translate=translate; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        w, h = img.size
        tx = random.uniform(-self.translate[0]*w, self.translate[0]*w)
        ty = random.uniform(-self.translate[1]*h, self.translate[1]*h)
        pad_px = int(max(abs(tx), abs(ty))) + 6
        work = _pad(img, pad_px)
        out = TF.affine(work, angle=0.0, translate=(int(tx), int(ty)), scale=1.0, shear=[0.0,0.0],
                        interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work))
        return _resize_back(out, (w, h))

class ColorContrastJitter:
    """
    Brightness and contrast jitter for grayscale images.

    Args:
        brightness (float): Max +/- brightness factor delta.
        contrast (float): Max +/- contrast factor delta.
        p (float): Probability to apply.
    """
    def __init__(self, brightness=0.2, contrast=0.3, p: float = 0.7):
        self.brightness=brightness; self.contrast=contrast; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        if self.brightness>0:
            img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-self.brightness,self.brightness))
        if self.contrast>0:
            img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-self.contrast,self.contrast))
        return img

class RandomBrightness:
    """
    Brightness-only jitter (no contrast change).

    Args:
        max_delta (float): Max +/- brightness factor delta.
        p (float): Probability to apply.
    """
    def __init__(self, max_delta: float = 0.25, p: float = 0.5):
        self.max_delta=max_delta; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        factor = 1.0 + random.uniform(-self.max_delta, self.max_delta)
        return ImageEnhance.Brightness(img).enhance(factor)

class RandomShrinkIntoCanvas:
    """
    Scales image down and pastes onto a background of original size.

    Args:
        min_scale (float): Lower bound of uniform scale.
        max_scale (float): Upper bound of uniform scale.
        p (float): Probability to apply.
    """
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
    """
    Zooms in slightly without cropping text by padding first, resizing up,
    then resizing back to the original shape.

    Args:
        min_scale (float): Lower bound (>1) of zoom factor.
        max_scale (float): Upper bound of zoom factor.
        p (float): Probability to apply.
    """
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
    """
    Adds light/dark geometric outlines (rect/circle/triangle) as mild clutter.

    Args:
        n_blobs (Tuple[int,int]): How many blobs to draw.
        p (float): Probability to apply.
    """
    def __init__(self, n_blobs=(1,3), p: float = 0.5):
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
    """
    Adds many tiny grayscale dots (speckle) around the estimated background tone.

    Args:
        n_dots (Tuple[int,int]): Number of dots to draw.
        radius (Tuple[int,int]): Dot radius range.
        p (float): Probability to apply.
    """
    def __init__(self, n_dots=(80,180), radius=(0,1), p: float = 0.6):
        self.n_dots=n_dots; self.radius=radius; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L")
        draw = ImageDraw.Draw(img)
        w, h = img.size; bg = _bg_gray_from_corners(img)
        nd = random.randint(*self.n_dots)
        rmin, rmax = self.radius
        for _ in range(nd):
            r = random.randint(max(0, rmin), max(0, rmax))
            x = random.randint(0, w-1); y = random.randint(0, h-1)
            col = int(max(0, min(255, bg + random.randint(-70, 70))))
            if r <= 0: draw.point((x, y), fill=col)
            else: draw.ellipse((x-r, y-r, x+r, y+r), fill=col, outline=None)
        return img

class RandomElasticDistortion:
    """
    Elastic warp via random smoothed displacement fields and grid_sample.

    Args:
        p (float): Probability to apply.
        alpha (float): Displacement magnitude scale.
        sigma (float): Gaussian blur sigma for smoothing the noise.
    """
    def __init__(self, p: float=0.30, alpha: float=2.2, sigma: float=8.0):
        self.p=p; self.alpha=alpha; self.sigma=sigma
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        t = _pil_to_torch(img)  # [1,1,H,W]
        _, _, H, W = t.shape
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None]
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
    """
    Gentle sinusoidal spatial wobble in both axes.

    Args:
        p (float): Probability to apply.
        amp (float): Displacement amplitude in normalized coords.
        freq (float): Sinusoid frequency (cycles over image extent).
    """
    def __init__(self, p: float=0.30, amp: float=0.02, freq: float=1.4):
        self.p=p; self.amp=amp; self.freq=freq
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        t = _pil_to_torch(img); _, _, H, W = t.shape
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None]
        disp_x = self.amp * torch.sin(2*math.pi*self.freq*yy)
        disp_y = self.amp * torch.sin(2*math.pi*self.freq*xx)
        disp = torch.stack((disp_x, disp_y), dim=-1).unsqueeze(0)
        warped = F.grid_sample(t, grid + disp, mode="bilinear", padding_mode="border", align_corners=True)
        return _torch_to_pil(warped)

class AddNonASCIIChars:
    """
    Paste semi-transparent distractor glyphs (e.g., '*', '#', '?', '✓') over the image.

    Args:
        glyphs (str): Characters to sample from.
        n_chars (Tuple[int,int]): Number of glyphs to add.
        p (float): Probability to apply.
        size_frac (Tuple[float,float]): Glyph height as a fraction of image height.
    """
    def __init__(self, glyphs="*#?✓", n_chars=(1,3), p: float=0.6, size_frac=(0.45,0.70)):
        self.glyphs=list(glyphs); self.n_chars=n_chars; self.p=p; self.size_frac=size_frac
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        w, h = img.size
        base = img.convert("L")
        bg = _bg_gray_from_corners(img)
        for _ in range(random.randint(*self.n_chars)):
            g = random.choice(self.glyphs)
            fsize = max(12, int(random.uniform(*self.size_frac) * h))
            try: font = ImageFont.truetype(font=None, size=fsize)
            except: font = ImageFont.load_default()
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
            if tw2 >= w or th2 >= h: continue
            x = random.randint(0, w - tw2); y = random.randint(0, h - th2)
            tile_L, tile_A = tile.split()
            base.paste(tile_L, (x, y), mask=tile_A)
        return base

class AddSymbolDistractors:
    """
    Draw simple geometric symbol marks (*, #, ?, &, tick) without fonts.

    Args:
        symbols (List[str]|None): Allowed symbol names to draw.
        n_symbols (Tuple[int,int]): How many to draw.
        p (float): Probability to apply.
    """
    def __init__(self, symbols=None, n_symbols=(2,5), p: float=0.5):
        self.symbol_set = symbols or ['*', 'tick', '?', '#', '&']
        self.count_range = n_symbols; self.p=p
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p: return image
        canvas = image.copy(); pen = ImageDraw.Draw(canvas)
        w, h = canvas.size; num = random.randint(*self.count_range)
        for _ in range(num):
            shape = random.choice(self.symbol_set)
            ox = random.randint(0, max(1, w - 30)); oy = random.randint(0, max(1, h - 30))
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
    Duplicate a random rectangular patch and paste it with a small random shift,
    to emulate adjacent-character overlap without labels.

    Args:
        n_overlaps (Tuple[int,int]): How many patches to duplicate.
        max_shift (Tuple[int,int]): Max shift in pixels (x,y).
        patch_size (Tuple[int,int]): Patch width/height range (px).
        p (float): Probability to apply.
    """
    def __init__(self, n_overlaps=(1,2), max_shift=(5,10), patch_size=(20,40), p: float=0.6):
        self.n_overlaps=n_overlaps; self.max_shift=max_shift; self.patch_size=patch_size; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        img = img.convert("L"); w, h = img.size; out = img.copy()
        for _ in range(random.randint(*self.n_overlaps)):
            pw = random.randint(*self.patch_size); ph = random.randint(*self.patch_size)
            x1 = random.randint(0, max(1, w - pw - 1)); y1 = random.randint(0, max(1, h - ph - 1))
            patch = img.crop((x1, y1, x1 + pw, y1 + ph))
            dx = random.randint(-self.max_shift[0], self.max_shift[0])
            dy = random.randint(-self.max_shift[1], self.max_shift[1])
            x2 = min(max(0, x1 + dx), w - pw); y2 = min(max(0, y1 + dy), h - ph)
            out.paste(patch, (x2, y2))
        return out

class RandomInvert:
    """
    Invert grayscale intensities (negative image) with probability p.

    Args:
        p (float): Probability to apply.
    """
    def __init__(self, p: float=0.1):
        self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        g = img.convert("L"); arr = 255 - np.array(g, dtype=np.uint8)
        return Image.fromarray(arr, mode="L")

class RandomPerspectiveSafe:
    """
    Mild perspective warp using padded image and jittered corner endpoints.

    Args:
        distortion_scale (float): Corner jitter strength (0..1, mild recommended).
        p (float): Probability to apply.
    """
    def __init__(self, distortion_scale: float=0.25, p: float=0.35):
        self.distortion_scale=distortion_scale; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        w, h = img.size; pad_px = max(6, int(max(w, h) * 0.08))
        work = _pad(img, pad_px); W, H = work.size
        dx, dy = self.distortion_scale * 0.25 * W, self.distortion_scale * 0.25 * H
        tl = (random.uniform(0, dx), random.uniform(0, dy))
        tr = (W - random.uniform(0, dx), random.uniform(0, dy))
        bl = (random.uniform(0, dx), H - random.uniform(0, dy))
        br = (W - random.uniform(0, dx), H - random.uniform(0, dy))
        src = [(0,0),(W,0),(0,H),(W,H)]; dst=[tl,tr,bl,br]
        out = TF.perspective(work, src, dst, interpolation=Image.BILINEAR, fill=_bg_gray_from_corners(work))
        return _resize_back(out, (w, h))


# Public builder

# ---- Aug toggles (defaults) ----
DEFAULT_AUG_FLAGS: Dict[str, bool] = {
    "rotate": True, "shear": True, "translate": True,
    "gaussian_noise": True, "dots_lines": True,
    "color_contrast": True, "brightness": True,
    "shrink": True, "enlarge": True,
    "blobs": True, "dots": True,
    "elastic": True, "sine": True,
    "glyphs": True, "symbol_distractors": True, "overlap_sim": True,
    "invert": True, "perspective": True,
    "eraser": True,
}

def _preset_for_strength(strength: str) -> Dict[str, bool]:
    s = str(strength).lower().strip()
    if s == "mild":
        # Only your original ops
        return {
            "rotate": True,   # small vanilla rotation/affine (below)
            "shear": False, "translate": False,
            "gaussian_noise": True, "dots_lines": True,
            "color_contrast": False, "brightness": False,
            "shrink": False, "enlarge": False,
            "blobs": False, "dots": False,
            "elastic": False, "sine": False,
            "glyphs": False, "symbol_distractors": False, "overlap_sim": False,
            "invert": False, "perspective": False,
            "eraser": True,
        }
    elif s == "strong":
        return {**DEFAULT_AUG_FLAGS}
    else:
        # "medium" default
        return {**DEFAULT_AUG_FLAGS}

def build_augment(
    img_h: int = 64,
    img_w: int = 256,
    p_geom: float = 0.15,
    p_noise: float = 0.10,
    gauss_std_px: Tuple[float, float] = (1.0, 2.0),
    # NEW (optional): you can ignore these and old calls still work
    strength: str = "mild",
    rotation_mode: str = "resize",
    aug_flags: Optional[Dict[str, bool]] = None,
) -> T.Compose:
    """
    Returns a PIL->PIL Compose. Defaults replicate the ORIGINAL behavior.
    - strength="mild": original ops only (geom + noise/dots + small eraser).
    - "medium"/"strong": adds richer augmentations. You can override with aug_flags.
    """
    flags = _preset_for_strength(strength)
    if aug_flags:
        flags.update(aug_flags)

    # === Geometric stage ===
    if strength.lower() == "mild":
        # Your original tiny rotation/affine combined under one RandomApply
        geom = T.RandomApply([
            T.RandomRotation(1.5, fill=255),
            T.RandomAffine(degrees=0, translate=(0.02, 0.02), shear=1.0, fill=255),
        ], p=p_geom)
    else:
        # Safer, padded transforms (each optional via flags)
        geom_ops = []
        if flags.get("rotate", True):
            geom_ops.append(RandomSafeRotate(degrees=18, p=0.7))
        if flags.get("shear", True):
            geom_ops.append(RandomSafeShear(shear=8, p=0.5))
        if flags.get("translate", True):
            geom_ops.append(RandomSafeTranslate(translate=(0.08, 0.08), p=0.6))
        # wrap: if none selected fall back to tiny affine
        if len(geom_ops) == 0:
            geom = T.RandomApply([
                T.RandomRotation(1.5, fill=255),
                T.RandomAffine(degrees=0, translate=(0.02, 0.02), shear=1.0, fill=255),
            ], p=p_geom)
        else:
            # apply a random subset order with overall probability p_geom
            geom = T.RandomApply([T.RandomOrder(geom_ops)], p=p_geom)

    # === Noise/clutter stage ===
    if strength.lower() == "mild":
        noise = T.RandomOrder([
            RandomGaussianNoise(p=p_noise, std_px=gauss_std_px),
            RandomDotsLines(p=p_noise, n_dots=(2, 6), n_lines=(0, 1)),
            RandomEraser(p=0.04, area=(0.005, 0.02)) if flags.get("eraser", True) else T.Lambda(lambda x: x),
        ])
        extra = []  # no extra ops
    else:
        stack = []
        if flags.get("gaussian_noise", True):
            stack.append(RandomGaussianNoise(p=max(0.15, p_noise), std_px=gauss_std_px))
        if flags.get("dots_lines", True):
            stack.append(RandomDotsLines(p=max(0.12, p_noise), n_dots=(3, 8), n_lines=(0, 2)))
        if flags.get("color_contrast", True):
            stack.append(ColorContrastJitter(brightness=0.2, contrast=0.3, p=0.7))
        if flags.get("brightness", True):
            stack.append(RandomBrightness(max_delta=0.25, p=0.5))
        if flags.get("eraser", True):
            stack.append(RandomEraser(p=0.05, area=(0.01, 0.03)))
        noise = T.RandomOrder(stack)

        # optional extras applied AFTER noise
        extra = []
        if flags.get("shrink", True):
            extra.append(RandomShrinkIntoCanvas(min_scale=0.82, max_scale=0.95, p=0.45))
        if flags.get("enlarge", True):
            extra.append(RandomEnlargeSafely(min_scale=1.05, max_scale=1.20, p=0.45))
        if flags.get("blobs", True):
            extra.append(AddBlobs(n_blobs=(1, 3), p=0.5))
        if flags.get("dots", True):
            extra.append(AddDots(n_dots=(80, 180), radius=(0, 1), p=0.6))
        if flags.get("elastic", True):
            extra.append(RandomElasticDistortion(p=0.30, alpha=2.2, sigma=8.0))
        if flags.get("sine", True):
            extra.append(RandomSinusoidalDistortion(p=0.30, amp=0.02, freq=1.4))
        if flags.get("perspective", True):
            extra.append(RandomPerspectiveSafe(distortion_scale=0.25, p=0.35))
        if flags.get("invert", True):
            extra.append(RandomInvert(p=0.10))
        # keep glyph/symbol/overlap near the end
        if flags.get("glyphs", True):
            extra.append(AddNonASCIIChars(glyphs="*#?✓", n_chars=(1, 3), p=0.6, size_frac=(0.45, 0.70)))
        if flags.get("symbol_distractors", True):
            extra.append(AddSymbolDistractors(p=0.5, n_symbols=(2, 5)))
        if flags.get("overlap_sim", True):
            extra.append(SimulateCharacterOverlap(
                n_overlaps=(1, 2), max_shift=(5, 10), patch_size=(20, 40), p=0.6
            ))

    # Final compose: geom -> noise -> (extras if any)
    ops = [geom, noise] + (extra if len(extra) else [])
    return T.Compose(ops)
