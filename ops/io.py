from typing import Union, Tuple, List
import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

# ─────────────── Image <-> PIL <-> NumPy (RGB/BGR) ───────────────
def np_rgb_to_pil(img_rgb: np.ndarray) -> Image.Image:
    """HxWx3 uint8 RGB -> PIL RGB"""
    return Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")

def np_bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """HxWx3 uint8 BGR -> PIL RGB"""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    """PIL -> HxWx3 uint8 RGB"""
    return np.asarray(img.convert("RGB"))

def pil_to_np_bgr(img: Image.Image) -> np.ndarray:
    """PIL -> HxWx3 uint8 BGR"""
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)

# ─────────────── Base64 PNG helpers ───────────────

def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_png_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def np_rgb_to_b64_png(img_rgb: np.ndarray) -> str:
    return pil_to_b64_png(np_rgb_to_pil(img_rgb))

def np_bgr_to_b64_png(img_bgr: np.ndarray) -> str:
    return pil_to_b64_png(np_bgr_to_pil(img_bgr))

def b64_png_to_np_rgb(b64_str: str) -> np.ndarray:
    return pil_to_np_rgb(b64_png_to_pil(b64_str))

def b64_png_to_np_bgr(b64_str: str) -> np.ndarray:
    return pil_to_np_bgr(b64_png_to_pil(b64_str))

def np_to_data_url_png(img_rgb_or_pil: Union[np.ndarray, Image.Image]) -> str:
    """Encode an HxWx3 uint8 RGB array or PIL.Image to a data URL PNG."""
    if isinstance(img_rgb_or_pil, Image.Image):
        pil = img_rgb_or_pil.convert("RGB")
    else:
        pil = np_rgb_to_pil(img_rgb_or_pil)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# ───────────────────────────── helpers: masks NPZ ───────────────────────

def load_indexed_masks_npz(npz_path: Union[str, Path]) -> np.ndarray:
    """
    Returns masks as (T, H, W) uint16/uint8: 0=bg, 1..N object indices.
    Accepts common layouts:
      - key 'masks': (T,H,W)
      - first 3D array found in the NPZ
    """
    data = np.load(str(npz_path))
    if 'masks' in data:
        arr = data['masks']
    else:
        arr = None
        for k in data.files:
            v = data[k]
            if isinstance(v, np.ndarray) and v.ndim == 3:
                arr = v
                break
        if arr is None:
            raise ValueError(f"No (T,H,W) array found in {npz_path}")
    arr = np.asarray(arr)
    if arr.dtype not in (np.uint8, np.uint16, np.int32, np.int16):
        arr = arr.astype(np.int32)
    return arr


def load_indexed_mask_png(png_path: Union[str, Path]) -> np.ndarray:
    """Load a single indexed mask PNG as (H,W) uint8 array."""
    img = Image.open(str(png_path))
    assert img.mode == "P", "Only indexed PNGs are supported"
    arr = np.asarray(img)
    return arr


def ids_in_mask(mask_indexed: np.ndarray) -> List[int]:
    """Return sorted positive mask IDs present in an indexed mask."""
    return sorted(int(i) for i in np.unique(mask_indexed) if int(i) > 0)

# ───────────────────────────── helpers: video ─────────────────────────────

def open_video_reader(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap

def read_video_meta(cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return count, width, height, float(fps)

def make_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create writer: {path}")
    return writer

