from typing import Union, Tuple, List, Dict

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from .misc import ensure_rgb_array

# ───────────────────────────── helpers: colors & overlay ─────────────────────────

def _build_shared_palette(n_obj: int):
    """Return RGB tuples in [0,255] with good separation (Tab10/20 then HSV)."""
    if n_obj <= 10:
        base = plt.colormaps.get_cmap('tab10')
        cols = [tuple(int(c*255) for c in base(i)[:3]) for i in range(n_obj)]
    elif n_obj <= 20:
        base = plt.colormaps.get_cmap('tab20')
        cols = [tuple(int(c*255) for c in base(i)[:3]) for i in range(n_obj)]
    else:
        hues = np.linspace(0, 1, n_obj, endpoint=False)
        hsv = np.column_stack([hues, np.ones_like(hues), np.ones_like(hues)])
        rgb = (hsv_to_rgb(hsv) * 255).astype(np.int32)
        cols = [tuple(map(int, row)) for row in rgb]
    return cols

def _outline_mask(mask_bool: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Internal outline ring (so neighbors don't fight for boundary pixels)."""
    if thickness < 1:
        thickness = 1
    u8 = mask_bool.astype(np.uint8)
    inner = cv2.erode(u8, None, iterations=thickness)
    ring  = cv2.subtract(u8, inner)
    return (ring * 255).astype(np.uint8)

def _find_mask_centre(mask_bool: np.ndarray) -> Tuple[int, int]:
    mb = mask_bool.astype(bool)
    if not mb.any():
        return (0, 0)
    padded = np.pad(mb, pad_width=1, mode='constant', constant_values=0)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    dist *= mb
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)

def overlay_with_indexed_mask(
    frame_rgb: Union[np.ndarray, Image.Image],
    indexed_mask: np.ndarray,                 # (H,W) 0=bg, 1..N
    edge_px: int = 3,
    font_px: int = 16,        # fixed label size in *pixels*
    dpi: int = 100,           # canvas DPI; fontsize points = font_px * 72 / dpi
    text_box_alpha: float = 0.85,
) -> Image.Image:
    """
    Colored fill + inside outline (NumPy/OpenCV), crisp labels (matplotlib).
    Returns a PIL RGB image.
    """
    out = ensure_rgb_array(frame_rgb)

    h, w = out.shape[:2]
    if indexed_mask.shape != (h, w):
        indexed_mask = cv2.resize(indexed_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    num_objs = int(indexed_mask.max())
    if num_objs == 0:
        return Image.fromarray(out)

    colors = _build_shared_palette(num_objs)

    # --------- Inside outlines ----------
    for idx in range(1, num_objs + 1):
        mask = (indexed_mask == idx)
        if not mask.any():
            continue
        edges = _outline_mask(mask, thickness=edge_px)
        color = colors[(idx - 1) % len(colors)]
        out[edges > 0] = color

    # --------- Add crisp text with matplotlib ----------
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(out)

    fontsize_pt = (font_px * 72.0) / dpi  # px → pt
    for idx in range(1, num_objs + 1):
        mask = (indexed_mask == idx)
        if not mask.any():
            continue
        cX, cY = _find_mask_centre(mask)
        colour = tuple(c / 255.0 for c in colors[(idx - 1) % len(colors)])
        ax.text(
            cX, cY, str(idx),
            color=colour, fontsize=fontsize_pt, weight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='black', ec='none', alpha=text_box_alpha)
        )

    # Grab canvas -> numpy -> PIL
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 4)[..., 1:]
    plt.close(fig)

    return Image.fromarray(buf)


def highlight_with_indexed_masks(
    frame_rgb: Union[np.ndarray, Image.Image],
    indexed_mask: np.ndarray,                         # (H,W) ints (0 bg, 1..N)
    *,
    blur_ksize: int = 51,
    edge_px: int = 3,
    dim_factor: float = 0.2,
    outline_color: Tuple[int, int, int] = (230, 230, 230),  # WHITE
) -> Image.Image:
    """
    Blur+dim everything NOT selected; keep selected regions original; draw WHITE inside outlines.
    """
    # normalize inputs
    return_pil = isinstance(frame_rgb, Image.Image)
    if return_pil:
        img = np.array(frame_rgb.convert("RGB"))
    else:
        img = np.asarray(frame_rgb).copy()
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
    img = img.astype(np.uint8)

    h, w = img.shape[:2]
    if indexed_mask.shape != (h, w):
        indexed_mask = cv2.resize(indexed_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # union of all selected ids
    union = indexed_mask.astype(bool)

    # blur + dim background
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    dimmed_blur = (blurred.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)

    blended = img.copy()
    blended[~union] = dimmed_blur[~union]

    # WHITE outlines for each object group
    unique_ids = np.unique(indexed_mask)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        obj_mask = indexed_mask == obj_id
        edges = _outline_mask(obj_mask, thickness=edge_px)
        blended[edges > 0] = outline_color
    return Image.fromarray(blended)


_COLORMAP = plt.colormaps.get_cmap("Set1")

def _id_to_tab10_color(class_id: int) -> Tuple[int, int, int]:
    """Map object ID to a distinct RGB color using matplotlib's tab10."""
    # tab10 has 10 discrete colors (0..9)
    rgb = _COLORMAP(class_id % 10)[:3]   # normalize to [0,1]
    return tuple(int(255 * c) for c in rgb)

def highlight_masks_black_background(
    frame_rgb: Union[np.ndarray, Image.Image],
    indexed_masks: Dict[str, np.ndarray],  # (H, W) ints (0 bg, 1..N)
    class2id: Dict[str, int],
    edge_px: int = 3
) -> np.ndarray:
    """
    Keep selected regions as-is; set everything else to black.
    Draw per-class colored *inside* outlines for each instance (IDs 1..K) using tab10.
    """
    # normalize input image to uint8 RGB np.ndarray
    return_pil = isinstance(frame_rgb, Image.Image)
    if return_pil:
        img = np.array(frame_rgb.convert("RGB"))
    else:
        img = np.asarray(frame_rgb).copy()
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.uint8)

    H, W = img.shape[:2]

    # union over all classes (any id > 0)
    union = np.zeros((H, W), dtype=bool)
    # resize/align class idmaps if needed and cache them
    aligned_idmaps: Dict[str, np.ndarray] = {}
    for cls_name, idmap in indexed_masks.items():
        if idmap is None:
            continue
        m = np.asarray(idmap)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        aligned_idmaps[cls_name] = m
        union |= (m.astype(bool))

    # start with black background, copy original on union
    out = np.zeros_like(img)
    out[union] = img[union]

    # draw inside outlines for each class with its class color
    for cls_name, idmap in aligned_idmaps.items():
        class_color = _id_to_tab10_color(class2id[cls_name])
        unique_ids = np.unique(idmap)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            obj_mask = (idmap == obj_id)
            if not obj_mask.any():
                continue
            edges = _outline_mask(obj_mask, thickness=max(1, int(edge_px)))
            out[edges > 0] = class_color

    return out
