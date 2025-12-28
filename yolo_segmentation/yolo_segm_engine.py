# yolo_segm_engine.py
from __future__ import annotations
from typing import Dict, Tuple
import argparse
from os import path as osp
import glob

import cv2
import numpy as np
import matplotlib
from tqdm import tqdm
import torch
from ultralytics import YOLO


class YOLO11SegEngineConfig:
    def __init__(
        self,
        weights: str = "weights/yolo11l-seg.pt",
        device: str = "cuda",
        imgsz: int = 960,
        conf: float = 0.60,
        iou: float = 0.70,
        max_det: int = 300,
        mask_thresh: float = 0.70,
        overlay: bool = False,
        overlay_alpha: float = 0.50,
    ):
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.mask_thresh = mask_thresh
        self.overlay = overlay
        self.overlay_alpha = overlay_alpha


class YOLO11SegEngine:
    """
    Minimal engine:
      input:  (H, W, 3) uint8 frame
      output: (H, W) uint8 ID map (0=bg, 1..N)
    """
    def __init__(self, cfg: YOLO11SegEngineConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        # quick warmup
        _ = self.model.predict(
            np.zeros((32, 32, 3), dtype=np.uint8),
            imgsz=32, conf=cfg.conf, iou=cfg.iou,
            device=cfg.device, verbose=False
        )

    @torch.inference_mode()
    def process_single_frame(self, frame: np.ndarray, channel_order: str = "BGR") -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        assert channel_order in ("BGR", "RGB")
        assert frame.ndim == 3 and frame.shape[2] == 3

        if channel_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        H, W = frame.shape[:2]

        res = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            max_det=self.cfg.max_det,
            retina_masks=True,   # masks at original resolution
            verbose=False
        )[0]

        # ----- resolve class name map (id -> name) -----
        names = getattr(self.model, "names", None)
        if names is None and hasattr(self.model, "model"):
            names = getattr(self.model.model, "names", None)

        # Build a robust class-id -> class-name mapping
        class_id_to_name: Dict[int, str] = {}
        if isinstance(names, dict):
            class_id_to_name = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            class_id_to_name = {i: str(n) for i, n in enumerate(names)}
        else:
            class_id_to_name = {}

        zeros = np.zeros((H, W), dtype=np.uint8)
        out: Dict[str, np.ndarray] = {}
        classname_to_idx: Dict[str, int] = {}
        for cid, cname in class_id_to_name.items():
            out[cname] = zeros.copy()
            classname_to_idx[cname] = int(cid)

        # If masks are missing and we *do* know the classes, return all-zero maps
        if res.masks is None or res.masks.data is None:
            if not class_id_to_name:
                return {}, {}
            return out, classname_to_idx

        # ----- gather raw predictions -----
        masks_f = res.masks.data.detach().cpu().numpy()
        boxes = res.boxes

        if boxes is not None and boxes.conf is not None:
            confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        else:
            confs = np.ones((masks_f.shape[0],), dtype=np.float32)

        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_idx = boxes.cls.detach().cpu().numpy().astype(np.int32)
        else:
            cls_idx = np.zeros((masks_f.shape[0],), dtype=np.int32)

        def _cls_to_name(c: int) -> str:
            if c in class_id_to_name:
                return class_id_to_name[c]
            # If unseen id, create a synthetic name and register it
            cname = f"class_{int(c)}"
            class_id_to_name[c] = cname
            if cname not in out:
                out[cname] = zeros.copy()
                classname_to_idx[cname] = int(c)
            return cname

        # ----- filter by confidence -----
        keep = confs >= float(self.cfg.conf)
        if keep.sum() == 0:
            return out, classname_to_idx

        masks_f = masks_f[keep]
        confs = confs[keep]
        cls_idx = cls_idx[keep]

        # binarize like Ultralytics plot()
        masks_b = masks_f > float(self.cfg.mask_thresh)   # (M, H, W) bool
        if masks_b.shape[0] == 0:
            return out, classname_to_idx

        # ----- per-class idmaps -----
        for c in np.unique(cls_idx):
            sel = (cls_idx == c)
            if not np.any(sel):
                continue

            masks_c = masks_b[sel]           # (K, H, W)
            confs_c = confs[sel]             # (K,)
            if masks_c.shape[0] == 0:
                continue

            # cap per class to 254 instances (uint8 safety)
            if masks_c.shape[0] > 254:
                top = np.argsort(-confs_c)[:254]
                masks_c = masks_c[top]
                confs_c = confs_c[top]

            # per-pixel best within this class
            scores = np.where(masks_c, confs_c[:, None, None], -1.0)  # (K, H, W)
            bestconf = scores.max(axis=0)
            bestidx = scores.argmax(axis=0)

            idmap_c = np.where(bestconf < 0, 0, bestidx + 1).astype(np.uint8)  # 0=bg, 1..K

            # ensure shape HxW (in case model returned a different size)
            if idmap_c.shape != (H, W):
                idmap_c = cv2.resize(idmap_c, (W, H), interpolation=cv2.INTER_NEAREST)

            cls_name = _cls_to_name(int(c))
            out[cls_name] = idmap_c
            classname_to_idx[cls_name] = int(c)

        return out, classname_to_idx


def _colorize_idmap(idmap: np.ndarray) -> np.ndarray:
    # pick a categorical colormap
    cmap = matplotlib.colormaps["tab20"]   # 20 distinct-ish colors

    # make sure we have integer ids
    idmap = idmap.astype(np.int32)

    # build a color table big enough for your max id
    max_id = int(idmap.max())
    colors = np.zeros((max_id + 1, 3), dtype=np.uint8)

    for i in range(max_id + 1):
        # cmap returns RGBA in 0..1
        r, g, b, _ = cmap(i % cmap.N)   # wrap around if > 20
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]

    # map ids -> RGB by indexing
    rgb = colors[idmap]  # shape (H, W, 3), RGB

    # convert to BGR for OpenCV
    bgr = rgb[..., ::-1]

    return bgr



def _overlay_from_idmap(frame_bgr: np.ndarray, idmap: np.ndarray, alpha: float) -> np.ndarray:
    """Create an overlay by blending the colorized idmap onto the original frame only where idmap>0."""
    color = _colorize_idmap(idmap)
    overlay = frame_bgr.copy()
    mask = idmap > 0
    if mask.any():
        # blend only on mask regions
        overlay[mask] = cv2.addWeighted(frame_bgr[mask], 1.0 - alpha, color[mask], alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video or '0' for webcam")
    parser.add_argument("--weights", default="runs/exp_robotarm/weights/best.pt")
    parser.add_argument("--save", default="", help="Optional: path to save output video")
    parser.add_argument("--show", action="store_true", help="Show overlay on screen")
    parser.add_argument("--overlay", action="store_true", help="Write [IDMAP | OVERLAY] side-by-side")
    parser.add_argument("--alpha", type=float, default=0.50, help="Overlay alpha (0..1)")
    args = parser.parse_args()

    cfg = YOLO11SegEngineConfig(weights=args.weights, overlay=args.overlay, overlay_alpha=args.alpha)
    engine = YOLO11SegEngine(cfg)

    if osp.isdir(args.video):
        cap = sorted(glob.glob(osp.join(args.video, "*.jpg")))
        print(args.video)
        fps = 30
        H, W = cv2.imread(cap[0]).shape[:2]
        n_frames = len(cap)
    else:
        cap = cv2.VideoCapture(0 if args.video.isdigit() else args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = (W * 2, H) if cfg.overlay else (W, H)
        writer = cv2.VideoWriter(args.save, fourcc, fps, out_size)

    idx = 0
    for _ in tqdm(range(n_frames)):
        if isinstance(cap, list):
            if idx >= len(cap):
                break
            frame = cv2.imread(cap[idx])
        else:
            ret, frame = cap.read()
            if not ret:
                break
        idx += 1

        idmaps, _ = engine.process_single_frame(frame, channel_order="BGR")
        obj_idmap = idmaps["object"]
        arm_idmap = idmaps["robot_arm"]

        obj_idmap_shifted = obj_idmap.copy()
        obj_idmap_shifted[obj_idmap > 0] += 1
        idmap = obj_idmap_shifted.copy()
        idmap[arm_idmap == 1] = 1

        # Left: colorized ID map
        id_vis = _colorize_idmap(idmap)

        if cfg.overlay:
            # Right: overlay on original
            over = _overlay_from_idmap(frame, idmap, cfg.overlay_alpha)
            vis = np.concatenate([id_vis, over], axis=1)
        else:
            vis = id_vis

        if writer:
            writer.write(vis)
        if args.show:
            cv2.imshow("YOLO Segmentation", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    if not isinstance(cap, list):
        cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
