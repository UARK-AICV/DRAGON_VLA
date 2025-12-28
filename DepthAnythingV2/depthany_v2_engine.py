from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal, Optional
import argparse

import cv2
import numpy as np
import torch
import matplotlib
from tqdm import tqdm

from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

EncoderName = Literal["vits", "vitb", "vitl", "vitg"]
NormalizeMode = Literal["scale", "minmax", "none"]

@dataclass
class DepthAnyV2EngineConfig:
    # Model config
    encoder: EncoderName = "vitl"
    weights: str = ""  # leave empty to auto-resolve from encoder

    # Runtime
    device: str = "auto"       # "auto" | "cuda" | "mps" | "cpu"
    input_size: int = 518
    use_fp16: bool = False     # CUDA autocast

    # Output control
    normalize: NormalizeMode = "scale"   # "scale" (0..scale_meters) | "minmax" | "none"
    scale_meters: float = 2.5
    colorize: bool = True                # if True, return BGR visualization
    grayscale: bool = False              # if True (with colorize), 3-ch gray instead of colormap
    return_uint8: bool = True            # when colorize=True, emit uint8

    # Colormap
    palette: str = "Spectral_r"          # matplotlib palette name; leave "" to use OpenCV inferno

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolved_weights(self) -> str:
        if not self.weights:
            return f"DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_{self.encoder}.pth"
        # Template
        if "{encoder}" in self.weights:
            return self.weights.format(encoder=self.encoder)
        # Directory
        if os.path.isdir(self.weights):
            return os.path.join(
                self.weights,
                f"depth_anything_v2_metric_hypersim_{self.encoder}.pth",
            )
        # File path
        return self.weights


class DepthAnyV2Engine:
    _MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, cfg: Optional[DepthAnyV2EngineConfig] = None):
        self.cfg = cfg or DepthAnyV2EngineConfig()
        self.device = self.cfg.resolved_device()

        # Build model
        encoder_cfg = self._MODEL_CONFIGS[self.cfg.encoder]
        self.model = DepthAnythingV2(**encoder_cfg)

        # Load weights (encoder-aware)
        weights_path = self.cfg.resolved_weights()
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DepthAnythingV2 weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # Precision
        self._amp_enabled = bool(self.cfg.use_fp16 and self.device == "cuda")

        # Colormap
        self._mpl_cmap = None
        if self.cfg.palette:
            self._mpl_cmap = matplotlib.colormaps.get_cmap(self.cfg.palette)

    # ─────────── Single-frame API (for pipeline) ───────────
    @torch.inference_mode()
    def run_single_frame(
        self,
        rgb_np: np.ndarray,
        *,
        return_viz: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Args:
            rgb_np: (H, W, 3) uint8 RGB image.
            return_viz:
              - True  → returns (H,W,3) BGR visualization (uint8 if cfg.return_uint8).
              - False → returns raw float32 meters if normalize='none', else float32 in [0,1].
              - None  → uses cfg.colorize.

        Returns:
            np.ndarray as described above.
        """
        if rgb_np is None or rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
            raise ValueError("run_single_frame expects a RGB image of shape (H, W, 3).")
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        # DepthAnythingV2.infer_image in your repo accepts BGR np.uint8 directly.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp_enabled):
            depth_m = self.model.infer_image(bgr_np, self.cfg.input_size)  # (H,W) float32 meters

        if return_viz is None:
            return_viz = self.cfg.colorize

        if not return_viz:
            return self._maybe_normalize(depth_m)

        return self._visualize(depth_m)

    # ─────────── Helpers ───────────
    def _maybe_normalize(self, depth_m: np.ndarray) -> np.ndarray:
        mode = self.cfg.normalize
        if mode == "none":
            return depth_m.astype(np.float32, copy=False)

        if mode == "minmax":
            dmin = np.nanmin(depth_m)
            dmax = np.nanmax(depth_m)
            if not np.isfinite(dmin) or not np.isfinite(dmax) or (dmax - dmin) <= 1e-12:
                return np.zeros_like(depth_m, dtype=np.float32)
            norm = (depth_m - dmin) / (dmax - dmin)
            return np.clip(norm, 0.0, 1.0).astype(np.float32)

        if mode == "scale":
            s = max(float(self.cfg.scale_meters), 1e-6)
            norm = np.clip(depth_m / s, 0.0, 1.0)
            return norm.astype(np.float32)
        raise ValueError(f"Invalid normalize mode: {mode}")

    def _visualize(self, depth_m: np.ndarray) -> np.ndarray:
        # Normalize to [0,1] for visualization input
        depth01 = (
            self._maybe_normalize(depth_m)
            if self.cfg.normalize != "none"
            else np.clip(depth_m / max(self.cfg.scale_meters, 1e-6), 0.0, 1.0).astype(np.float32)
        )
        depth255 = (depth01 * 255.0).astype(np.float32)

        if self.cfg.grayscale:
            viz = np.repeat(depth255[..., None], 3, axis=-1)  # pretend RGB
        else:
            rgba = self._mpl_cmap(depth255.astype(np.uint8))  # (H,W,4) in [0,1]
            viz = (rgba[..., :3] * 255.0).astype(np.float32)  # RGB

        return viz.astype(np.uint8) if self.cfg.return_uint8 else viz.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Test DepthAnyV2Engine on a video")
    p.add_argument("--video", required=True, type=str, help="Input video path")
    p.add_argument("--out", type=str, default="", help="Output mp4 path (auto if empty)")
    p.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument("--weights", type=str, default="", help='Weights path/dir/template (supports "{encoder}")')
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--input-size", type=int, default=518, help="Model input resolution")
    p.add_argument("--normalize", type=str, default="scale", choices=["scale", "minmax", "none"])
    p.add_argument("--scale-meters", type=float, default=2.5, help="Clamp range for normalize=scale")
    p.add_argument("--grayscale", action="store_true", help="Use 3-ch grayscale viz instead of colormap")
    p.add_argument("--palette", type=str, default="Spectral_r", help="Matplotlib palette ('' for OpenCV inferno)")
    p.add_argument("--fp16", action="store_true", help="Enable CUDA autocast FP16")
    p.add_argument("--pred-only", action="store_true", help="Write prediction only (no side-by-side)")
    p.add_argument("--margin", type=int, default=50, help="Margin width for side-by-side mode")
    p.add_argument("--limit-frames", type=int, default=0, help="Optional cap on processed frames (0=all)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    # Derive output path if not specified
    if not args.out:
        base = os.path.splitext(os.path.basename(args.video))[0]
        suffix = "_pred" if args.pred_only else "_sxs"
        args.out = os.path.join(os.path.dirname(args.video), f"{base}{suffix}.mp4")

    # Build engine
    cfg = DepthAnyV2EngineConfig(
        encoder=args.encoder,
        weights=args.weights,          # auto-resolves if "" or a dir/template
        device=args.device,
        input_size=args.input_size,
        use_fp16=args.fp16,
        normalize=args.normalize,
        scale_meters=args.scale_meters,
        colorize=True,                 # we want a visual for video output
        grayscale=args.grayscale,
        palette=args.palette,
    )
    engine = DepthAnyV2Engine(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # handle 0 or NaN
        fps = 30.0

    out_w = w if args.pred_only else (w * 2 + args.margin)
    out_h = h

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {args.out}")

    frame_idx = 0
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), desc="DepthAnyV2")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Run engine → BGR visualization (uint8)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            viz_rgb = engine.run_single_frame(frame_rgb, return_viz=True)
            viz_bgr = cv2.cvtColor(viz_rgb, cv2.COLOR_RGB2BGR)

            if args.pred_only:
                writer.write(viz_bgr)
            else:
                # Side-by-side: original | margin | prediction
                if viz_bgr.shape[:2] != frame_bgr.shape[:2]:
                    # (Should match, but be robust)
                    viz_bgr = cv2.resize(viz_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
                split = np.ones((h, args.margin, 3), dtype=np.uint8) * 255
                combo = cv2.hconcat([frame_bgr, split, viz_bgr])
                writer.write(combo)
            frame_idx += 1
            pbar.update(1)
            if args.limit_frames and frame_idx >= args.limit_frames:
                break
    finally:
        cap.release()
        writer.release()

    print(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()

