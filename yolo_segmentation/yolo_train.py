#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from copy import deepcopy

import yaml
from ultralytics import YOLO


def load_cfg(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() in (".yml", ".yaml"):
        return yaml.safe_load(p.read_text())
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported cfg type: {p.suffix}")


def _resolve_save_dir(model, train_ret, overrides) -> Path:
    """
    Robustly get the run directory across Ultralytics versions.
    Priority:
      1) model.trainer.save_dir (newer versions)
      2) train_ret.save_dir (older versions)
      3) SETTINGS['runs_dir'] + project/name (fallback)
    """
    # 1) Preferred: trainer.save_dir
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None):
        return Path(trainer.save_dir)

    # 2) Older return object
    if train_ret is not None and hasattr(train_ret, "save_dir"):
        return Path(train_ret.save_dir)

    # 3) Fallback: compose from settings + overrides
    try:
        from ultralytics.utils import SETTINGS  # global YOLO settings
        runs_root = Path(SETTINGS.get("runs_dir", "runs"))
    except Exception:
        runs_root = Path("runs")

    project = overrides.get("project", None)
    name = overrides.get("name", "train")

    # If you used project='.' (flat under runs_dir), don't add a subfolder
    base = runs_root if (project in (None, "", ".")) else runs_root / project
    return base / name


def train_once(model_path: str, overrides: dict) -> Path:
    from ultralytics import YOLO
    model = YOLO(model_path)
    ret = model.train(**overrides)
    save_dir = _resolve_save_dir(model, ret, overrides)
    print(f"[info] results saved to: {save_dir}")
    return save_dir


def derive_or(base: dict, key: str, fallback):
    """Use base[key] if present else fallback."""
    return base.get(key, fallback)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="YAML/JSON base config (Ultralytics overrides style)")
    # optional quick overrides (kept minimal)
    ap.add_argument("--name", type=str, default=None, help="Run name (overrides cfg.name)")
    ap.add_argument("--device", type=str, default=None, help='e.g. "0" or "0,1" (overrides cfg.device)')
    ap.add_argument("--model", type=str, default=None, help="Override cfg.model (e.g., yolo11l-seg.pt)")

    # phase 1 (frozen heads)
    ap.add_argument("--epochs1", type=int, default=None, help="Phase-1 epochs (default cfg.epochs or 12)")
    ap.add_argument("--freeze1", type=int, default=10, help="Frozen layers in Phase-1 (e.g., 10)")

    # phase 2 (unfreeze / partial)
    ap.add_argument("--epochs2", type=int, default=15, help="Phase-2 epochs; 0 to skip")
    ap.add_argument("--freeze2", type=int, default=0, help="Frozen layers in Phase-2 (0 = full unfreeze)")

    # minor deltas for Phase-2 (if not set, derive from Phase-1)
    ap.add_argument("--lr0_2", type=float, default=None, help="Phase-2 lr0 (default: 0.5 * Phase-1 lr0)")
    ap.add_argument("--lrf_2", type=float, default=None, help="Phase-2 lrf (default: 0.5 * Phase-1 lrf, min 0.02)")
    ap.add_argument("--mixup_2", type=float, default=None, help="Phase-2 mixup (default: 0.67 * Phase-1 mixup)")
    ap.add_argument("--copy_paste_2", type=float, default=None, help="Phase-2 copy_paste (default: 0.67 * Phase-1)")
    ap.add_argument("--scale_2", type=float, default=None, help="Phase-2 scale (default: 0.8 * Phase-1 scale)")

    args = ap.parse_args()
    base = load_cfg(args.cfg)

    # gentle helpers
    def _val(k, dflt):  # read from base with fallback
        return derive_or(base, k, dflt)

    # optional user overrides
    if args.name is not None:
        base["name"] = args.name
    if args.device is not None:
        base["device"] = args.device
    if args.model is not None:
        base["model"] = args.model

    # ---------------- Phase 1 ----------------
    p1 = deepcopy(base)
    p1["task"] = _val("task", "segment")
    p1["freeze"] = args.freeze1
    p1["epochs"] = args.epochs1 if args.epochs1 is not None else _val("epochs", 12)
    # keep name from cfg or override via --name
    p1["name"] = p1.get("name", "exp1")

    print(f"\n[Phase 1] freeze={p1['freeze']}, epochs={p1['epochs']}, name={p1['name']}")
    phase1_dir = train_once(model_path=p1.get("model", "yolo11s-seg.pt"), overrides=p1)

    # obtain checkpoint
    wdir = phase1_dir / "weights"
    ckpt = wdir / "last.pt"
    if not ckpt.exists():
        ckpt = wdir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found in {wdir}")

    # optionally skip phase 2
    if args.epochs2 <= 0:
        print("\n[Done] Phase-2 skipped (epochs2 <= 0)")
        print("[Artifacts] Phase-1:", phase1_dir)
        return

    # ---------------- Phase 2 ----------------
    p2 = deepcopy(base)
    p2["task"] = _val("task", "segment")
    p2["freeze"] = args.freeze2
    p2["epochs"] = args.epochs2
    # compute minor deltas for phase-2 if not provided
    lr0_1 = float(_val("lr0", 0.002))
    lrf_1 = float(_val("lrf", 0.10))
    mixup_1 = float(_val("mixup", 0.15))
    cp_1 = float(_val("copy_paste", 0.30))
    scale_1 = float(_val("scale", 0.15))

    p2["lr0"] = args.lr0_2 if args.lr0_2 is not None else max(lr0_1 * 0.5, 1e-4)
    p2["lrf"] = args.lrf_2 if args.lrf_2 is not None else max(lrf_1 * 0.5, 0.02)
    p2["mixup"] = args.mixup_2 if args.mixup_2 is not None else max(mixup_1 * 0.67, 0.0)
    p2["copy_paste"] = args.copy_paste_2 if args.copy_paste_2 is not None else max(cp_1 * 0.67, 0.0)
    p2["scale"] = args.scale_2 if args.scale_2 is not None else max(scale_1 * 0.8, 0.05)

    # suffix name so runs don't collide
    p2["name"] = f"{p1['name']}_unfreeze"

    print(f"\n[Phase 2] freeze={p2['freeze']}, epochs={p2['epochs']}, name={p2['name']}")
    phase2_dir = train_once(model_path=str(ckpt), overrides=p2)

    # quick val on best of phase 2 (optional)
    try:
        best2 = phase2_dir / "weights" / "best.pt"
        if best2.exists():
            YOLO(str(best2)).val(
                task=p2["task"],
                data=p2.get("data", base["data"]),
                imgsz=p2.get("imgsz", base.get("imgsz", 960)),
                batch=p2.get("batch", base.get("batch", 8)),
                device=str(p2.get("device", base.get("device", "0"))),
            )
    except Exception as e:
        print(f"[warn] val skipped: {e}")

    print("\n[Artifacts] Phase-1:", phase1_dir)
    print("[Artifacts] Phase-2:", phase2_dir)


if __name__ == "__main__":
    main()
