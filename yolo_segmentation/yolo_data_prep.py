from __future__ import annotations
import argparse, os, sys, glob, shutil, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image
import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ───────────────────────────── dataclasses ─────────────────────────────
@dataclass
class SourceCfg:
    name: str
    image_glob: str
    mask_dirname: str
    mask_ext: str = ".png"
    klass: str = "object"  # class name (maps via class_map)
    instance_policy: str = "largest"  # 'largest'|'hull'|'skip'|'keep_all'
    priority: int = 0  # higher wins in conflicts

# ─────────────────────────── helpers: paths/keys ───────────────────────────

def get_video_dir(img_path: Path) -> Path:
    """Return the <video> directory given an image under .../<video>/images/<file>."""
    return img_path.parent.parent


def infer_mask_path(img_path: Path, mask_dirname: str, mask_ext: str) -> Path:
    """Single-mask case: .../<video>/images/<stem>.jpg → .../<video>/<mask_dirname>/<stem>.<ext>"""
    if img_path.parent.name == mask_dirname:
        return img_path.with_suffix(mask_ext)
    return img_path.parent.parent / mask_dirname / (img_path.stem + mask_ext)


def frame_key_for(img_path: Path) -> Tuple[str, str]:
    """Key used to unify frames across sources: (video_dir_name, frame_stem).
    This is robust if different roots have the same <video> folder naming.
    """
    return (get_video_dir(img_path).name, img_path.stem)


def infer_view_from_path(img_path: Path) -> str:
    """
    Infer camera view from path.
    Default heuristic: filename or parent dirs contain 'wrist' or 'base'.
    Adjust this to match your dataset convention.
    """
    s = str(img_path).lower()
    if "wrist" in s:
        return "wrist"
    if "base" in s:
        return "base"
    # Fallback: treat as base if nothing matches
    return "base"


def load_backgrounds(bg_dir: Optional[Path]) -> List[np.ndarray]:
    """
    Load background images from a directory. Returns a list of BGR images.
    Only the first 4 are used (for versions 2–5).
    """
    if bg_dir is None:
        return []

    paths = sorted(
        p for p in bg_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    bgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            bgs.append(img)
    if len(bgs) < 4:
        print(f"[warn] Only {len(bgs)} backgrounds found in {bg_dir}, "
              f"expected at least 4.", file=sys.stderr)
    return bgs[:4]


# ─────────────────────── helpers: contours & polygons ───────────────────────

def find_external_contours(mask: np.ndarray,
                           min_area: float = 10.0,
                           use_chain_none: bool = True,
                           approx_eps_frac: float = 0.002) -> List[np.ndarray]:
    mode = cv2.RETR_EXTERNAL
    chain = cv2.CHAIN_APPROX_NONE if use_chain_none else cv2.CHAIN_APPROX_SIMPLE
    cnts, _ = cv2.findContours(mask, mode, chain)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        if use_chain_none:
            c2 = c
        else:
            eps = approx_eps_frac * cv2.arcLength(c, True)
            c2 = cv2.approxPolyDP(c, eps, True)
        if len(c2) >= 3:
            out.append(c2)
    return out


def contour_to_yolo_poly(contour: np.ndarray, w: int, h: int) -> List[float]:
    pts = contour.squeeze(1).astype(np.float32)
    pts[:, 0] /= max(1, w)
    pts[:, 1] /= max(1, h)
    return pts.reshape(-1).tolist()


def make_bg_variants_for_frame(
    im: np.ndarray,
    occupied: np.ndarray,
    bg_list: List[np.ndarray],
    stem: str,
    img_suffix: str,
    img_split_dir: Path,
    lbl_split_dir: Path,
    lbl_dst: Path,
    valid_lines: int,
    stats_split: Dict[str, int],
):
    """
    Create background-augmented variants for a single frame.

    - im: original image (H, W, 3) BGR
    - occupied: uint8 mask (H, W) with 255 on foreground (objects)
    - bg_list: list of bg images (BGR), we use up to 4 → versions 2–5
    - stem: base filename stem (without extension)
    - img_suffix: original image extension (e.g., '.jpg')
    - img_split_dir, lbl_split_dir: target split dirs
    - lbl_dst: path to original label file (already written)
    - valid_lines: number of instances in label file
    - stats_split: stats[split] dict to update
    """
    if valid_lines == 0:
        return
    if not bg_list:
        return

    h, w = im.shape[:2]
    fg_mask = (occupied > 0).astype(np.uint8)
    fg_mask_3 = fg_mask[..., None]  # (H,W,1)
    bg_mask_3 = 1 - fg_mask_3

    # Use up to 4 backgrounds → versions 2,3,4,5
    for idx, bg in enumerate(bg_list[:4], start=2):
        # Resize background to match image
        bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)

        # Blend: keep foreground from original, background from bg_resized
        blended = (im.astype(np.float32) * fg_mask_3 +
                   bg_resized.astype(np.float32) * bg_mask_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        img_variant_name = f"{stem}_bg{idx}{img_suffix}"
        lbl_variant_name = f"{stem}_bg{idx}.txt"

        img_variant_path = img_split_dir / img_variant_name
        lbl_variant_path = lbl_split_dir / lbl_variant_name

        # Write augmented image
        cv2.imwrite(str(img_variant_path), blended)

        # Copy labels 1:1
        shutil.copy2(lbl_dst, lbl_variant_path)

        # Update stats
        stats_split['images'] += 1
        stats_split['instances'] += valid_lines

# ───────────────────────────── IO helpers ─────────────────────────────

def stage_image(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except (FileExistsError, OSError):
            shutil.copy2(src, dst)


def split_by_video(videos: List[str], val_ratio: float, seed: int):
    rng = random.Random(seed)
    vids = videos[:]
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * val_ratio))) if len(vids) > 1 else 0
    val = set(vids[:n_val])
    train = [v for v in vids if v not in val]
    val = [v for v in vids if v in val]
    if not train and vids:
        train, val = vids[:1], vids[1:]
    return train, val

# ─────────────────────────── core merge logic ───────────────────────────

def read_mask(path: Path) -> np.ndarray:
    """
    Read an ID mask where each pixel value is an object ID.
    Returns (mask, max_id).
    """
    try:
        with Image.open(path) as im:
            # If paletted, np.array gives indices (0..255)
            mask = np.array(im, dtype=np.int32)
    except Exception:
        return None

    if mask.ndim != 2:
        # Unexpected format, e.g., RGB — collapse to one channel
        mask = mask[..., 0].astype(np.int32)

    if mask.size == 0:
        return None

    return mask

def to_binary(mask_like: np.ndarray) -> np.ndarray:
    """Convert any label mask to 0/255 uint8 binary."""
    return ((mask_like > 0).astype(np.uint8)) * 255

def shrink_against_occupied(bin_mask: np.ndarray, occupied: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Subtract already-occupied pixels; return (shrunken_mask, keep_ratio).
    keep_ratio = area_after / max(area_before, 1)
    """
    before = int((bin_mask > 0).sum())
    if before == 0:
        return bin_mask, 0.0
    shrunken = np.where(occupied > 0, 0, bin_mask)
    after = int((shrunken > 0).sum())
    return shrunken.astype(np.uint8), (after / before if before else 0.0)

def select_single_contour(cnts: List[np.ndarray], policy: str) -> Optional[np.ndarray]:
    if not cnts:
        return None
    if len(cnts) == 1:
        return cnts[0]
    if policy == "skip":
        return None
    if policy == "hull":
        pts = np.vstack([c.reshape(-1, 2) for c in cnts])
        return cv2.convexHull(pts.reshape(-1, 1, 2))
    # default: largest
    return max(cnts, key=cv2.contourArea)


def detect_category_from_filename(mask_path: Path) -> Optional[str]:
    """Hook for future per-instance categories via filename.
    E.g., "ketchup__0000123_01.png" → returns "ketchup" if prefix exists.
    Currently returns None (use source.class).
    """
    name = mask_path.stem  # e.g., ketchup__0000123_01
    if "__" in name:
        cat, _rest = name.split("__", 1)
        if cat:
            return cat
    return None

# ─────────────────────────────── main ───────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Merge multiple workspaces into YOLOv11-Seg dataset")
    ap.add_argument("--config", type=Path, required=True, help="Path to sources.yaml")
    ap.add_argument("--out", type=Path, required=True, help="Output dataset root")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio by VIDEO (0-1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    ap.add_argument("--min-area-px", type=int, default=100, help="Min contour area to keep")
    ap.add_argument("--use-chain-none", action="store_true", help="Use CHAIN_APPROX_NONE for densest contours")
    ap.add_argument("--min-keep-ratio", type=float, default=0.25,
                    help="Drop a lower-priority mask if overlap shrink leaves < this fraction of its original area")
    ap.add_argument("--bg-base-dir", type=Path, default=None,
                    help="Directory containing 4 background images for BASE-view frames")
    ap.add_argument("--bg-wrist-dir", type=Path, default=None,
                    help="Directory containing 4 background images for WRIST-view frames")
    args = ap.parse_args()

    # 1) Load YAML config
    cfg = yaml.safe_load(Path(args.config).read_text())
    class_map: List[str] = cfg.get("class_map", ["object"])  # order defines ids
    class_to_id: Dict[str, int] = {c: i for i, c in enumerate(class_map)}

    bg_base = load_backgrounds(args.bg_base_dir)
    bg_wrist = load_backgrounds(args.bg_wrist_dir)

    sources_cfg: List[SourceCfg] = []
    for s in cfg.get("sources", []):
        sources_cfg.append(SourceCfg(
            name=s["name"],
            image_glob=s["image_glob"],
            mask_dirname=s["mask_dirname"],
            mask_ext=s.get("mask_ext", ".png"),
            klass=s.get("class", "object"),
            instance_policy=s.get("instance_policy", "largest"),
            priority=int(s.get("priority", 0)),
        ))

    if not sources_cfg:
        print("[error] No sources in config.", file=sys.stderr)
        sys.exit(1)

    # 2) Discover images per source and build frame index
    # frame_index[key] = {
    #   'image_path': Path,
    #   'video_key': str,
    #   'w': int, 'h': int (lazy),
    #   'masks': [ (priority, class_id, mask_path, source_name) ]
    # }
    frame_index: Dict[Tuple[str, str], Dict] = {}

    for scfg in sources_cfg:
        img_paths = [Path(p) for p in glob.glob(scfg.image_glob, recursive=True)]
        img_paths = [p for p in img_paths if p.is_file()]
        if not img_paths:
            print(f"[warn] No images for source {scfg.name}: {scfg.image_glob}", file=sys.stderr)
            continue

        for img_p in img_paths:
            key = frame_key_for(img_p)
            rec = frame_index.setdefault(key, {
                'image_path': img_p,  # default; may be overwritten by priority
                'video_key': str(get_video_dir(img_p).resolve()),
                'masks': [],
                'image_source_priority': scfg.priority,
                'view': infer_view_from_path(img_p),
            })

            # Choose the most prioritized image path if multiple sources share the same key
            if rec['image_path'] != img_p and scfg.priority > rec['image_source_priority']:
                rec['image_path'] = img_p
                rec['image_source_priority'] = scfg.priority
                rec['view'] = infer_view_from_path(img_p)

            cid = class_to_id.get(scfg.klass, None)
            if cid is None:
                print(f"[warn] class '{scfg.klass}' not in class_map; skipping masks for {img_p}", file=sys.stderr)
                continue

            mp = infer_mask_path(img_p, scfg.mask_dirname, scfg.mask_ext)
            if mp.exists():
                rec['masks'].append((scfg.priority, cid, mp, scfg.name))


    if not frame_index:
        print("[error] No frames discovered.", file=sys.stderr)
        sys.exit(1)

    # 3) Group by video for splitting
    videos = sorted({rec['video_key'] for rec in frame_index.values()})
    train_videos, val_videos = split_by_video(videos, args.val_ratio, args.seed)
    split_of_video = {vk: 'train' for vk in train_videos}
    split_of_video.update({vk: 'val' for vk in val_videos})

    # 4) Prepare output dirs
    img_train = args.out / "images" / "train"
    img_val   = args.out / "images" / "val"
    lbl_train = args.out / "labels" / "train"
    lbl_val   = args.out / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        'train': {'images': 0, 'instances': 0, 'videos': len(train_videos)},
        'val':   {'images': 0, 'instances': 0, 'videos': len(val_videos)},
        'per_class': {c: 0 for c in class_map},
        'per_source': {s.name: 0 for s in sources_cfg},
    }

    # 5) Iterate frames deterministically (by video, then by frame name)
    keys_sorted = sorted(frame_index.keys(), key=lambda k: (k[0], k[1]))

    for key in tqdm(keys_sorted, desc="Merging frames"):
        rec = frame_index[key]
        img_p: Path = rec['image_path']
        video_key: str = rec['video_key']
        split = split_of_video.get(video_key, 'train')

        # Stage image once
        video_name = Path(video_key).name
        stem = f"{video_name}__{img_p.stem}"
        img_dst = (img_train if split == 'train' else img_val) / f"{stem}{img_p.suffix}"
        lbl_dst = (lbl_train if split == 'train' else lbl_val) / f"{stem}.txt"
        stage_image(img_p, img_dst, copy=args.copy)

        # Read image shape (for normalization); fail-safe via imread
        im = cv2.imread(str(img_p))
        if im is None:
            # Remove staged file on failure
            try:
                if img_dst.exists():
                    img_dst.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        h, w = im.shape[:2]

        # Build polylines: list[(cid, contour)]
        polylines: List[Tuple[int, np.ndarray]] = []
        occupied = np.zeros((h, w), dtype=np.uint8)

        # Sort masks by priority high→low for deterministic behavior
        for prio, cid, mp, sname in sorted(rec['masks'], key=lambda x: -x[0]):
            m_id = read_mask(mp)
            if m_id is None:
                continue

            unique_ids = [int(v) for v in np.unique(m_id) if v != 0]
            if not unique_ids:
                continue

            for oid in unique_ids:
                ibin = ((m_id == oid).astype(np.uint8) * 255)

                # Shrink against alread-placed higher-priority regions
                shrunken, keep_ratio = shrink_against_occupied(ibin, occupied)
                min_keep = getattr(args, "min_keep_ratio", 0.25)
                if keep_ratio < min_keep:
                    print(f"[warn] dropped instance source={sname}, "
                          f"class={class_map[cid]} at frame {video_name}__{img_p.stem}",
                          file=sys.stderr)
                    continue

                # Contours & per-source policy
                cnts = find_external_contours(
                    shrunken,
                    min_area=args.min_area_px,
                    use_chain_none=args.use_chain_none,
                    approx_eps_frac=0.002
                )

                source_policy = next((sc.instance_policy for sc in sources_cfg if sc.name == sname), 'largest')
                if source_policy == 'keep_all':
                    chosen = cnts
                else:
                    c = select_single_contour(cnts, policy=source_policy)
                    chosen = [c] if c is not None else []

                # Add chosen contours and update occupied via filled polygon
                for c in chosen:
                    polylines.append((cid, c))
                    occ = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(occ, [c], 255)
                    occupied = np.where(occ > 0, 255, occupied).astype(np.uint8)

        # Write label file; if empty, remove image
        valid_lines = 0
        if polylines:
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            with open(lbl_dst, 'w') as f:
                for cid, cnt in polylines:
                    poly = contour_to_yolo_poly(cnt, w, h)
                    if len(poly) >= 6:
                        f.write(f"{cid} " + " ".join(f"{v:.6f}" for v in poly) + "\n")
                        valid_lines += 1
                        stats['per_class'][class_map[cid]] += 1
            if valid_lines == 0:
                try:
                    lbl_dst.unlink(missing_ok=True)
                except Exception:
                    pass
        if valid_lines == 0:
            # remove staged image if no usable labels
            try:
                if img_dst.exists():
                    if (not args.copy) and img_dst.is_symlink():
                        img_dst.unlink(missing_ok=True)
                    else:
                        img_dst.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        stats[split]['images'] += 1
        stats[split]['instances'] += valid_lines

        view = rec['view']
        if view == "wrist":
            bg_list = bg_wrist
        elif view == "base":
            bg_list = bg_base

        make_bg_variants_for_frame(
            im=im,
            occupied=occupied,
            bg_list=bg_list,
            stem=stem,
            img_suffix=img_p.suffix,
            img_split_dir=img_train if split == 'train' else img_val,
            lbl_split_dir=lbl_train if split == 'train' else lbl_val,
            lbl_dst=lbl_dst,
            valid_lines=valid_lines,
            stats_split=stats[split],
        )

        # Per-source stats (approximate): count 1 per mask contributor line
        for _ in range(valid_lines):
            # cannot reliably attribute to specific source after merges; skip detailed per_source increments
            pass

    # 6) Write dataset YAML
    yaml_path = args.out / "merged-seg.yaml"
    yaml_text = (
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" +
        "".join([f"  {i}: {name}\n" for i, name in enumerate(class_map)])
    )
    yaml_path.write_text(yaml_text)

    # Summary
    print("\n[done] Dataset prepared at:", args.out.resolve())
    print("YAML:", yaml_path.resolve())
    for split in ("train", "val"):
        print(f"  {split}: {stats[split]['images']} images, {stats[split]['instances']} instances, {stats[split]['videos']} videos")
    print("Per-class instances:")
    for name, cnt in stats['per_class'].items():
        print(f"  - {name}: {cnt}")
    print("\nNotes:")
    print(" - Frames are unified by (video_dir.name, frame_stem). Ensure consistent video folder naming across sources.")
    print(" - For multi-instance sources, files matching '<stem>.png' and '<stem>_*.png' are collected.")
    print(" - Instance policy per source: 'largest'|'hull'|'skip'|'keep_all'.")


if __name__ == "__main__":
    main()

