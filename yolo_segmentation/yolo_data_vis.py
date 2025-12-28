#!/usr/bin/env python3

import argparse, sys, time, random
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

def list_images(img_dir: Path, exts=(".jpg",".jpeg",".png")):
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in exts])

def parse_label_file(txt_path: Path):
    anns = []
    if not txt_path.exists():
        return anns
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except:
            continue
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            cx, cy, w, h = vals
            anns.append({"cls": cls, "type": "bbox", "bbox": (cx, cy, w, h)})
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
            anns.append({"cls": cls, "type": "poly", "poly": pts})
    return anns

def denorm_xyxy(box01, w, h):
    cx, cy, bw, bh = box01
    x1 = (cx - bw/2.0) * (w - 1)
    y1 = (cy - bh/2.0) * (h - 1)
    x2 = (cx + bw/2.0) * (w - 1)
    y2 = (cy + bh/2.0) * (h - 1)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def denorm_poly(poly01, w, h):
    pts = poly01.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * (w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * (h - 1)
    return pts.astype(np.int32).reshape(-1, 1, 2)

def draw_ann(img, anns, alpha=0.65, show_overlay=True):
    if not show_overlay:
        return img
    out = img.copy()
    overlay = img.copy()
    h, w = img.shape[:2]
    for k, ann in enumerate(anns):
        rng = np.random.RandomState(k * 977 + 123)
        color = (int(rng.randint(50, 255)), int(rng.randint(50, 255)), int(rng.randint(50, 255)))
        if ann["type"] == "bbox":
            x1, y1, x2, y2 = denorm_xyxy(ann["bbox"], w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        else:
            pts = denorm_poly(ann["poly"], w, h)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], True, color, 2)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

def infer_video_key(path: Path, delim="__", fallback_parent=True):
    name = path.stem
    if delim in name:
        return name.split(delim, 1)[0]
    return path.parent.name if fallback_parent else name

def build_video_index(img_dir: Path, lbl_dir: Path, delim="__", shuffle=False, seed=42, allowlist=None):
    images = list_images(img_dir)
    if not images:
        print(f"[error] no images in {img_dir}", file=sys.stderr)
        sys.exit(1)
    groups = defaultdict(list)
    for p in images:
        key = infer_video_key(p, delim=delim, fallback_parent=True)
        if allowlist and key not in allowlist:
            continue
        groups[key].append(p)
    if not groups:
        print("[error] no images matched filters", file=sys.stderr)
        sys.exit(1)
    for key in groups:
        groups[key].sort(key=lambda x: x.name)
    video_keys = sorted(groups.keys())
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(video_keys)
    videos = []
    for key in video_keys:
        frames = []
        for ip in groups[key]:
            lp = lbl_dir / f"{ip.stem}.txt"
            frames.append((ip, lp))
        videos.append((key, frames))
    return videos

def must_open_window(win_name: str, fullscreen: bool):
    # Fail fast with a helpful error if HighGUI can't open a window.
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception as e:
        msg = (
            f"[error] cv2.namedWindow failed: {e}\n"
            "Tips:\n"
            " - Ensure a DISPLAY is available (e.g., run outside headless, or `export DISPLAY=:0`).\n"
            " - Avoid headless runners for this script (don't use xvfb/uv-run headless).\n"
            " - Use `pip install opencv-python` (not headless) and a GUI session (QT/GTK).\n"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Dataset root with images/ and labels/")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--fps", type=float, default=15.0, help="Playback FPS")
    ap.add_argument("--alpha", type=float, default=0.65, help="Overlay alpha")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle video order")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delim", type=str, default="__", help="Video/frame delimiter in filename")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--show-info", action="store_true")
    ap.add_argument("--videos", type=str, default=None, help="Comma list of video keys to play (e.g. cam01,cam02)")
    args = ap.parse_args()

    img_dir = args.root / "images" / args.split
    lbl_dir = args.root / "labels" / args.split
    if not img_dir.exists() or not lbl_dir.exists():
        print("[error] missing split dirs:", img_dir, lbl_dir, file=sys.stderr)
        sys.exit(1)

    allowlist = set([s.strip() for s in args.videos.split(",")]) if args.videos else None
    videos = build_video_index(img_dir, lbl_dir, delim=args.delim, shuffle=args.shuffle,
                               seed=args.seed, allowlist=allowlist)
    total_videos = len(videos)
    total_frames = sum(len(frames) for _, frames in videos)
    if total_frames == 0:
        print("[error] no frames found", file=sys.stderr)
        sys.exit(1)
    print(f"[info] Found {total_videos} videos, {total_frames} frames")

    # Create window (must succeed)
    win = "YOLO Dataset Player"
    must_open_window(win, args.fullscreen)

    # Playback state
    vi, fi = 0, 0
    paused = False
    show_overlay = True
    show_info = args.show_info
    alpha = max(0.1, min(0.95, args.alpha))
    delay_ms = max(1, int(1000.0 / max(1e-3, args.fps)))
    last_tick = time.time()

    while True:
        key_name, frames = videos[vi]
        if not frames:
            vi = (vi + 1) % total_videos
            fi = 0
            continue

        img_path, lbl_path = frames[fi]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            # Skip unreadable frame
            fi = (fi + 1) % len(frames)
            continue

        anns = parse_label_file(lbl_path)
        vis = draw_ann(img, anns, alpha=alpha, show_overlay=show_overlay)

        if show_info:
            info = f"[{vi+1}/{total_videos} {fi+1}/{len(frames)}] num_anns={len(anns)} video={key_name}"
            cv2.putText(vis, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 230, 30), 1, cv2.LINE_AA)

        cv2.imshow(win, vis)

        wait = 0 if paused else delay_ms
        k = cv2.waitKey(wait) & 0xFF

        if paused:
            if k in (ord('q'), 27): break
            elif k == ord(' '): paused = False
            elif k == ord('t'): show_info = not show_info
            elif k == ord('i'): show_overlay = not show_overlay
            elif k == ord('['): delay_ms = min(2000, delay_ms + 10)
            elif k == ord(']'): delay_ms = max(1, delay_ms - 10)
            elif k == ord('f'):
                prop = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
            elif k == ord(','):
                fi -= 1
                if fi < 0:
                    vi = (vi - 1 + total_videos) % total_videos
                    fi = len(videos[vi][1]) - 1
            elif k == ord('.'):
                fi += 1
                if fi >= len(frames):
                    vi = (vi + 1) % total_videos
                    fi = 0
        else:
            if k in (ord('q'), 27): break
            elif k == ord(' '): paused = True
            elif k == ord('t'): show_info = not show_info
            elif k == ord('i'): show_overlay = not show_overlay
            elif k == ord('['): delay_ms = min(2000, delay_ms + 10)
            elif k == ord(']'): delay_ms = max(1, delay_ms - 10)
            elif k == ord('f'):
                prop = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

            now = time.time()
            if (now - last_tick) * 1000.0 >= delay_ms:
                last_tick = now
                fi += 1
                if fi >= len(frames):
                    vi = (vi + 1) % total_videos
                    fi = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
