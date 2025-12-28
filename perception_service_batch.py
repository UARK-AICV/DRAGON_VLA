#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple, List

from perception_core import PerceptionService, process_video


def find_episode_dirs(video_root: Path, subject: str) -> List[Tuple[Path, str]]:
    """
    Discover episode directories for a subject.
    Each episode dir should contain camera_base.mp4 and optionally camera_wrist.mp4.
    Returns list of (episode_dir, timestamp).
    """
    subj_root = video_root / subject
    if not subj_root.exists():
        return []

    # Depth=2: e.g., <subject>/<status>/<timestamp>/
    # "success" or "fail" etc. will be captured automatically.
    dirs = [p for p in subj_root.glob("success/*") if p.is_dir()]
    episodes: List[Tuple[Path, str]] = []
    for d in sorted(dirs):
        ts = d.name
        episodes.append((d, ts))
    return episodes


def process_episode_pair(
    service: PerceptionService,
    *,
    base_path: Path,
    wrist_path: Path,
    out_base: Path,
    out_wrist: Path,
) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_wrist.parent.mkdir(parents=True, exist_ok=True)

    process_video(
        service=service,
        input_base=str(base_path),
        input_wrist=str(wrist_path),
        out_base=str(out_base),
        out_wrist=str(out_wrist),
        select_objects="",
        exclude_objects="",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch video processing with PerceptionService (subject-level)."
    )
    # Dataset structure
    p.add_argument("--video-root", type=Path, required=True,
                   help="Root directory containing subject folders")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Directory to write overlay videos")
    p.add_argument("--subject", type=str, required=True,
                   help="Subject name (e.g., 'ketchup', 'oliveoil')")

    # Output layout
    p.add_argument("--per-episode-dir", action="store_true",
                   help="Write outputs under OUT_ROOT/<subject>/<status>/<timestamp>/")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run, without processing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    episodes = find_episode_dirs(args.video_root, args.subject)
    if not episodes:
        print(f"[INFO] No episodes found for subject '{args.subject}' under {args.video_root}", file=sys.stderr)
        return

    service = PerceptionService(use_vlm=False)

    for ep_dir, ts in episodes:
        base_vid = ep_dir / "camera_base.mp4"
        wrist_vid = ep_dir / "camera_wrist.mp4"
        assert base_vid.exists() and wrist_vid.exists(), f"Missing base/wrist video: {base_vid}, {wrist_vid}"

        if args.per_episode_dir:
            dest_dir = args.out_root / args.subject / ep_dir.parent.name / ts
            out_base = dest_dir / "camera_base_depth_masked.mp4"
            out_wrist = dest_dir / "camera_wrist_depth_masked.mp4"
        else:
            dest_dir = args.out_root
            out_base = dest_dir / f"{args.subject}_{ts}_camera_base.mp4"
            out_wrist = dest_dir / f"{args.subject}_{ts}_camera_wrist.mp4"

        if not args.overwrite and out_base.exists() and (out_wrist is None or out_wrist.exists()):
            print(f"[SKIP] Already exists: {out_base.name}" + (f", {out_wrist.name}" if out_wrist else ""))
            continue

        print(f"→ Processing {args.subject} / {ts} → {dest_dir}")
        if args.dry_run:
            continue

        process_episode_pair(
            service=service,
            base_path=base_vid,
            wrist_path=wrist_vid,
            out_base=out_base,
            out_wrist=out_wrist,
        )

    print(f"✓ All episodes processed for subject '{args.subject}' into {args.out_root}")


if __name__ == "__main__":
    main()

