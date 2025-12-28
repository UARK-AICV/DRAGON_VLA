# yolo11_video_single_window.py
import cv2
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video or '0' for webcam")
    ap.add_argument("--weights", default="runs/exp_robotarm_unfreeze/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--save", type=str, default="", help="Optional: output .mp4 path")
    args = ap.parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(0 if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    # Optional writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, imgsz=args.imgsz, verbose=False)
        vis = results[0].plot()  # draw boxes/masks onto a copy

        if writer:
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
