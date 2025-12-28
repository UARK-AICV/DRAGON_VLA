# OBEYED-VLA: Clutter-Resistant Vision-Language-Action Models through Object-Centric and Geometry Grounding

## [Project Page](https://uark-aicv.github.io/OBEYED_VLA)

This is the official code for the perception grounding module of OBEYED-VLA. OBEYED-VLA decouples perception from control using frozen VLM-based object-centric grounding plus masked-depth geometric grounding, then fine-tunes a VLA only on clean single-object demos. This codebase exposes two main entrypoints:
- `perception_service_batch.py`: batch processing to generate training data (mask selection, depth, and overlays).
- `perception_service_fastapi.py`: FastAPI service for deployment.

**Note:** this codebase currently does not contain the real-world robot interface or the action reasoning policy. We developed our robot interface from [diffusion-policy](https://github.com/real-stanford/diffusion_policy) to work with UR10e. For action reasoning, we employ [openpi](https://github.com/Physical-Intelligence/openpi).

## Environment (uv recommended)
1) Install uv: https://docs.astral.sh/uv/
2) From repo root, resolve deps from `pyproject.toml`: `uv sync` (creates/uses a 3.10 env automatically).

## Checkpoints
- **YOLO segmentation**: default path `weights/yolo11l-seg.pt` (included).
- **DepthAnythingV2**: download the metric depth weights (not bundled):
  ```bash
  mkdir -p DepthAnythingV2/checkpoints
  wget -O DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    https://huggingface.co/LiheYoung/Depth-Anything-V2/resolve/main/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
  ```
  Adjust the filename if you switch encoders (`vits`, `vitb`, `vitl`, `vitg`).
- **Cutie**: if weights are missing, place them under `Cutie/weights/`:
  ```bash
  mkdir -p Cutie/weights
  wget -O Cutie/weights/cutie-base-mega.pth \
    https://huggingface.co/SHI-Labs/Cutie/resolve/main/weights/cutie-base-mega.pth
  ```
  Update to the latest links from the Cutie repo if needed.

## VLM configuration
- The perception service queries a hosted VLM; update `VLM_MODEL` and `BASE_URL` in `perception_core.py` to match your served model and endpoint.
- We use [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (8B-Instruct) and serve it via vLLM; adapt to your deployment as needed.

## Batch processing (training data)
- Script: `batch_processing.sh`
- Defaults: reads videos under `assets/grocery_items` and writes outputs there.
- Usage: `./batch_processing.sh <subject1> [subject2 ...]`
- Env toggles:
  - `VIDEO_ROOT` (default `assets/grocery_items`)
  - `OUT_ROOT` (default `assets/grocery_items`)
  - `OVERWRITE=1|0`, `PER_EPISODE_DIR=1|0`, `DRY_RUN=1|0`
- Internally calls `perception_service_batch.py` with YOLO masks, VLM mask selection, and depth rendering.

## Deployment (FastAPI)
- Entry: `perception_service_fastapi.py`
- Run (from repo root):
  ```
  uv run uvicorn perception_service_fastapi:app --host 0.0.0.0 --port 8000
  ```
- Accepts image uploads and returns processed outputs using the same perception core.

## Notes
- Ensure GPU access for best performance (YOLO + DepthAnythingV2).
- Paths are relative; keep `weights/yolo11l-seg.pt` and downloaded DepthAnythingV2 checkpoints in expected locations.
- If modifying weights paths, update `YOLO11SegEngineConfig` or pass a custom path.
