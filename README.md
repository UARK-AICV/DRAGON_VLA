# DRAGON-VLA: Distraction-Robust Vision-Language-Action Models through Object-Centric and Geometry Grounding

### [[Project Page]](https://uark-aicv.github.io/DRAGON_VLA/)

This is the official code for the perception grounding module of DRAGON-VLA (Distraction-Robust Action via Geometric and Object-ceNtric Grounding). DRAGON-VLA is an integrated perception–control system that improves instruction–scene alignment under visual distraction through frozen, two-stage object-centric grounding and masked relative-depth grounding. The grounded observations condition a VLA fine-tuned only on clean, single-object demonstrations.

This codebase exposes two main entrypoints:
- `perception_service_batch.py`: batch processing to generate training data (mask selection, depth, and overlays).
- `perception_service_fastapi.py`: FastAPI service for deployment.

## Notes
- This codebase currently does not contain the real-world robot interface or the action reasoning policy. We developed our robot interface from [diffusion-policy](https://github.com/real-stanford/diffusion_policy) to work with UR10e. For action reasoning, we employ [openpi](https://github.com/Physical-Intelligence/openpi).
- Ensure you have at least 24GB of GPU for YOLO + DepthAnythingV2.
- Qwen3-VL 8B-Instruct requires one A6000 GPU, if you do not have one, you can change the code to use OpenAI's API instead.
- Paths are relative; keep `weights/yolo11l-seg.pt` and downloaded DepthAnythingV2 checkpoints in expected locations.
- If modifying weights paths, update `YOLO11SegEngineConfig` or pass a custom path.

## Environment (uv recommended)
1) Install [uv](https://docs.astral.sh/uv) following the instructions.
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

## Batch processing (training data preprocessing)
- Script: `batch_processing.sh`
- Defaults: reads videos under `assets/grocery_items` and writes outputs there.
- Usage: `./batch_processing.sh <subject1> [subject2 ...]`
- Env toggles:
  - `VIDEO_ROOT` (default `assets/grocery_items`)
  - `OUT_ROOT` (default `assets/grocery_items`)
  - `OVERWRITE=1|0`, `PER_EPISODE_DIR=1|0`, `DRY_RUN=1|0`
- Internally calls `perception_service_batch.py` to generate perception grounded data for VLA policy training.

## Deployment (FastAPI)
- Entry: `perception_service_fastapi.py`
- Run (from repo root):
  ```
  uv run uvicorn perception_service_fastapi:app --host 0.0.0.0 --port 8000
  ```
- Accepts image uploads and returns processed outputs using the same perception core.
