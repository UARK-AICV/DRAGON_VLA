from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import numpy as np
import io

from perception_core import PerceptionService


app = FastAPI()
app.state.service = PerceptionService(use_vlm=True)


def _npy_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _bytes_to_npy(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


@app.post("/process")
def process_frame(
    select_objects: str = Form(...),
    exclude_objects: str = Form(""),
    is_base_init: bool = Form(False),
    is_wrist_init: bool = Form(False),
    base_npy: UploadFile = File(...),   # client sends .npy (H,W,3) uint8
    wrist_npy: UploadFile = File(...),
):
    base_bytes = base_npy.file.read()
    wrist_bytes = wrist_npy.file.read()

    base_rgb_np = _bytes_to_npy(base_bytes)
    wrist_rgb_np = _bytes_to_npy(wrist_bytes)

    base_overlay, wrist_overlay = app.state.service.process(
        select_objects=select_objects,
        exclude_objects=exclude_objects,
        is_base_init=is_base_init,
        is_wrist_init=is_wrist_init,
        base_rgb_np=base_rgb_np,
        wrist_rgb_np=wrist_rgb_np,
    )

    # Pack both arrays into multipart/mixed response
    boundary = "overlayboundary"
    parts = []
    for name, arr in [
        ("overlay_base.npy", base_overlay),
        ("overlay_wrist.npy", wrist_overlay),
    ]:
        payload = _npy_to_bytes(arr)
        parts.append(
            f"--{boundary}\r\n"
            "Content-Type: application/octet-stream\r\n"
            f"Content-Disposition: attachment; filename={name}\r\n\r\n"
            .encode("utf-8") + payload + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    return Response(
        content=b"".join(parts),
        media_type=f"multipart/mixed; boundary={boundary}",
    )


@app.post("/reset")
def reset():
    return {"ok": True}

