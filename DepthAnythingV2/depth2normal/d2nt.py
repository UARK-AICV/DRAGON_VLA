# d2nt.py
import numpy as np
import cv2
from .utils import (
    get_filter,
    get_DAG_filter,
    vector_normalization,
    MRF_optim,
)

DEFAULT_VERSION = 'd2nt_v3'


def normals_to_rgb(normals: np.ndarray, return_uint8: bool = True) -> np.ndarray:
    """
    Convert normals in [-1,1] to RGB image in [0,255].

    Args:
        normals: (H, W, 3) float32 array, values in [-1,1].
        return_uint8: if True, return uint8 image; else float32 in [0,1].

    Returns:
        RGB visualization (H, W, 3).
    """
    rgb = (normals + 1.0) / 2.0  # map -1..1 → 0..1
    if return_uint8:
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def depth_to_normal(
    depth: np.ndarray,
    cam_fx: float | None = None,
    cam_fy: float | None = None,
    u0: float | None = None,
    v0: float | None = None,
    version: str = DEFAULT_VERSION,
) -> np.ndarray:
    """
    Convert depth map to normal map.

    Parameters
    ----------
    depth : np.ndarray
        Input depth map (uint16 or float), shape (H, W).
    cam_fx, cam_fy : float, optional
        Focal lengths. If None, fake intrinsics will be generated.
    u0, v0 : float, optional
        Principal point. If None, set to image center.
    version : str
        Pipeline version: ['d2nt_basic', 'd2nt_v2', 'd2nt_v3'].

    Returns
    -------
    est_normal : np.ndarray
        Estimated surface normal map, shape (H, W, 3), float32 normalized.
    """

    # convert depth to float32
    depth = depth.astype(np.float32)
    depth = depth * 1000
    h, w = depth.shape[:2]

    # --- Fake intrinsics if not provided ---
    if cam_fx is None or cam_fy is None or u0 is None or v0 is None:
        # assume ~60° FOV → focal length ≈ image size
        cam_fx = w if cam_fx is None else cam_fx
        cam_fy = h if cam_fy is None else cam_fy
        u0 = (w - 1) / 2 if u0 is None else u0
        v0 = (h - 1) / 2 if v0 is None else v0

    # build u and v coordinate maps
    u_map = np.ones((h, 1), dtype=np.float32) * np.arange(1, w + 1, dtype=np.float32) - u0
    v_map = np.arange(1, h + 1, dtype=np.float32).reshape(h, 1) * np.ones((1, w), dtype=np.float32) - v0

    # get depth gradients
    if version == 'd2nt_basic':
        Gu, Gv = get_filter(depth)
    else:
        Gu, Gv = get_DAG_filter(depth)

    # Depth-to-Normal translation
    est_nx = Gu * cam_fx
    est_ny = Gv * cam_fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))

    # normalize vectors
    est_normal = vector_normalization(est_normal)

    # refinement
    if version == 'd2nt_v3':
        est_normal = MRF_optim(depth, est_normal)

    est_normal = normals_to_rgb(est_normal)
    return est_normal

