from typing import Dict, List, Tuple, Optional, Literal, Union
import numpy as np
from PIL import Image
import cv2


def ensure_pil(x: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.stack([x]*3, axis=-1)
        return Image.fromarray(x.astype(np.uint8)).convert("RGB")


def ensure_rgb_array(x: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(x, Image.Image):
        return np.array(x.convert("RGB"))
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    return x.astype(np.uint8)


def build_per_object_mask(
    mask_img: Union[np.ndarray, Image.Image],                          # FastSAM indexed mask, mode 'L' or 'P'
    track_objs: Dict[str, List[int]],               # {"ketchup":[3,7], "gripper":[12,15,...], ...}
    object_order: Optional[List[str]] = None,       # fixed label order; default = sorted(track_objs.keys())
    overlap_policy: Literal["first", "smallest", "error"] = "smallest",
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Build a new indexed mask with ONE label per object (1..K), merging the given fragment IDs.

    Returns:
        label_mask: 2D ndarray, bg=0, objects labeled 1..K (in order of `object_order`)
        label_to_name: {label:int -> name:str}
        name_to_label: {name:str -> label:int}

    Notes:
      - Ignores fragment IDs that don't exist in `mask_img`.
      - If different objects' fragment sets *overlap* on pixels:
          * "first"   -> earlier object in `object_order` keeps those pixels
          * "smallest" -> assign overlap to the object whose union area is larger
          * "error"   -> raise ValueError
    """
    mask_arr = np.asarray(mask_img) if isinstance(mask_img, Image.Image) else mask_img
    H, W = mask_arr.shape

    # Determine object order (stable labeling)
    names = list(track_objs.keys()) if object_order is None else list(object_order)

    out = np.zeros((H, W), dtype=mask_arr.dtype)    # final per-object labels (0=bg)
    label_to_name: Dict[int, str] = {}
    name_to_label: Dict[str, int] = {}

    # Precompute per-object unions (binary) and areas
    unions: Dict[str, np.ndarray] = {}
    areas: Dict[str, int] = {}
    present_ids = set(np.unique(mask_arr).tolist())

    for name in names:
        ids = [int(i) for i in (track_objs.get(name) or []) if int(i) in present_ids and int(i) != 0]
        if not ids:
            unions[name] = np.zeros((H, W), dtype=bool)
            areas[name]  = 0
            continue
        sel = np.isin(mask_arr, ids)
        unions[name] = sel
        areas[name] = int(sel.sum())

    # Resolve overlaps according to policy
    if overlap_policy == "smallest":
        # Order names by descending area so smallest gets first claim
        ordered_names = sorted(names, key=lambda n: areas[n], reverse=True)
    else:
        ordered_names = names

    claimed = np.zeros((H, W), dtype=bool)
    label = 1

    for name in ordered_names:
        U = unions[name]
        if areas[name] == 0:
            continue

        if overlap_policy == "error":
            overlap = np.logical_and(claimed, U)
            if overlap.any():
                raise ValueError(f"Overlap detected for object '{name}'. "
                                 f"Consider overlap_policy='first' or 'smallest'.")

        # Pixels we can newly claim for this object
        new_pixels = np.logical_and(U, ~claimed)
        if new_pixels.any():
            out[new_pixels] = label
            claimed[new_pixels] = True
            label_to_name[label] = name
            name_to_label[name] = label
            label += 1
        else:
            # Object had only overlaps (no free pixels)
            # Still register mapping if you need stable IDs even when empty
            if name not in name_to_label:
                label_to_name[label] = name
                name_to_label[name] = label
                label += 1

    return out, label_to_name, name_to_label


def arbitrate_detector_tracker_masks(
    detector_idx: np.ndarray,
    tracker_idx: np.ndarray,
    object_names: List[str],
    iou_agree_high: float = 0.55,
    iou_conflict_low: float = 0.20,
    contain_thresh: float = 0.7,
    conflict_winner: str = "detector",
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Compare detector vs tracker masks per object, build final mask.
    Returns:
        out_idx: 2D mask, 0=bg, 1..K for each object in object_names order
        decisions: {name: "detector"|"tracker"}
    """
    assert detector_idx.shape == tracker_idx.shape
    H, W = detector_idx.shape
    out_idx = np.zeros((H, W), dtype=np.uint8)
    decisions: Dict[str, str] = {}

    for k, name in enumerate(object_names, start=1):
        V = (detector_idx == k)
        C = (tracker_idx == k)
        av, ac = V.sum(), C.sum()

        if av == 0 and ac == 0:
            winner = "tracker"
            mask = np.zeros_like(V)
        elif av > 0 and ac == 0:
            winner = "detector"
            mask = V
        elif av == 0 and ac > 0:
            winner = "tracker"
            mask = C
        else:
            inter = np.logical_and(V, C).sum()
            union = np.logical_or(V, C).sum()
            iou = inter / max(1, union)
            crv = inter / max(1, av)  # V inside C
            crc = inter / max(1, ac)  # C inside V

            if iou >= iou_agree_high:
                if av >= ac:
                    winner, mask = "detector", V
                else:
                    winner, mask = "tracker", C
            elif iou <= iou_conflict_low:
                if conflict_winner == "detector":
                    winner, mask = "detector", V
                else:
                    winner, mask = "tracker", C
            else:
                if crv >= contain_thresh:
                    winner, mask = "tracker", C
                elif crc >= contain_thresh:
                    winner, mask = "detector", V
                else:
                    if conflict_winner == "detector":
                        winner, mask = "detector", V
                    else:
                        winner, mask = "tracker", C

        out_idx[mask] = k
        decisions[name] = winner

    return out_idx, decisions

