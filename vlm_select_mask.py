#!/usr/bin/env python3
# openai_mask_identifier_video.py

from typing import Any, Union, Optional, Dict, List

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

from Cutie.tracking_stream import CutieStreamTracker
from ops.io import np_to_data_url_png, ids_in_mask
from ops.overlay import overlay_with_indexed_mask, highlight_with_indexed_masks
from ops.misc import (
    ensure_pil,
    build_per_object_mask,
)
from ops.vlm import build_messages
from ops.parsing import parse_answer


class VLMMaskSelector:
    def __init__(self,
                 base_url: str,
                 model_name: str,
                 api_key: str = "EMPTY",
                 cutie_max_size: int = -1):
        print("⏳  Connecting to OpenAI-compatible API ...", flush=True)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        print("✅  Client ready.\n", flush=True)

        # tracking + policy
        self.tracker = CutieStreamTracker(max_size=cutie_max_size)
        self._tracker_initialized = False
        self._track_objs: Dict[str, List[int]] = {}
        self._object_order: List[str] = []

    # ─────────────────────────────── single image API ───────────────────────────
    def _vlm_pick_ids(
        self,
        frame_rgb_np: Union[np.ndarray, Image.Image],
        indexed_mask: np.ndarray,                  # (H,W) ints, 0=bg, 1..N
        object_names: List[str],
        reference_crops: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Render overlay (outlines+numbers) and query VLM which IDs correspond to each object."""
        overlay_pil = overlay_with_indexed_mask(frame_rgb=frame_rgb_np, indexed_mask=indexed_mask)
        valid_ids = ids_in_mask(indexed_mask)
        data_url = np_to_data_url_png(overlay_pil)
        reference_urls: Optional[Dict[str, str]] = None
        if reference_crops:
            reference_urls = {
                name: np_to_data_url_png(crop)
                for name, crop in reference_crops.items()
            }
        messages = build_messages(
            data_url,
            object_names,
            valid_ids=valid_ids,
            reference_images=reference_urls,
        )

        raw_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        ).choices[0].message.content.strip()

        parsed_answer = parse_answer(raw_response)
        return {"parsed_answer": parsed_answer, "raw_answer": raw_response}

    def query_single_image(
        self,
        rgb_np: np.ndarray,
        object_names: List[str],
        exclude_object_names: List[str],
        is_init: bool = False,
        objects_mask: Optional[np.ndarray] = None,
        vlm_downsample: int = 1,
        reference_crops: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        if is_init:
            if objects_mask is None:
                raise ValueError("objects_mask is required when is_init=True")
            rgb_vlm = rgb_np
            objects_mask_vlm = objects_mask
            if vlm_downsample > 1:
                H, W = rgb_np.shape[:2]
                rgb_vlm = cv2.resize(
                    rgb_np,
                    (W // vlm_downsample, H // vlm_downsample),
                    interpolation=cv2.INTER_AREA,
                )
                objects_mask_vlm = cv2.resize(
                    objects_mask.astype(np.uint8),
                    (W // vlm_downsample, H // vlm_downsample),
                    interpolation=cv2.INTER_NEAREST,
                )
            vlm_blob = self._vlm_pick_ids(
                rgb_vlm,
                objects_mask_vlm,
                object_names + exclude_object_names,
                reference_crops=reference_crops,
            )
            obj_ids = vlm_blob["parsed_answer"] or {}
            print('obj ids: ', obj_ids)
            for name in object_names:
                obj_ids.setdefault(name, [])

            mask_arr, _, _ = build_per_object_mask(
                objects_mask, obj_ids, object_order=object_names)
            self._track_objs = {name: obj_ids.get(name, []) for name in object_names}
            self._object_order = list(object_names)
            try:
                self.tracker.run_single_frame(
                    ensure_pil(rgb_np),
                    init_mask_img=objects_mask,
                    track_objs=self._track_objs,
                )
                self._tracker_initialized = True
            except ValueError:
                self._tracker_initialized = False
        else:
            if not self._tracker_initialized:
                H, W = rgb_np.shape[:2]
                mask_arr = np.zeros((H, W), dtype=np.uint8)
            else:
                mask_arr = self.tracker.run_single_frame(ensure_pil(rgb_np))

        #overlay_pil = highlight_with_indexed_masks(rgb_np, mask_arr, dim_factor=0.2)
        #overlay_pil = overlay_with_indexed_mask(rgb_np, mask_arr)

        return {
            "mask_arr": mask_arr,
            #"overlay_pil": overlay_pil,
        }
