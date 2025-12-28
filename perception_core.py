from __future__ import annotations
from typing import Dict, Optional, Tuple
from numpy.typing import NDArray

import cv2
import numpy as np
from tqdm import tqdm

from DepthAnythingV2.depthany_v2_engine import DepthAnyV2Engine, DepthAnyV2EngineConfig

from yolo_segmentation.yolo_segm_engine import YOLO11SegEngine, YOLO11SegEngineConfig
from vlm_select_mask import VLMMaskSelector
from ops.io import (
    open_video_reader, read_video_meta, make_writer,
)
from ops.overlay import highlight_masks_black_background

VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BASE_URL = "localhost:8080/v1"

class PerceptionService:
    def __init__(
        self,
        use_vlm: bool = False,
        base_vlm_downsample: int = 1,
        wrist_vlm_downsample: int = 3,
    ):
        self.depth_engine = DepthAnyV2Engine(DepthAnyV2EngineConfig(
            encoder="vitl",
            input_size=518,
            palette="rainbow",
            scale_meters=2.0,
        ))
        self.segmenter = YOLO11SegEngine(YOLO11SegEngineConfig(
            device="cuda",
            imgsz=960,
        ))

        if use_vlm:
            self.base_selector = VLMMaskSelector(
                base_url=BASE_URL,
                model_name=VLM_MODEL,
                cutie_max_size=-1,
                conflict_winner="vlm",
            )

            self.wrist_selector = VLMMaskSelector(
                base_url=BASE_URL,
                model_name=VLM_MODEL,
                cutie_max_size=-1,
                conflict_winner="vlm",
            )
        else:
            self.base_selector = None
            self.wrist_selector = None

        self.use_vlm = use_vlm
        assert base_vlm_downsample >= 1
        assert wrist_vlm_downsample >= 1
        self.base_vlm_downsample = base_vlm_downsample
        self.wrist_vlm_downsample = wrist_vlm_downsample

    def _build_object_crops(
        self,
        rgb_np: NDArray[np.uint8],
        mask_arr: NDArray[np.uint8],
        object_names: list[str],
        short_side: int = 50,
    ) -> Dict[str, NDArray[np.uint8]]:
        crops: Dict[str, NDArray[np.uint8]] = {}
        for idx, name in enumerate(object_names, start=1):
            obj_mask = mask_arr == idx
            if not obj_mask.any():
                continue
            ys, xs = np.where(obj_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = rgb_np[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            scale = float(short_side) / float(min(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
            crops[name] = crop_resized
        return crops

    def process_view(
        self,
        rgb_np: NDArray[np.uint8],
        select_object: str,
        exclude_objs: list[str],
        selector: Optional[VLMMaskSelector] = None,
        is_init: bool = False,
        reference_crops: Optional[Dict[str, NDArray[np.uint8]]] = None,
    ) -> Tuple[NDArray[np.uint8], Dict[str, NDArray[np.uint8]]]:
        depth_viz_np = self.depth_engine.run_single_frame(rgb_np)
        robot_objects_mask, class2id = self.segmenter.process_single_frame(
            rgb_np, channel_order="RGB"
        )
        base_crops: Dict[str, NDArray[np.uint8]] = {}
        if selector is not None:
            objects_mask_vlm = robot_objects_mask.get("object", None)
            vlm_downsample = (
                self.wrist_vlm_downsample
                if selector is self.wrist_selector
                else self.base_vlm_downsample
            )
            object_names = [select_object, "grey bin"]
            if is_init:
                result = selector.query_single_image(rgb_np=rgb_np,
                    object_names=object_names, exclude_object_names=exclude_objs,
                    is_init=is_init, objects_mask=objects_mask_vlm,
                    vlm_downsample=vlm_downsample,
                    reference_crops=reference_crops,
                )
            else:
                result = selector.query_single_image(rgb_np=rgb_np,
                    object_names=object_names, exclude_object_names=exclude_objs,
                    objects_mask=objects_mask_vlm,
                    reference_crops=reference_crops,
                )
            mask = {
                "object": result["mask_arr"],
                "robot_arm": robot_objects_mask.get("robot_arm", None),
            }
            if selector is self.base_selector:
                base_crops = self._build_object_crops(
                    rgb_np, result["mask_arr"], object_names
                )
        else:
            mask = robot_objects_mask
        overlay = highlight_masks_black_background(depth_viz_np, mask, class2id)
        #overlay = highlight_masks_black_background(rgb_np, mask, class2id)
        return overlay, base_crops

    def process(self, select_objects: str,
                exclude_objects: str, is_base_init: bool, is_wrist_init: bool,
                base_rgb_np: NDArray, wrist_rgb_np: NDArray) -> Tuple[NDArray, NDArray]:
        select_objs = select_objects.split(";")
        exclude_objs = exclude_objects.split(";")

        base_overlay, base_crops = self.process_view(rgb_np=base_rgb_np,
            select_objs=select_objs, exclude_objs=exclude_objs,
            selector=self.base_selector, is_init=is_base_init,
        )
        wrist_overlay, _ = self.process_view(rgb_np=wrist_rgb_np,
            select_object=select_objs, exclude_objs=exclude_objs,
            selector=self.wrist_selector, is_init=is_wrist_init,
            reference_crops=base_crops,
        )
        return base_overlay, wrist_overlay

def process_video(service: PerceptionService, input_base: str, input_wrist: str,
    out_base: str, out_wrist: str, select_objects: str, exclude_objects: str,
    limit_frames: Optional[int] = None, wrist_init_period: int = 10,
):
    cap_base = open_video_reader(input_base)
    count_b, w_b, h_b, fps_b = read_video_meta(cap_base)

    cap_wrist = open_video_reader(input_wrist)
    count_w, w_w, h_w, fps_w = read_video_meta(cap_wrist)
    fps = fps_b if abs(fps_b - fps_w) < 1e-2 else fps_b

    assert count_b == count_w, "Videos must have the same number of frames"
    limit_frames = count_b if (limit_frames is None or limit_frames > count_b) else limit_frames

    writer_base = make_writer(out_base, w_b, h_b, fps)
    writer_wrist = make_writer(out_wrist, int(w_w), int(h_w), fps)

    try:
        for frame_idx in tqdm(range(limit_frames)):
            _, fb = cap_base.read()
            _, fw = cap_wrist.read()
            fb = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
            fw = cv2.cvtColor(fw, cv2.COLOR_BGR2RGB)

            base_rgb_np, wrist_rgb_np = service.process(
                select_objects=select_objects,
                exclude_objects=exclude_objects,
                is_base_init=(frame_idx == 0),
                is_wrist_init=(frame_idx % wrist_init_period == 0),
                base_rgb_np=fb,
                wrist_rgb_np=fw,
            )

            base_bgr_np = cv2.cvtColor(np.asarray(base_rgb_np), cv2.COLOR_RGB2BGR)
            wrist_bgr_np = cv2.cvtColor(np.asarray(wrist_rgb_np), cv2.COLOR_RGB2BGR)

            # Resize overlays to writer sizes if needed (safety)
            if base_bgr_np.shape[1] != w_b or base_bgr_np.shape[0] != h_b:
                base_bgr_np = cv2.resize(base_bgr_np, (w_b, h_b), interpolation=cv2.INTER_LINEAR)
            writer_base.write(base_bgr_np)

            if wrist_bgr_np.shape[1] != w_w or wrist_bgr_np.shape[0] != h_w:
                wrist_bgr_np = cv2.resize(wrist_bgr_np, (w_w, h_w), interpolation=cv2.INTER_LINEAR)
            writer_wrist.write(wrist_bgr_np)

    finally:
        cap_base.release()
        cap_wrist.release()
        writer_base.release()
        writer_wrist.release()
