import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, Union

from torchvision.transforms.functional import to_tensor

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


class CutieStreamTracker:
    def __init__(self, max_size: int = 480):
        self.max_size = max_size
        self.been_init = False

        self.model = get_default_model()
        self.processor = InferenceCore(self.model, cfg=self.model.cfg)
        self.processor.max_internal_size = self.max_size

    def _process_initial_mask(
            self,
            mask_img: Union[Image.Image, np.ndarray],
            track_objs: Optional[Dict[str, list]] = None):
        self.reset()

        if isinstance(mask_img, Image.Image):
            assert mask_img.mode in ['L', 'P'], "Initial mask must be grayscale or palette-based"
            mask_array = np.array(mask_img)
        mask_array = np.array(mask_img)

        # ---- choose which IDs to track ----
        if track_objs is None:
            present_ids = [int(obj) for obj in np.unique(mask_array) if obj != 0]
            obj_ids = [i for i in present_ids if i != 0]
            out = mask_array.copy()
        else:
            out = np.zeros_like(mask_array, dtype=mask_array.dtype)
            obj_ids = []
            label = 1

            for name, ids in track_objs.items():
                if not ids:
                    continue
                grp = np.isin(mask_array, ids)
                if grp.any():
                    out[grp] = label
                    obj_ids.append(label)
                    label += 1

            if not obj_ids:
                raise ValueError("No objects found in the mask that match the specified tracking IDs")

        self.been_init = True
        return torch.from_numpy(out).cuda(), obj_ids

    def reset(self):
        """Hard reset of memory."""
        if self.been_init:
            self.processor = InferenceCore(self.model, cfg=self.model.cfg)
            self.processor.max_internal_size = self.max_size
            self.been_init = False

    @torch.inference_mode()
    def run_single_frame(self,
                         rgb_img: Image.Image,
                         init_mask_img: Optional[Union[Image.Image, np.ndarray]] = None,
                         track_objs: Optional[Dict[str, list[int]]] = None):
        image_tensor = to_tensor(rgb_img).cuda().float()
        if init_mask_img is not None:
            initial_mask_tensor, obj_ids = self._process_initial_mask(init_mask_img, track_objs)
            output_prob = self.processor.step(image_tensor, initial_mask_tensor, objects=obj_ids)
        else:
            output_prob = self.processor.step(image_tensor)

        mask = self.processor.output_prob_to_mask(output_prob).cpu().numpy().astype(np.uint8)
        return mask

