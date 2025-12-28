import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from torchvision.transforms.functional import to_tensor
import hydra

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


class CutieVideoSegmenter:
    def __init__(self, max_size: int = 480):
        self.max_size = max_size

        self.model = get_default_model()
        self.processor = InferenceCore(self.model, cfg=self.model.cfg)
        self.processor.max_internal_size = self.max_size

        self.palette = None
        self.objects = []
        self.initial_mask = None

    def _read_initial_mask(self):
        mask_img = Image.open(self.mask_path)
        assert mask_img.mode in ['L', 'P'], f"Initial mask must be grayscale or palette-based"

        self.palette = mask_img.getpalette()
        mask_array = np.array(mask_img)
        present_ids = [int(obj) for obj in np.unique(mask_array) if obj != 0]

        # ---- choose which IDs to track ----
        if self.tracking_ids is None:
            self.objects = [i for i in present_ids if i != 0]
        else:
            wanted = set(self.tracking_ids)
            self.objects = [i for i in present_ids if i in wanted]
            if not self.objects:
                raise ValueError(f"No objects found in the mask that match the specified tracking IDs: {self.tracking_ids}")

            # zero-out pixels not in wanted IDs
            mask_array = np.where(np.isin(mask_array, self.objects), mask_array, 0)

        self.initial_mask = torch.from_numpy(mask_array).cuda()

    def _extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        ret, frame = cap.read()
        while ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

            ret, frame = cap.read()
        cap.release()
        return frames
    
    def get_nframes(self):
        cap = cv2.VideoCapture(self.video_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return nframes

    @torch.inference_mode()
    def run(self, video_path, mask_path, output_dir, tracking_ids=None):
        self.video_path = video_path
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.tracking_ids = tracking_ids
        os.makedirs(output_dir, exist_ok=True)

        #if self.get_nframes() == len(os.listdir(os.path.join(output_dir))):
        #    print(f"Skipping {video_path} as it has already been processed.")
        #    return

        self._read_initial_mask()
        frames = self._extract_frames()

        ti = 0
        video_name = os.path.dirname(video_path).split('/')[-1]
        for ti, frame in enumerate(tqdm(frames, desc=f"Processing {ti} {video_name}")):
            image_tensor = to_tensor(frame).cuda().float()

            if ti == 0:
                output_prob = self.processor.step(image_tensor, self.initial_mask, objects=self.objects)
            else:
                output_prob = self.processor.step(image_tensor)

            mask = self.processor.output_prob_to_mask(output_prob)
            mask_img = Image.fromarray(mask.cpu().numpy().astype(np.uint8), mode='P')
            mask_img.putpalette(self.palette)
            mask_img.save(os.path.join(self.output_dir, f'{ti:05d}_predmask.png'))


def parse_args():
    parser = argparse.ArgumentParser(description="Video segmentation with Cutie")
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to JSONL file with video name and mask ids to track')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing initial masks (PNG)')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing initial masks (PNG)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predicted masks')
    parser.add_argument('--max_size', type=int, default=-1, help='Resize shorter edge of image to this size. Default is -1 (no resizing)')

    return parser.parse_args()


def main():
    args = parse_args()
    for line in open(args.jsonl_path):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        segmenter = CutieVideoSegmenter(
            max_size=args.max_size,
        )
        data = eval(line.strip())
        initial_img_path = data['image']
        video_name = os.path.splitext(os.path.basename(initial_img_path))[0]
        track_ids = [id for ids in data['parsed_answer'].values() for id in ids]
        if not track_ids:
            print(f"No valid tracking IDs found for {video_name}. Skipping.")
            continue

        video_path = os.path.join(args.input_dir, f"{video_name}/camera_base.mp4")
        mask_path = os.path.join(args.mask_dir, f"{video_name}.png")
        output_dir = os.path.join(args.output_dir, video_name)

        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist. Skipping.")
            continue

        if not os.path.exists(mask_path):
            print(f"Mask file {mask_path} does not exist. Skipping.")
            continue

        segmenter.run(video_path=video_path,
                      mask_path=mask_path,
                      output_dir=output_dir,
                      tracking_ids=track_ids)

if __name__ == '__main__':
    main()
