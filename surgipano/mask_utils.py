"""Endoscope mask utilities: white=valid, black=excluded.

Input mask is defined in resized full-video coordinates, then cropped using
StitchInferencer.roi so it aligns with preprocess_frame() output.
"""
from pathlib import Path
import cv2
import numpy as np
import torch

class ScopeMask:
    def __init__(self, path, input_width, input_height, margin=12):
        self.path = path
        self.margin = max(0, int(margin))
        self.full = None
        if path:
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f'Cannot load scope mask: {path}')
            self.full = cv2.resize(m, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
            self.full = np.uint8(self.full > 127) * 255

    def crop(self, stitch, frame_shape):
        h, w = frame_shape[:2]
        if self.full is None:
            return np.full((h, w), 255, np.uint8)
        x, y, rw, rh = (int(v) for v in stitch.roi)
        if x < 0 or y < 0 or x+rw > self.full.shape[1] or y+rh > self.full.shape[0]:
            raise ValueError(f'ROI {stitch.roi} outside mask size {self.full.shape}')
        cropped = self.full[y:y+rh, x:x+rw]
        if cropped.shape != (h,w):
            raise ValueError(f'Mask crop {cropped.shape} != preprocessed frame {(h,w)}')
        return cropped.copy()

    def eroded(self, mask):
        if not self.margin:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.margin+1, 2*self.margin+1))
        return cv2.erode(mask, k)

    def inject_stitch(self, stitch, valid_mask, device):
        # Stitch's ellipse_mask convention is 1 outside the valid scope.
        outside = (valid_mask == 0).astype(np.uint8)
        stitch.ellipse_mask = torch.from_numpy(np.ascontiguousarray(outside)).to(device).unsqueeze(0).unsqueeze(0)
