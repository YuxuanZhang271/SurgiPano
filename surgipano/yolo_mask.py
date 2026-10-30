# surgipano/yolo_mask.py
"""YOLO segmentation mask that plugs into Stitch-seg's masking API.

Contract (identical to stitch_seg.inferencer.load_masking_model/_2):
    forward(x: (N, C, H, W)) -> (N, 1, H, W)
    output values: 1.0 = mask out (don't blend), 0.0 = keep
The stitcher feeds its preprocessed frame tensor; range is auto-detected.
"""
from __future__ import annotations
import cv2
import numpy as np
import torch


class YOLOMask(torch.nn.Module):
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.45,
        class_ids: list[int] | None = None,
        dilate_px: int = 12,
        input_range: str = "auto",       # auto | unit | signed | imagenet | raw
        input_channels: str = "rgb",
        invert: bool = False,
    ):
        super().__init__()
        from ultralytics import YOLO
        self.yolo = YOLO(weights)
        self.device = device
        self.conf, self.iou = conf, iou
        self.class_ids = class_ids
        self.dilate_px = int(dilate_px)
        self.input_range = input_range
        self.input_channels = input_channels
        self.invert = invert
        self._kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2*self.dilate_px+1, 2*self.dilate_px+1))
            if self.dilate_px > 0 else None
        )

    def _denormalize(self, imgs: np.ndarray) -> np.ndarray:
        rng = self.input_range
        if rng == "auto":
            lo, hi = float(imgs.min()), float(imgs.max())
            rng = "signed" if lo < -0.5 else ("unit" if hi <= 1.01 else "raw")
        if rng == "signed":
            imgs = (imgs + 1.0) / 2.0
        elif rng == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
            std  = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
            imgs = imgs * std + mean
        return np.clip(imgs * 255.0, 0, 255).astype(np.uint8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        arr = self._denormalize(x.detach().cpu().float().numpy())
        out = torch.zeros(N, 1, H, W, dtype=torch.float32)
        for i in range(N):
            hwc = arr[i].transpose(1, 2, 0)
            bgr = (cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
                   if C == 3 and self.input_channels == "rgb" else hwc)
            r = self.yolo.predict(
                bgr, conf=self.conf, iou=self.iou, classes=self.class_ids,
                verbose=False, device=self.device,
            )[0]
            if r.masks is None or len(r.masks) == 0:
                continue
            mask = np.zeros((H, W), dtype=np.uint8)
            for poly in r.masks.xy:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            if self._kernel is not None:
                mask = cv2.dilate(mask, self._kernel)
            out[i, 0] = torch.from_numpy(mask).float()
        if self.invert:
            out = 1.0 - out
        return out.to(device=x.device, dtype=x.dtype)