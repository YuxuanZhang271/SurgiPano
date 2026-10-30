# surgipano/masking_setup.py
from pathlib import Path
import torch
import stitch_seg.inferencer as inferencer_mod
from surgipano.yolo_mask import YOLOMask


class ZeroMask(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1],
                           device=x.device, dtype=x.dtype)


def setup_masking(cfg,
                  yolo_tool_weights: str | None = None,
                  yolo_port_weights: str | None = None,
                  yolo_tool_classes: list[int] | None = None,
                  yolo_port_classes: list[int] | None = None,
                  yolo_conf: float = 0.25,
                  dilate_px: int | None = None) -> dict:
    """Patch Stitch-seg's masking model loaders. Idempotent; safe to call once."""
    if dilate_px is None:
        dilate_px = getattr(cfg, "tool_mask_dilate_px", 12)
    info = {"tool": "zero", "port": "zero"}

    if yolo_tool_weights and Path(yolo_tool_weights).is_file():
        m = YOLOMask(yolo_tool_weights, device=str(cfg.device),
                     conf=yolo_conf, class_ids=yolo_tool_classes,
                     dilate_px=dilate_px)
        inferencer_mod.load_masking_model = lambda _: m
        info["tool"] = f"yolo:{yolo_tool_weights}"
    else:
        inferencer_mod.load_masking_model = lambda _: ZeroMask()
        if yolo_tool_weights:
            info["tool"] = f"missing:{yolo_tool_weights}"

    if yolo_port_weights and Path(yolo_port_weights).is_file():
        m2 = YOLOMask(yolo_port_weights, device=str(cfg.device),
                      conf=yolo_conf, class_ids=yolo_port_classes,
                      dilate_px=dilate_px)
        inferencer_mod.load_masking_model2 = lambda _: m2
        info["port"] = f"yolo:{yolo_port_weights}"
    else:
        inferencer_mod.load_masking_model2 = lambda _: ZeroMask()
        if yolo_port_weights:
            info["port"] = f"missing:{yolo_port_weights}"
    return info