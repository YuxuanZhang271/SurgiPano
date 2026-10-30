# surgipano/evaluate.py
from __future__ import annotations

import argparse, json, time
from pathlib import Path

# ---- path bootstrap (same as run_integrated.py) ----
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_THIRD = _ROOT / "third_party"
for p in (_ROOT, _THIRD, _THIRD / "tapnet", _THIRD / "Stitch-seg"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import numpy as np
import torch

from stitch_seg import StitchInferencer
from stitch_seg.config import apply_stitch_defaults
from surgipano.masking_setup import setup_masking
from surgipano.run_integrated import TapirTracker
from surgipano import stitch_utils as sd


def _parse_ids(s: str | None) -> list[int] | None:
    return [int(v) for v in s.split(",")] if s else None


def _run_one(video, scope_mask, width, height,
             enable_tapir, enable_yolo, tapir_ckpt,
             yolo_tool, yolo_port, yolo_tool_classes, yolo_port_classes,
             yolo_conf, max_frames, device):
    """Return a metrics dict for one configuration."""
    cfg = apply_stitch_defaults(sd.DemoCfg())
    mask_info = setup_masking(
        cfg,
        yolo_tool_weights=(yolo_tool if enable_yolo else None),
        yolo_port_weights=(yolo_port if enable_yolo else None),
        yolo_tool_classes=(yolo_tool_classes if enable_yolo else None),
        yolo_port_classes=(yolo_port_classes if enable_yolo else None),
        yolo_conf=yolo_conf,
    )
    inferencer = StitchInferencer(None, cfg=cfg)

    scope_inv = None
    if scope_mask:
        scope_inv = sd.load_scope_mask(scope_mask, width, height)

    tapir = None
    if enable_tapir:
        tapir = TapirTracker(tapir_ckpt, device, grid_spacing=64,
                             model_hw=256, visibility="occlusion")

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    stitch_ms, tapir_ms, yolo_ms = [], [], []
    reset_count = 0
    orig_reset = inferencer.reset_state

    def counted_reset(clear_preprocess_state=False):
        nonlocal reset_count
        reset_count += 1
        return orig_reset(clear_preprocess_state=clear_preprocess_state)

    inferencer.reset_state = counted_reset

    # Wrap the masking module so we can time it per frame (YOLO runs inside).
    if enable_yolo:
        _orig_step_canvas = inferencer.step_canvas

        def timed_step_canvas(frame_u):
            t = time.time()
            _orig_step_canvas(frame_u)
            yolo_ms.append((time.time() - t) * 1000)

        inferencer.step_canvas = timed_step_canvas

    mask_injected = scope_inv is None
    frame_idx = 0
    t_wall = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        frame_u = inferencer.preprocess_frame(frame)

        if not mask_injected:
            rx, ry, rw, rh = [int(v) for v in inferencer.roi]
            m_crop = scope_inv[ry:ry+rh, rx:rx+rw]
            inferencer.ellipse_mask = torch.from_numpy(
                np.ascontiguousarray(m_crop)).to(cfg.device)[None, None]
            mask_injected = True

        if tapir is not None:
            if tapir.query_features is None or frame_idx % 300 == 0:
                tapir.anchor(frame)
            t1 = time.time()
            tapir.step(frame)
            tapir_ms.append((time.time() - t1) * 1000)

        t0 = time.time()
        inferencer.step_canvas(frame_u)
        if not enable_yolo:
            stitch_ms.append((time.time() - t0) * 1000)

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
    cap.release()

    canvas_np = sd.to_numpy_image(inferencer.canvas)
    return {
        "config": ("tapir" if enable_tapir else "baseline")
                  + ("+yolo" if enable_yolo else ""),
        "tapir_enabled": bool(enable_tapir),
        "yolo_enabled": bool(enable_yolo),
        "masking": mask_info,
        "frames": frame_idx,
        "wall_s": round(time.time() - t_wall, 2),
        "stitch_ms_mean": round(float(np.mean(stitch_ms)), 2) if stitch_ms else None,
        "stitch_ms_p95": round(float(np.percentile(stitch_ms, 95)), 2) if stitch_ms else None,
        "yolo_step_ms_mean": round(float(np.mean(yolo_ms)), 2) if yolo_ms else None,
        "yolo_step_ms_p95": round(float(np.percentile(yolo_ms, 95)), 2) if yolo_ms else None,
        "tapir_ms_mean": round(float(np.mean(tapir_ms)), 2) if tapir_ms else None,
        "tapir_ms_p95": round(float(np.percentile(tapir_ms, 95)), 2) if tapir_ms else None,
        "reset_count": reset_count,
        "canvas_hw": list(canvas_np.shape[:2]) if canvas_np is not None else None,
        "roi_xywh": list(inferencer.roi),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scope-mask", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=600)
    ap.add_argument("--tapir_checkpoint",
                    default=str(_THIRD / "tapnet" / "checkpoints" /
                                "causal_bootstapir_checkpoint.pt"))
    ap.add_argument("--yolo_tool", default=None)
    ap.add_argument("--yolo_port", default=None)
    ap.add_argument("--yolo_tool_classes", default=None)
    ap.add_argument("--yolo_port_classes", default=None)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(_ROOT / "data" / "eval.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which configs to run.
    yolo_available = (
        (args.yolo_tool and Path(args.yolo_tool).is_file()) or
        (args.yolo_port and Path(args.yolo_port).is_file())
    )
    configs = [("baseline", False, False), ("tapir", True, False)]
    if yolo_available:
        configs += [("yolo", False, True), ("tapir+yolo", True, True)]
    else:
        print("[info] no YOLO weights provided -> running baseline vs tapir only")

    yolo_tool_classes = _parse_ids(args.yolo_tool_classes)
    yolo_port_classes = _parse_ids(args.yolo_port_classes)

    results = []
    for name, tapir_on, yolo_on in configs:
        print(f"\n=== {name} ===")
        r = _run_one(
            args.video, args.scope_mask, args.width, args.height,
            tapir_on, yolo_on, args.tapir_checkpoint,
            args.yolo_tool, args.yolo_port,
            yolo_tool_classes, yolo_port_classes, args.yolo_conf,
            args.max_frames, device)
        r["config"] = name
        results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n=== summary ===")
    hdr = f"{'config':12s} {'frames':>6s} {'stitch':>8s} {'tapir':>8s} {'yolo':>8s} {'resets':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['config']:12s} {r['frames']:6d} "
              f"{str(r['stitch_ms_mean']):>8s} "
              f"{str(r['tapir_ms_mean']):>8s} "
              f"{str(r['yolo_step_ms_mean']):>8s} "
              f"{r['reset_count']:6d}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()