"""Common utilities for Stitch / TAPIR experiments."""
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch


def image_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 4: x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3): x = x.permute(1, 2, 0)
        x = x.numpy()
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.size and x.max() <= 1: x *= 255
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def normalize_h(H):
    if H is None: return None
    if isinstance(H, torch.Tensor): H = H.detach().cpu().numpy()
    H = np.asarray(H, np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10: return None
    if abs(H[2, 2]) > 1e-10: H = H / H[2, 2]
    return H


def transform(H, points):
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def valid_h(H, width, height):
    H = normalize_h(H)
    if H is None: return False
    c = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    try:
        q = transform(H, c)
        if not np.isfinite(q).all(): return False
        area = abs(cv2.contourArea(q.astype(np.float32)))
        if area < width * height * 0.02 or area > width * height * 50: return False
        return True
    except cv2.error: return False


def estimate_h(prev_pts, curr_pts, width, height, threshold=3.0):
    """Return current->previous homography from TAPIR (prev,curr) point pairs."""
    if len(prev_pts) < 4: return None, 0, float('nan')
    H, mask = cv2.findHomography(np.asarray(curr_pts, np.float32), np.asarray(prev_pts, np.float32),
                                  cv2.RANSAC, threshold)
    if not valid_h(H, width, height): return None, 0, float('nan')
    ok = mask.ravel().astype(bool) if mask is not None else np.ones(len(curr_pts), bool)
    errors = np.linalg.norm(transform(H, curr_pts) - prev_pts, axis=1)
    return normalize_h(H), int(ok.sum()), float(np.median(errors[ok])) if ok.any() else float('nan')


def stitch_error(H, prev_H, prev_pts, curr_pts):
    if H is None or prev_H is None or not len(curr_pts): return float('nan')
    try:
        p = transform(prev_H, prev_pts)
        q = transform(H, curr_pts)
        return float(np.median(np.linalg.norm(p - q, axis=1)))
    except (cv2.error, ValueError): return float('nan')


def canvas_init(width, height, scale=3):
    cw, ch = int(width * scale), int(height * scale)
    offset = np.array([[1, 0, (cw - width) / 2], [0, 1, (ch - height) / 2], [0, 0, 1]], np.float64)
    return np.zeros((ch, cw, 3), np.uint8), np.zeros((ch, cw), np.uint8), offset


def compose(canvas, coverage, frame_rgb, H, mask=None, alpha=0.9):
    h, w = frame_rgb.shape[:2]
    if mask is None: mask = np.full((h, w), 255, np.uint8)
    warped = cv2.warpPerspective(frame_rgb, H, (canvas.shape[1], canvas.shape[0]))
    valid = cv2.warpPerspective(mask, H, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST) > 0
    fresh = valid & (coverage == 0)
    overlap = valid & (coverage > 0)
    canvas[fresh] = warped[fresh]
    canvas[overlap] = np.clip(alpha * canvas[overlap].astype(np.float32) +
                              (1-alpha) * warped[overlap].astype(np.float32), 0, 255).astype(np.uint8)
    coverage[valid] = 255
    return int(valid.sum())


def mask_for_frame(frame, scope_mask_path=None, margin=12):
    h, w = frame.shape[:2]
    if scope_mask_path:
        m = cv2.imread(str(scope_mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None: raise FileNotFoundError(scope_mask_path)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = np.uint8(m > 127) * 255
    else:
        m = np.full((h, w), 255, np.uint8)
    if margin:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1))
        m = cv2.erode(m, k)
    return m


def save_rgb(path, arr):
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def write_metrics(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def make_stitch(args):
    from stitch_seg import StitchInferencer
    from stitch_seg.config import apply_stitch_defaults
    class Config: pass
    cfg = Config()
    for key, val in dict(method='pyramid', bbox_mode='internal', apply_ellipse_mask=True,
                         equalize_hist_rgb=False, laplacian_var_min=60,
                         canvas_scale_x=args.canvas_scale, canvas_scale_y=args.canvas_scale,
                         canvas_superres_scale=1.0, alpha_overlap=0.95,
                         seg_min_score=0.3, seg_max_coverage=0.6,
                         tool_mask_dilate_px=12, output_mode='logits', num_classes=13,
                         device=torch.device(args.device),
                         aliked_weights=str(Path(args.aliked_weights).resolve()),
                         lightglue_weights=str(Path(args.lightglue_weights).resolve()),
                         tool_detector_weights=None, port_detector_weights=None).items(): setattr(cfg, key, val)
    cfg = apply_stitch_defaults(cfg)
    # Mirrors the uploaded run_demo.py's missing-detector behavior.
    import stitch_seg.inferencer as mod
    class ZeroMask(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device, dtype=x.dtype)
    mod.load_masking_model = lambda _cfg: ZeroMask()
    mod.load_masking_model2 = lambda _cfg: ZeroMask()
    return StitchInferencer(None, cfg=cfg)


def frame_iterator(path, width, height, max_frames=None, stride=1):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0; yielded = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or (max_frames is not None and yielded >= max_frames): break
            if index % stride == 0:
                bgr = cv2.resize(bgr, (width, height))
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fps
                yielded += 1
            index += 1
    finally: cap.release()


def add_arguments(parser):
    parser.add_argument('--video', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--scope-mask', default=None)
    parser.add_argument('--aliked-weights', default='weights/aliked-n16.pth')
    parser.add_argument('--lightglue-weights', default='weights/aliked_lightglue_v0-1_arxiv.pth')
    parser.add_argument('--canvas-scale', type=int, default=3)
    parser.add_argument('--save-every', type=int, default=100)


def expected_frame_count(video, max_frames=None, stride=1):
    """Number of processed frames, or None when source length is unknown."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n <= 0:
        return max_frames
    count = (n + max(1, stride) - 1) // max(1, stride)
    return min(count, max_frames) if max_frames is not None else count
