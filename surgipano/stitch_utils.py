# surgipano/stitch_utils.py
"""Helper library for the Stitch-seg panorama pipeline.

Originally this lived at ``Stitch-seg/demo/run_demo.py``; it was moved into
the SurgiPano package so that ``run_integrated.py`` can import from a stable
location regardless of where the ``third_party/Stitch-seg`` submodule is
checked out. The imperative ``main()`` from the original demo is gone — use
``python -m surgipano.run_integrated --enable_tapir 0`` for the baseline.

Exports:
    DemoCfg               - config object consumed by StitchInferencer
    PanoramaVideoWriter   - accumulating panorama -> fixed-size mp4
    load_scope_mask       - load + invert a scope mask PNG
    to_numpy_image        - tensor/np -> uint8 HWC
    crop_to_valid         - trim canvas to canvas_mask bbox
    project_points_to_canvas / project_corners_to_canvas
    draw_quad / draw_scope_outline / _put_label
    scope_contour
    _ZeroMask, _patch_out_masking_models, _masking_weights_available
        (legacy masking helpers; superseded by surgipano.masking_setup)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import contextlib
import io

# ---- package-relative paths ----
_HERE = Path(__file__).resolve().parent                       # SurgiPano/surgipano/
_ROOT = _HERE.parent                                          # SurgiPano/
_STITCH_REPO = _ROOT / "third_party" / "Stitch-seg"           # Stitch-seg submodule

# Allow `import stitch_seg` regardless of how this module is loaded.
if str(_STITCH_REPO) not in sys.path:
    sys.path.insert(0, str(_STITCH_REPO))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch


# ============================================================================
# Weights layout (relative to the Stitch-seg submodule)
# ============================================================================
ALIKED_WEIGHTS = "weights/aliked-n16.pth"
LIGHTGLUE_WEIGHTS = "weights/aliked_lightglue_v0-1_arxiv.pth"
TOOL_WEIGHTS: str | None = None
PORT_WEIGHTS: str | None = None


# ============================================================================
# Preview / overlay constants (mutated at runtime by run_integrated.py)
# ============================================================================
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 720
DRAW_ROI = True
DRAW_SCOPE = True


# ============================================================================
# Legacy masking stubs (kept for backward compat; prefer surgipano.masking_setup)
# ============================================================================
class _ZeroMask(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
        )


def _patch_out_masking_models() -> bool:
    import stitch_seg.inferencer as inferencer_mod
    inferencer_mod.load_masking_model = lambda cfg: _ZeroMask()
    inferencer_mod.load_masking_model2 = lambda cfg: _ZeroMask()
    return True


def _masking_weights_available(cfg) -> bool:
    for attr in ("tool_detector_weights", "port_detector_weights"):
        path = getattr(cfg, attr, None)
        if path and Path(path).is_file():
            return True
    return False


# ============================================================================
# Scope mask
# ============================================================================
def load_scope_mask(path: str, width: int, height: int) -> np.ndarray:
    """Load a scope mask PNG; return uint8 with 1 = outside the scope."""
    png = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if png is None:
        raise SystemExit(f"[error] Cannot read scope mask: {path}")
    if png.shape[:2] != (height, width):
        png = cv2.resize(png, (width, height), interpolation=cv2.INTER_NEAREST)
    inverted = (png <= 127).astype(np.uint8)
    inverted = cv2.dilate(inverted, np.ones((5, 5), np.uint8), iterations=2)
    return inverted


# ============================================================================
# Configuration object consumed by StitchInferencer
# ============================================================================
class DemoCfg:
    def __init__(self):
        self.method = "pyramid"
        self.bbox_mode = "internal"
        self.apply_ellipse_mask = True
        self.equalize_hist_rgb = False
        self.laplacian_var_min = 60
        self.canvas_scale_x = 3
        self.canvas_scale_y = 3
        self.canvas_superres_scale = 1.0
        self.alpha_overlap = 0.95
        self.seg_min_score = 0.3
        self.seg_max_coverage = 0.6
        self.tool_mask_dilate_px = 12
        self.output_mode = "logits"
        self.num_classes = 13
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.aliked_weights = str(_STITCH_REPO / ALIKED_WEIGHTS)
        self.lightglue_weights = str(_STITCH_REPO / LIGHTGLUE_WEIGHTS)
        self.tool_detector_weights = (
            str(_STITCH_REPO / TOOL_WEIGHTS) if TOOL_WEIGHTS else None
        )
        self.port_detector_weights = (
            str(_STITCH_REPO / PORT_WEIGHTS) if PORT_WEIGHTS else None
        )


# ============================================================================
# Helpers
# ============================================================================
def to_numpy_image(img) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        t = t.float()
        if t.max() <= 1.0:
            t = t * 255.0
        return t.clamp(0, 255).byte().numpy()
    if isinstance(img, np.ndarray) and img.dtype != np.uint8:
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    return img


def crop_to_valid(canvas: np.ndarray, canvas_mask: torch.Tensor | None):
    if canvas_mask is None:
        return canvas, (0, 0)
    mask = canvas_mask.detach().cpu()
    if mask.dim() == 4:
        mask = mask[0]
    if mask.dim() == 3:
        mask = mask[0]
    if mask.dim() != 2:
        return canvas, (0, 0)
    ys, xs = torch.nonzero(mask > 0, as_tuple=True)
    if ys.numel() == 0:
        return canvas, (0, 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y1 = min(y1, canvas.shape[0])
    x1 = min(x1, canvas.shape[1])
    return canvas[y0:y1, x0:x1], (x0, y0)


def project_points_to_canvas(inferencer, pts_np: np.ndarray) -> np.ndarray:
    H = inferencer._current_to_canvas_h().to(torch.float32)
    pts = np.asarray(pts_np, dtype=np.float32)
    if not torch.isfinite(H).all():
        return np.full_like(pts, np.nan, dtype=np.float32)
    device = H.device
    p = torch.from_numpy(pts).to(device)
    homo = torch.cat([p, torch.ones(len(p), 1, device=device)], dim=1)
    q = (H @ homo.T).T
    w = q[:, 2:3]
    w = torch.where(w.abs() < 1e-6, torch.full_like(w, float("nan")), w)
    return (q[:, :2] / w).cpu().numpy()


def project_corners_to_canvas(inferencer, w: int, h: int):
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    return project_points_to_canvas(inferencer, corners)


def draw_quad(image: np.ndarray, quad: np.ndarray, label: str = "") -> np.ndarray:
    out = image.copy()
    if quad is None:
        return out
    quad = np.asarray(quad, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return out
    h, w = out.shape[:2]
    lim = max(w, h) + 1_000_000
    quad = np.clip(quad, -lim, lim)
    poly = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], True, (0, 255, 255), 4, cv2.LINE_AA)
    if label:
        x = int(np.floor(quad[:, 0].min()))
        y = int(np.floor(quad[:, 1].min()))
        _put_label(out, label, (x + 10, max(26, y + 28)), (0, 255, 255))
    return out


def draw_scope_outline(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    out = image.copy()
    if contour is None or len(contour) < 2:
        return out
    c = np.asarray(contour, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        return out
    h, w = out.shape[:2]
    lim = max(w, h) + 1_000_000
    c = np.clip(c, -lim, lim)
    cv2.polylines(out, [np.round(c).astype(np.int32)], True, (0, 128, 255),
                  4, cv2.LINE_AA)
    return out


def _put_label(img: np.ndarray, text: str, org, color) -> None:
    ox, oy = int(org[0]), int(org[1])
    cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


def scope_contour(ellipse_mask: torch.Tensor | None):
    if ellipse_mask is None:
        return None
    m = ellipse_mask.detach().cpu()
    if m.dim() == 4:
        m = m[0]
    if m.dim() == 3:
        m = m[0]
    if m.dim() != 2:
        return None
    scope = (m == 0).numpy().astype(np.uint8)
    contours, _ = cv2.findContours(scope, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return biggest.reshape(-1, 2).astype(np.float32)


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily redirect sys.stdout to a StringIO.

    TAPIR's per-frame diagnostics use print() and would otherwise flood the
    progress bar. Matches the helper of the same name in
    ``surgipano/tapir_standalone.py``.
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

        
# ============================================================================
# Video writer
# ============================================================================
class PanoramaVideoWriter:
    """Writes the accumulating panorama (optionally side-by-side with the
    raw frame) to a fixed-size, letterboxed video."""

    def __init__(self, path: Path, size_wh, fps: float = 30.0):
        self.path = Path(path)
        self.size_wh = (int(size_wh[0]), int(size_wh[1]))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.writer = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0

    def _ensure_writer(self):
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, self.size_wh, True,
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {self.path}")
        return self.writer

    @staticmethod
    def _fit(img: np.ndarray, box_w: int, box_h: int):
        h, w = img.shape[:2]
        scale = min(box_w / max(w, 1), box_h / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR), nw, nh

    def write(self, canvas_bgr: np.ndarray,
              mask: torch.Tensor | None = None,
              side_frame_bgr: np.ndarray | None = None):
        if canvas_bgr is None or canvas_bgr.size == 0:
            return
        tw, th = self.size_wh
        out = np.zeros((th, tw, 3), dtype=np.uint8)

        if side_frame_bgr is not None:
            pan_w = (tw * 2) // 3
            src_w = tw - pan_w
            cropped, _ = crop_to_valid(canvas_bgr, mask)
            pan_img, nw, nh = self._fit(cropped, pan_w, th)
            out[(th - nh) // 2:(th - nh) // 2 + nh,
                (pan_w - nw) // 2:(pan_w - nw) // 2 + nw] = pan_img
            src_img, nw2, nh2 = self._fit(side_frame_bgr, src_w, th)
            out[(th - nh2) // 2:(th - nh2) // 2 + nh2,
                pan_w + (src_w - nw2) // 2:pan_w + (src_w - nw2) // 2 + nw2] = src_img
            cv2.line(out, (pan_w, 0), (pan_w, th), (200, 200, 200), 2)
            _put_label(out, "Panorama", (10, 30), (0, 255, 255))
            _put_label(out, "Original", (pan_w + 10, 30), (0, 255, 255))
        else:
            cropped, _ = crop_to_valid(canvas_bgr, mask)
            img, nw, nh = self._fit(cropped, tw, th)
            out[(th - nh) // 2:(th - nh) // 2 + nh,
                (tw - nw) // 2:(tw - nw) // 2 + nw] = img

        self._ensure_writer().write(out)
        self.frames_written += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

# ============================================================================
# GUI availability probe
# ============================================================================
_GUI_AVAILABLE: bool | None = None


def gui_available() -> bool:
    """Return True if cv2 has a working highgui backend.

    ``opencv-python-headless`` and some conda builds omit the highgui module
    and raise ``cv2.error: The function is not implemented`` on ``cv2.imshow``
    / ``cv2.namedWindow`` / ``cv2.destroyAllWindows``. Probe once and cache.
    """
    global _GUI_AVAILABLE
    if _GUI_AVAILABLE is not None:
        return _GUI_AVAILABLE
    try:
        cv2.namedWindow("__surgipano_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__surgipano_probe__")
        _GUI_AVAILABLE = True
    except cv2.error:
        _GUI_AVAILABLE = False
    return _GUI_AVAILABLE