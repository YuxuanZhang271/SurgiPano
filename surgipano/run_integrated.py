# surgipano/run_integrated.py
from __future__ import annotations

import argparse, os, sys, time, warnings
from pathlib import Path

# ---- path bootstrap: works for `python -m surgipano.run_integrated`
#      and `python surgipano/run_integrated.py`. ----
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_THIRD = _ROOT / "third_party"
for p in (_ROOT, _THIRD, _THIRD / "tapnet", _THIRD / "Stitch-seg"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import tree
from tqdm import tqdm

from stitch_seg import StitchInferencer
from stitch_seg.config import apply_stitch_defaults
from tapnet.torch import tapir_model
from surgipano.masking_setup import setup_masking
from surgipano import stitch_utils as sd


# ============================================================================
# TAPIR wrapper
# ============================================================================
class TapirTracker:
    """Causal TAPIR tracker over an arbitrary frame stream."""

    def __init__(self, checkpoint: str, device: torch.device,
                 grid_spacing: int = 64, model_hw: int = 256,
                 visibility: str = "occlusion"):
        self.device = device
        self.grid_spacing = grid_spacing
        self.model_hw = model_hw
        self.visibility = visibility
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = self.model.to(device).eval()
        torch.set_grad_enabled(False)
        self.query_features = None
        self.causal_state = None
        self.grid_points = None
        self.scale_x = self.scale_y = 1.0
        self.frame_shape = None

    def _to_model_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.model_hw and frame.shape[0] != self.model_hw:
            return cv2.resize(frame, (self.model_hw, self.model_hw),
                              interpolation=cv2.INTER_AREA)
        return frame

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(frames).float()
        return (t / 255.0 * 2 - 1).to(self.device)

    def anchor(self, frame: np.ndarray, mask_eroded: np.ndarray | None = None):
        self.frame_shape = frame.shape[:2]
        h, w = self.frame_shape
        self.scale_y = self.model_hw / h
        self.scale_x = self.model_hw / w

        pts = []
        y = self.grid_spacing / 2.0
        while y < h:
            x = self.grid_spacing / 2.0
            while x < w:
                yi, xi = int(y), int(x)
                if mask_eroded is None or mask_eroded[yi, xi] > 127:
                    pts.append((0.0, float(y), float(x)))
                x += self.grid_spacing
            y += self.grid_spacing
        if not pts:
            raise RuntimeError("TAPIR anchor produced no points.")
        self.grid_points = pts

        qp = np.array(pts, dtype=np.float32)
        qp[:, 1] *= self.scale_y
        qp[:, 2] *= self.scale_x
        qp_t = torch.from_numpy(qp).to(self.device)

        anchor_t = self._preprocess(self._to_model_frame(frame))[None, None]
        with sd._suppress_stdout():
            self.query_features = self.model.get_query_features(
                anchor_t, is_training=False, query_points=qp_t[None],
                feature_grids=self.model.get_feature_grids(
                    anchor_t, is_training=False))
        self.causal_state = self.model.construct_initial_causal_state(
            len(pts), len(self.query_features.resolutions) - 1)
        self.causal_state = tree.map_structure(
            lambda x: x.to(self.device), self.causal_state)
        return pts

    def step(self, frame: np.ndarray):
        if self.query_features is None:
            raise RuntimeError("Call .anchor(frame) before .step(frame).")
        ft = self._preprocess(self._to_model_frame(frame))[None, None]
        with sd._suppress_stdout():
            traj = self.model.estimate_trajectories(
                ft.shape[-3:-1], is_training=False,
                feature_grids=self.model.get_feature_grids(ft, is_training=False),
                query_features=self.query_features,
                query_points_in_video=None, query_chunk_size=64,
                causal_context=self.causal_state, get_causal_context=True)
        self.causal_state = traj["causal_context"]
        tracks = traj["tracks"][-1].cpu().numpy()
        occ    = traj["occlusion"][-1].cpu().numpy()
        unc    = traj["expected_dist"][-1].cpu().numpy()

        xy = tracks[0, :, 0, :].copy()
        xy[:, 0] /= self.scale_x
        xy[:, 1] /= self.scale_y
        if self.visibility == "all":
            vis = np.ones(len(xy), bool)
        elif self.visibility == "occlusion":
            vis = (1 - 1/(1+np.exp(-occ[0, :, 0]))) > 0.5
        else:
            vis = ((1 - 1/(1+np.exp(-occ[0, :, 0]))) *
                   (1 - 1/(1+np.exp(-unc[0, :, 0])))) > 0.5
        ref = np.array([[p[2], p[1]] for p in self.grid_points])
        diag = np.hypot(*self.frame_shape)
        keep = np.linalg.norm(xy - ref, axis=1) < 0.4 * diag
        return xy[keep & vis], keep & vis


# ============================================================================
# Helpers
# ============================================================================
def _parse_ids(s: str | None) -> list[int] | None:
    return [int(v) for v in s.split(",")] if s else None


# ============================================================================
# Main loop
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="SurgiPano: Stitch-seg panorama with optional TAPIR hints "
                    "and YOLO tool/port masking.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--scope-mask", default=None)
    ap.add_argument("--output-dir", default=str(_ROOT / "data" / "output"))
    # ---- TAPIR ----
    ap.add_argument("--enable_tapir", type=int, choices=[0, 1], default=0,
                    help="0 = Stitch-seg only; 1 = run TAPIR hints alongside.")
    ap.add_argument("--tapir_checkpoint",
                    default=str(_THIRD / "tapnet" / "checkpoints" /
                                "causal_bootstapir_checkpoint.pt"))
    ap.add_argument("--tapir_grid_spacing", type=int, default=64)
    ap.add_argument("--tapir_model_hw", type=int, default=256)
    ap.add_argument("--tapir_visibility",
                    choices=["full", "occlusion", "all"], default="occlusion")
    ap.add_argument("--tapir_anchor_every", type=int, default=300,
                    help="Re-anchor TAPIR every N frames (0 = never).")
    ap.add_argument("--tapir_disagree_px", type=float, default=25.0)
    ap.add_argument("--tapir_reset_on_static", type=int, default=0)
    # ---- YOLO segmentation ----
    ap.add_argument("--yolo_tool", default=None,
                    help="YOLO-seg weights for tool masking (None -> zero stub).")
    ap.add_argument("--yolo_port", default=None,
                    help="YOLO-seg weights for port masking (None -> zero stub).")
    ap.add_argument("--yolo_tool_classes", default=None,
                    help="Comma-separated class IDs kept from --yolo_tool.")
    ap.add_argument("--yolo_port_classes", default=None,
                    help="Comma-separated class IDs kept from --yolo_port.")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    # ---- stitcher options ----
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--no_display", action="store_true")
    args = ap.parse_args()

    # ---- headless fallback: some opencv builds lack highgui entirely ----
    if not args.no_display and not sd.gui_available():
        print("[warn] OpenCV has no GUI backend "
              "(opencv-python-headless or a build without GTK/Qt). "
              "Forcing --no_display. Outputs (mp4 / PNG) still written.")
        args.no_display = True

    warnings.filterwarnings(
        "ignore", message=".*output size is not a multiple of 8.*")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd.PREVIEW_WIDTH, sd.PREVIEW_HEIGHT = args.width, args.height

    # ---- stitcher config ----
    cfg = apply_stitch_defaults(sd.DemoCfg())

    # ---- segmentation (YOLO or zero-mask stub) ----
    mask_info = setup_masking(
        cfg,
        yolo_tool_weights=args.yolo_tool,
        yolo_port_weights=args.yolo_port,
        yolo_tool_classes=_parse_ids(args.yolo_tool_classes),
        yolo_port_classes=_parse_ids(args.yolo_port_classes),
        yolo_conf=args.yolo_conf,
    )
    print(f"[info] masking   : tool={mask_info['tool']}, port={mask_info['port']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inferencer = StitchInferencer(None, cfg=cfg)

    # ---- scope mask ----
    scope_inv = None
    if args.scope_mask:
        scope_inv = sd.load_scope_mask(
            str(Path(args.scope_mask).expanduser()), args.width, args.height)
        print(f"[info] scope mask : {args.scope_mask}")

    # ---- TAPIR ----
    tapir = None
    if args.enable_tapir:
        print("[info] TAPIR enabled — loading checkpoint …")
        tapir = TapirTracker(
            args.tapir_checkpoint, device,
            grid_spacing=args.tapir_grid_spacing,
            model_hw=args.tapir_model_hw,
            visibility=args.tapir_visibility)
        print(f"[info] TAPIR grid  : {tapir.grid_spacing}px, "
              f"model {tapir.model_hw}², anchor every {args.tapir_anchor_every}")
    else:
        print("[info] TAPIR disabled — running pure Stitch-seg baseline.")

    cap = cv2.VideoCapture(str(Path(args.video).expanduser()))
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames is not None:
        total = min(total, args.max_frames)

    tag = ("tapir" if tapir else "base") + ("+yolo" if (
        args.yolo_tool or args.yolo_port) else "")
    preview = sd.PanoramaVideoWriter(
        out_dir / f"panorama_preview_{tag}.mp4",
        (args.width, args.height), fps=src_fps)
    png_every = max(1, int(round(src_fps * 30)))
    png_dir = out_dir / "panoramas"
    png_dir.mkdir(exist_ok=True)

    mask_injected = scope_inv is None
    scope_pts = None
    frame_idx = 0
    tapir_anchor_at = 0
    n_tapir_frames = 0
    tapir_total_ms = 0.0
    pbar = tqdm(total=total, desc=f"Stitching [{tag}]",
                unit="frame", dynamic_ncols=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height),
                                   interpolation=cv2.INTER_LINEAR)
            frame_u = inferencer.preprocess_frame(frame)

            # ---- scope mask injection (once) ----
            if not mask_injected:
                rx, ry, rw, rh = [int(v) for v in inferencer.roi]
                m_crop = scope_inv[ry:ry+rh, rx:rx+rw]
                if tuple(m_crop.shape) != tuple(frame_u.shape[-2:]):
                    raise SystemExit(
                        f"[error] scope crop {m_crop.shape} != frame_u "
                        f"{tuple(frame_u.shape[-2:])}")
                inferencer.ellipse_mask = torch.from_numpy(
                    np.ascontiguousarray(m_crop)).to(cfg.device)[None, None]
                mask_injected = True
                scope_pts = sd.scope_contour(inferencer.ellipse_mask)

            # ---- TAPIR branch ----
            tapir_hint = None
            if tapir is not None:
                if tapir.query_features is None or (
                        args.tapir_anchor_every > 0 and
                        frame_idx - tapir_anchor_at >= args.tapir_anchor_every):
                    mask_eroded = None
                    if scope_inv is not None:
                        mask_eroded = (1 - scope_inv) * 255
                    tapir.anchor(frame, mask_eroded)
                    tapir_anchor_at = frame_idx
                t0 = time.time()
                xy, vis = tapir.step(frame)
                tapir_total_ms += (time.time() - t0) * 1000
                n_tapir_frames += 1
                if len(xy) >= 4:
                    tapir_hint = float(np.median(xy[:, 0]))

            # ---- stitcher step (alignment) ----
            t_stitch = time.time()
            inferencer.step_canvas(frame_u)
            stitch_ms = (time.time() - t_stitch) * 1000

            # ---- optional reset heuristic ----
            if (tapir is not None and tapir_hint is not None
                    and args.tapir_reset_on_static):
                H = inferencer._current_to_canvas_h().to(torch.float32)
                if torch.isfinite(H).all():
                    denom = float(H[2, 2])
                    tx = float(H[0, 2] / denom) if abs(denom) > 1e-9 else 0.0
                    if abs(tx) > args.tapir_disagree_px and abs(tapir_hint) < 1.0:
                        print(f"[warn] frame {frame_idx}: homography large "
                              f"(tx={tx:.1f}) but TAPIR motion ~0 -> reset hint")
                        inferencer.reset_state(clear_preprocess_state=False)

            # ---- preview / PNG ----
            canvas_np = sd.to_numpy_image(inferencer.canvas)
            canvas_bgr = None
            if canvas_np is not None:
                canvas_bgr = (cv2.cvtColor(canvas_np, cv2.COLOR_RGB2BGR)
                              if canvas_np.ndim == 3 and canvas_np.shape[2] == 3
                              else canvas_np)
                if sd.DRAW_ROI:
                    fh, fw = frame_u.shape[-2:]
                    quad = sd.project_corners_to_canvas(inferencer, fw, fh)
                    if np.all(np.isfinite(quad)):
                        canvas_bgr = sd.draw_quad(
                            canvas_bgr, quad, f"ROI frame {frame_idx}")
                if sd.DRAW_SCOPE and scope_pts is not None:
                    canvas_bgr = sd.draw_scope_outline(
                        canvas_bgr,
                        sd.project_points_to_canvas(inferencer, scope_pts))
                if tapir is not None and tapir_hint is not None:
                    sd._put_label(canvas_bgr,
                                  f"TAPIR tracks: {n_tapir_frames}",
                                  (10, canvas_bgr.shape[0] - 20),
                                  (0, 255, 0))
                preview.write(canvas_bgr, inferencer.canvas_mask, frame)
                if frame_idx % png_every == 0:
                    cv2.imwrite(str(png_dir / f"panorama_{frame_idx:06d}.png"),
                                canvas_bgr)

            if not args.no_display and canvas_bgr is not None:
                try:
                    cv2.imshow("SurgiPano", canvas_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except cv2.error as exc:
                    print(f"[warn] cv2.imshow failed ({exc}); "
                          f"continuing headless.")
                    args.no_display = True

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix_str(f"stitch {stitch_ms:.1f}ms")
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
    finally:
        pbar.close()
        cap.release()
        preview.close()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    summary = {
        "frames": frame_idx,
        "tapir_enabled": bool(args.enable_tapir),
        "tapir_avg_ms": (tapir_total_ms / n_tapir_frames) if n_tapir_frames else 0.0,
        "tapir_frames": n_tapir_frames,
        "masking": mask_info,
        "preview": str(preview.path),
    }
    print("[done]", summary)
    (out_dir / "run_summary.json").write_text(
        __import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()