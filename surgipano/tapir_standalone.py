# surgipano/tapir_standalone.py
"""Standalone TAPIR-on-video demo (was tapnet/tapnet/pytorch_video.py).

Reference entry point for the TAPIR causal pipeline. For the integrated
stitching pipeline, use ``python -m surgipano.run_integrated`` instead.

Usage:
    python -m surgipano.tapir_standalone \
        --video /path/to/video.mp4 \
        --checkpoint third_party/tapnet/checkpoints/causal_bootstapir_checkpoint.pt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure `third_party/tapnet/` is on sys.path so `import tapnet.torch`
# resolves to the TAPIR package. `SurgiPano/surgipano/` does not contain a
# `torch/` dir, so `import torch` always finds the installed PyTorch.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_TAPNET = _ROOT / "third_party" / "tapnet"
if str(_TAPNET) not in sys.path:
    sys.path.insert(0, str(_TAPNET))

import argparse
import contextlib
import io
import time
import warnings

import cv2
import numpy as np
from tapnet.torch import tapir_model
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tree

from surgipano import stitch_utils as sd

model = None
resize_enabled = False
model_h = None
model_w = None
scale_x = 1.0
scale_y = 1.0
grid_spacing = 64
grid_min_brightness = 0.0


# ============================================================================
# Helpers (identical to original pytorch_video.py)
# ============================================================================
def preprocess_frames(frames):
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def online_model_init(frames, points):
    frames = preprocess_frames(frames)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    features = model.get_query_features(
        frames, is_training=False, query_points=points,
        feature_grids=feature_grids)
    return features


def postprocess_occlusions(occlusions, expected_dist):
    return (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5


def online_model_predict(frames, features, causal_context):
    frames = preprocess_frames(frames)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1], is_training=False,
        feature_grids=feature_grids, query_features=features,
        query_points_in_video=None, query_chunk_size=64,
        causal_context=causal_context, get_causal_context=True)
    causal_context = trajectories["causal_context"]
    del trajectories["causal_context"]
    tracks = trajectories["tracks"][-1]
    occlusions = trajectories["occlusion"][-1]
    uncertainty = trajectories["expected_dist"][-1]
    visibles_full = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(uncertainty)) > 0.5
    visibles_occ = (1 - F.sigmoid(occlusions)) > 0.5
    return tracks, visibles_full, visibles_occ, causal_context


def get_frame(video_capture):
    r_val, image = video_capture.read()
    return r_val, image


def to_model_frame(frame):
    if resize_enabled:
        return cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
    return frame


@contextlib.contextmanager
def _suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def sample_points_from_mask(mask, num_points):
    coords = np.argwhere(mask > 127)
    if coords.size == 0:
        raise ValueError("Mask has no available (nonzero) area.")
    if len(coords) > 50000:
        coords = coords[:: len(coords) // 50000 + 1]
    coords = coords.astype(np.float32)
    centroid = coords.mean(axis=0)
    seed_idx = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
    min_dist = np.sum((coords - coords[seed_idx]) ** 2, axis=1)
    selected_idx = [seed_idx]
    while len(selected_idx) < num_points:
        if min_dist.max() == 0:
            break
        nxt = int(np.argmax(min_dist))
        selected_idx.append(nxt)
        min_dist = np.minimum(min_dist, np.sum((coords - coords[nxt]) ** 2, axis=1))
    return [(0.0, float(y), float(x)) for y, x in coords[selected_idx]]


def build_grid(frame, mask_eroded):
    h, w = frame.shape[:2]
    spacing = grid_spacing
    full_grid = []
    y = spacing / 2.0
    while y < h:
        x = spacing / 2.0
        while x < w:
            full_grid.append((0.0, float(y), float(x)))
            x += spacing
        y += spacing
    if mask_eroded is None and grid_min_brightness <= 0:
        return full_grid, list(full_grid)
    bright = None
    if grid_min_brightness > 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.blur(gray, (31, 31))
    filtered = []
    for t, y, x in full_grid:
        yi, xi = int(y), int(x)
        if mask_eroded is not None and mask_eroded[yi, xi] <= 127:
            continue
        if bright is not None and bright[yi, xi] <= grid_min_brightness:
            continue
        filtered.append((t, y, x))
    return full_grid, filtered


# ============================================================================
# Grid mode
# ============================================================================
def run_grid_mode(args, device, vc, first_frame, fps, effective_total):
    """Run --mode grid: fixed-grid motion-vector field, re-anchored every k
    frames. Owns vc/writer cleanup and prints its own summary."""
    global grid_spacing, grid_min_brightness
    grid_spacing = args.grid_spacing
    grid_min_brightness = args.min_brightness

    if args.points is not None:
        raise ValueError("--points applies to --mode track only.")
    print("Note: --num_points is ignored in grid mode.")

    mask_eroded = None
    if args.mask is not None:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {args.mask}")
        if mask.shape[:2] != first_frame.shape[:2]:
            print(
                f"Warning: mask size {mask.shape[:2]} differs from frame "
                f"{first_frame.shape[:2]}, resizing mask to match the frame."
            )
            mask = cv2.resize(
                mask, (first_frame.shape[1], first_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_eroded = mask
        if args.mask_margin > 0:
            print(f"Eroding mask by {args.mask_margin}px before sampling.")
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * args.mask_margin + 1, 2 * args.mask_margin + 1),
            )
            eroded = cv2.erode(mask, kernel)
            if np.any(eroded > 127):
                mask_eroded = eroded
            else:
                print(
                    "mask_margin too large; no available area left after erosion. "
                    "Falling back to no margin."
                )

    frame_h, frame_w = first_frame.shape[:2]
    frame_diag = np.hypot(frame_w, frame_h)
    k = args.vector_window

    writer = None
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, fps, (frame_w, frame_h)
        )
        if not writer.isOpened():
            vc.release()
            raise ValueError(f"Unable to open output video for writing: {args.output}")

    processed = 0
    start_time = time.time()
    last_postfix = time.time()
    first_anchor = True
    fallback_printed = False

    pbar = tqdm(
        total=effective_total, desc="Grid", unit="frame", dynamic_ncols=True
    )
    anchor = first_frame
    try:
        while True:
            full_grid, filtered_grid = build_grid(anchor, mask_eroded)
            if filtered_grid:
                grid_points = filtered_grid
            else:
                grid_points = full_grid
                if not fallback_printed:
                    print(
                        "Note: mask/brightness filtering removed all grid points; "
                        "using unfiltered grid for this anchor."
                    )
                    fallback_printed = True
            if first_anchor:
                print(
                    f"Grid: {len(grid_points)} points (from {len(full_grid)} before "
                    f"mask/brightness filtering), spacing {grid_spacing}, window {k}."
                )
                first_anchor = False

            qp = np.array(grid_points, dtype=np.float32)
            qp[:, 1] *= scale_y
            qp[:, 2] *= scale_x
            qp_t = torch.tensor(qp).to(device)
            anchor_t = torch.tensor(to_model_frame(anchor)).to(device)
            with _suppress_stdout():
                query_features = online_model_init(
                    frames=anchor_t[None, None], points=qp_t[None, :]
                )
            causal_state = model.construct_initial_causal_state(
                len(qp), len(query_features.resolutions) - 1
            )
            causal_state = tree.map_structure(
                lambda x: x.to(device), causal_state
            )

            window_frames = [anchor]
            next_anchor = None
            track = None
            vis_full = None
            vis_occ = None
            completed = False
            for j in range(1, k + 1):
                rval, f = get_frame(vc)
                if not rval:
                    break
                f_tensor = torch.tensor(to_model_frame(f)).to(device)
                with _suppress_stdout():
                    track, vis_full, vis_occ, causal_state = online_model_predict(
                        frames=f_tensor[None, None],
                        features=query_features,
                        causal_context=causal_state,
                    )
                window_frames.append(f)
                if j == k:
                    completed = True
                    next_anchor = f
                    window_frames.pop()
                    break

            vectors = []
            if track is not None:
                track_np = track.cpu().numpy()
                track_np[..., 0] = track_np[..., 0] / scale_x
                track_np[..., 1] = track_np[..., 1] / scale_y
                vis_full_np = vis_full.cpu().numpy()
                vis_occ_np = vis_occ.cpu().numpy()
                if args.visibility == "occlusion":
                    visible = vis_occ_np
                elif args.visibility == "all":
                    visible = np.ones(vis_full_np.shape, dtype=bool)
                else:
                    visible = vis_full_np
                for i, (t, y, x) in enumerate(grid_points):
                    if not visible[0, i, 0]:
                        continue
                    x2 = float(track_np[0, i, 0, 0])
                    y2 = float(track_np[0, i, 0, 1])
                    if np.hypot(x2 - x, y2 - y) > 0.4 * frame_diag:
                        continue
                    vectors.append((int(x), int(y), int(x2), int(y2)))

            for f_img in window_frames:
                for x, y, x2, y2 in vectors:
                    cv2.arrowedLine(
                        f_img, (x, y), (x2, y2), (0, 255, 0),
                        max(1, int(round(args.marker_scale))),
                        cv2.LINE_8, 0, 0.25,
                    )
                if writer is not None:
                    writer.write(f_img)
                if not args.no_display:
                    try:
                        cv2.imshow("Point Tracking", f_img)
                        if cv2.waitKey(1) & 0xFF == 27:
                            next_anchor = None
                            break
                    except cv2.error as exc:
                        print(f"[warn] cv2.imshow failed ({exc}); "
                              f"continuing headless.")
                        args.no_display = True
                processed += 1
                pbar.update(1)
                if time.time() - last_postfix > 1.0:
                    pbar.set_postfix_str(
                        f"{processed / (time.time() - start_time):.2f} fps")
                    last_postfix = time.time()
                if args.max_frames is not None and processed >= args.max_frames:
                    next_anchor = None
                    break

            if next_anchor is None:
                break
            anchor = next_anchor
    finally:
        pbar.close()

    elapsed = time.time() - start_time
    avg_fps = processed / elapsed if elapsed > 0 else 0.0

    if not args.no_display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    vc.release()
    if writer is not None:
        writer.release()

    summary = f"Processed {processed} frame(s), average {avg_fps:.2f} fps"
    if args.output:
        summary += f", output written to {args.output}"
    summary += f" (grid spacing {args.grid_spacing}, window {args.vector_window})"
    print(summary)


# ============================================================================
# Arg parser (--checkpoint default points into the tapnet submodule)
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline point tracking with PyTorch Online TAPIR."
    )
    parser.add_argument("--video", required=True,
                        help="Path to the input mp4 video file.")
    parser.add_argument(
        "--checkpoint",
        default=str(_TAPNET / "checkpoints" / "causal_bootstapir_checkpoint.pt"),
        help="Path to the TAPIR checkpoint.",
    )
    parser.add_argument("--num_points", type=int, default=16,
                        help="Maximum number of query points to track.")
    parser.add_argument(
        "--points", default=None,
        help=("Optional space-separated 'x,y' pixel coordinates in the first "
              "frame, e.g. \"100,150 200,150\". Skips interactive selection."),
    )
    parser.add_argument("--mask", default=None,
                        help="Path to a grayscale mask PNG (nonzero = available).")
    parser.add_argument("--mask_margin", type=int, default=15,
                        help="Erode the mask by this many pixels before sampling.")
    parser.add_argument("--min_brightness", type=float, default=0.0,
                        help="Only sample points where local brightness > value.")
    parser.add_argument("--output", default=None,
                        help="Optional path to write an annotated output mp4.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional maximum number of frames to process.")
    parser.add_argument("--no_display", action="store_true",
                        help="Headless mode: do not open any cv2 window.")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (default: cuda if available).")
    parser.add_argument(
        "--resize_height", type=int, default=256,
        help=("Square size for the model input (aspect stretched). 0 = original "
              "resolution (not recommended: non-square breaks the causal model)."),
    )
    parser.add_argument("--visibility",
                        choices=["full", "occlusion", "all"], default="full",
                        help="How tracked points are drawn.")
    parser.add_argument("--mode", choices=["track", "grid"], default="track",
                        help="'track' (default) or 'grid' motion-vector field.")
    parser.add_argument("--grid_spacing", type=int, default=64,
                        help="Grid spacing (original pixels) for --mode grid.")
    parser.add_argument("--vector_window", type=int, default=1,
                        help="Vector window k: anchor frame -> frame t+k.")
    parser.add_argument("--marker_scale", type=float, default=2.0,
                        help="Scale factor for drawn markers.")
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    global model, resize_enabled, model_h, model_w, scale_x, scale_y

    args = parse_args()

    # ---- headless fallback ----
    if not args.no_display and not sd.gui_available():
        print("[warn] OpenCV has no GUI backend; forcing --no_display.")
        args.no_display = True

    warnings.filterwarnings(
        "ignore", message=".*output size is not a multiple of 8.*")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Welcome to the TAPIR PyTorch video demo.")
    print("Tracking points in an offline video file.")
    print("Press ESC to exit (when display is enabled).")

    print("Creating model...")
    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model = model.eval()
    torch.set_grad_enabled(False)

    print("Opening video...")
    vc = cv2.VideoCapture(args.video)
    if not vc.isOpened():
        raise ValueError(f"Unable to open video: {args.video}")

    fps = vc.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
    frame_w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        effective_total = None
    elif args.max_frames is not None:
        effective_total = min(args.max_frames, total_frames)
    else:
        effective_total = total_frames

    rval, frame = get_frame(vc)
    if not rval:
        vc.release()
        raise ValueError(f"Unable to read the first frame of video: {args.video}")

    if args.resize_height and frame.shape[0] != args.resize_height:
        resize_enabled = True
        model_h = args.resize_height
        model_w = args.resize_height
        scale_y = model_h / frame.shape[0]
        scale_x = model_w / frame.shape[1]
        print(
            f"Model input resized to {model_w}x{model_h} "
            f"(square, aspect stretched; resize_height={args.resize_height})."
        )
    else:
        resize_enabled = False
        model_h = frame.shape[0]
        model_w = frame.shape[1]
        scale_y = 1.0
        scale_x = 1.0
        print("Model input running at original resolution.")

    if args.mode == "grid":
        run_grid_mode(args, device, vc, frame, fps, effective_total)
        return

    # ---- track mode ----
    selected_points = []

    if args.points is not None:
        if args.mask is not None:
            print("--points given, ignoring --mask.")
        try:
            for pair in args.points.split():
                x_str, y_str = pair.split(",")
                selected_points.append((0.0, float(y_str), float(x_str)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --points value {args.points!r}; expected space-separated "
                "'x,y' pairs, e.g. \"100,150 200,150\"."
            ) from exc
        print(f"Using {len(selected_points)} point(s) from --points.")
    elif args.mask is not None:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {args.mask}")
        if mask.shape[:2] != frame.shape[:2]:
            print(
                f"Warning: mask size {mask.shape[:2]} differs from frame "
                f"{frame.shape[:2]}, resizing mask to match the frame."
            )
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        sample_mask = mask
        if args.mask_margin > 0:
            print(f"Eroding mask by {args.mask_margin}px before sampling.")
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * args.mask_margin + 1, 2 * args.mask_margin + 1),
            )
            eroded = cv2.erode(mask, kernel)
            if np.any(eroded > 127):
                sample_mask = eroded
            else:
                print(
                    "mask_margin too large; no available area left after erosion. "
                    "Falling back to no margin."
                )
        if args.min_brightness > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = cv2.blur(gray, (31, 31))
            bright_mask = (
                (brightness > args.min_brightness) & (sample_mask > 127)
            ).astype(np.uint8) * 255
            if np.any(bright_mask > 127):
                print(
                    f"Excluding first-frame areas with local brightness below "
                    f"{args.min_brightness}."
                )
                sample_mask = bright_mask
            else:
                print(
                    "min_brightness too high; no available area left after filtering. "
                    "Falling back to brightness-unfiltered mask."
                )
        selected_points = sample_points_from_mask(sample_mask, args.num_points)
        print(f"Sampled {len(selected_points)} point(s) from mask {args.mask}.")
        print("Points: " + " ".join(
            f"{int(x)},{int(y)}" for _, y, x in selected_points))
    elif not args.no_display:
        display_frame = frame.copy()
        window_name = "Select Points"

        def mouse_click(event, x, y, flags, param):
            del flags, param
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(selected_points) < args.num_points:
                    selected_points.append((0.0, float(y), float(x)))
                    cv2.circle(display_frame, (x, y),
                               int(5 * args.marker_scale), (0, 255, 0), -1)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_click)

        print("Left click to add points, 'r' to reset, SPACE/ENTER to finish, "
              "ESC to quit.")
        while True:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                print("Point selection cancelled.")
                cv2.destroyWindow(window_name)
                vc.release()
                return
            elif key == ord("r"):
                selected_points.clear()
                display_frame = frame.copy()
            elif key in (13, 32):
                if not selected_points:
                    print("Please select at least one point before finishing.")
                    continue
                print(f"Selected {len(selected_points)} point(s).")
                break

        cv2.destroyWindow(window_name)
    else:
        vc.release()
        raise ValueError(
            "No query points available: pass --points (e.g. \"100,150 200,150\") "
            "or --mask (path to a mask PNG) when using --no_display."
        )

    if not selected_points:
        vc.release()
        raise ValueError("No query points selected.")

    num_selected = len(selected_points)
    query_points = np.array(selected_points, dtype=np.float32)
    query_points[:, 1] *= scale_y
    query_points[:, 2] *= scale_x
    query_points = torch.tensor(query_points).to(device)

    frame_tensor = torch.tensor(to_model_frame(frame)).to(device)
    with _suppress_stdout():
        query_features = online_model_init(
            frames=frame_tensor[None, None], points=query_points[None, :]
        )

    causal_state = model.construct_initial_causal_state(
        num_selected, len(query_features.resolutions) - 1
    )
    causal_state = tree.map_structure(lambda x: x.to(device), causal_state)

    writer = None
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, fps, (frame.shape[1], frame.shape[0])
        )
        if not writer.isOpened():
            vc.release()
            raise ValueError(f"Unable to open output video for writing: {args.output}")

    processed = 0
    start_time = time.time()
    last_postfix = time.time()

    pbar = tqdm(
        total=effective_total, desc="Tracking", unit="frame", dynamic_ncols=True
    )
    try:
        while True:
            frame_tensor = torch.tensor(to_model_frame(frame)).to(device)
            with _suppress_stdout():
                track, visibles_full, visibles_occ, causal_state = \
                    online_model_predict(
                        frames=frame_tensor[None, None],
                        features=query_features,
                        causal_context=causal_state,
                    )
            track = track.cpu().numpy()
            track[..., 0] = track[..., 0] / scale_x
            track[..., 1] = track[..., 1] / scale_y
            track = np.round(track)
            visibles_full = visibles_full.cpu().numpy()
            visibles_occ = visibles_occ.cpu().numpy()
            if args.visibility == "occlusion":
                visible = visibles_occ
            elif args.visibility == "all":
                visible = np.ones(visibles_full.shape, dtype=bool)
            else:
                visible = visibles_full

            numpy_frame = frame
            for i in range(num_selected):
                if visible[0, i, 0]:
                    cv2.circle(
                        numpy_frame,
                        (int(track[0, i, 0, 0]), int(track[0, i, 0, 1])),
                        int(5 * args.marker_scale),
                        (255, 0, 0), -1,
                    )

            if writer is not None:
                writer.write(numpy_frame)

            if not args.no_display:
                try:
                    cv2.imshow("Point Tracking", numpy_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except cv2.error as exc:
                    print(f"[warn] cv2.imshow failed ({exc}); "
                          f"continuing headless.")
                    args.no_display = True

            processed += 1
            pbar.update(1)
            if time.time() - last_postfix > 1.0:
                pbar.set_postfix_str(
                    f"{processed / (time.time() - start_time):.2f} fps")
                last_postfix = time.time()

            if args.max_frames is not None and processed >= args.max_frames:
                break

            rval, frame = get_frame(vc)
            if not rval:
                break
    finally:
        pbar.close()

    elapsed = time.time() - start_time
    avg_fps = processed / elapsed if elapsed > 0 else 0.0

    if not args.no_display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    vc.release()
    if writer is not None:
        writer.release()

    summary = f"Processed {processed} frame(s), average {avg_fps:.2f} fps"
    if args.output:
        summary += f", output written to {args.output}"
    print(summary)


if __name__ == "__main__":
    main()