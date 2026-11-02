"""Continuous TAPIR-guided geometric registration, no neural-network training.

Stitch provides a relative geometric candidate; TAPIR provides temporal point
correspondences. Robust refinement selects a current->previous transform and
builds an independent fused panorama. Does NOT change Stitch internal canvas.
"""
import argparse
import time
from pathlib import Path
import cv2
from tqdm.auto import tqdm
import numpy as np
from scipy.optimize import least_squares
from common import (expected_frame_count, add_arguments, frame_iterator, make_stitch, image_numpy, normalize_h,
                    estimate_h, transform, valid_h, canvas_init, compose, mask_for_frame,
                    stitch_error, write_metrics, save_rgb)
from tapir_adapter import TapirTracker
from video_preview import AlignmentVideo
from mask_utils import ScopeMask


def refine_h(H_stitch, H_tapir, prev_pts, curr_pts, w, h, stitch_weight=0.25):
    """Fit current->previous homography to TAPIR tracks plus soft Stitch prior."""
    if H_tapir is None: return H_stitch, 'stitch'
    if H_stitch is None: return H_tapir, 'tapir'
    # RANSAC TAPIR inliers prevent gross outliers dominating refinement.
    errs = np.linalg.norm(transform(H_tapir, curr_pts)-prev_pts, axis=1)
    ok = errs < 4.5
    if ok.sum() < 4: return H_stitch, 'stitch'
    src, dst = curr_pts[ok], prev_pts[ok]
    anchors = np.float32([[0,0],[w/2,0],[w,0],[0,h/2],[w/2,h/2],[w,h/2],[0,h],[w/2,h],[w,h]])
    stitch_targets = transform(H_stitch, anchors)
    def encode(H): return np.array([H[0,0],H[0,1],H[0,2],H[1,0],H[1,1],H[1,2],H[2,0],H[2,1]])
    def decode(x): return np.array([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],1.0]])
    def residual(x):
        H = decode(x)
        try:
            r_track = (transform(H, src)-dst).ravel()
            r_stitch = (transform(H, anchors)-stitch_targets).ravel()
            return np.r_[r_track, np.sqrt(stitch_weight)*r_stitch]
        except cv2.error: return np.full(2*len(src)+2*len(anchors), 1e5)
    result = least_squares(residual, encode(H_tapir), loss='soft_l1', f_scale=3,
                           max_nfev=35)
    H = normalize_h(decode(result.x))
    return (H, 'fused') if valid_h(H, w, h) else (H_tapir, 'tapir')


def main():
    p = argparse.ArgumentParser()
    add_arguments(p)
    p.add_argument('--tapir-checkpoint', required=True)
    p.add_argument('--tapir-resolution', type=int, default=256)
    p.add_argument('--tapir-spacing', type=int, default=48)
    p.add_argument('--tapir-refresh', type=int, default=30)
    p.add_argument('--ransac-threshold', type=float, default=3.0)
    p.add_argument('--stitch-weight', type=float, default=0.25)
    p.add_argument('--max-disagreement', type=float, default=25.0)
    p.add_argument('--preview-width', type=int, default=1600)
    p.add_argument('--preview-height', type=int, default=720)
    p.add_argument('--preview-every', type=int, default=1)
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--mask-margin', type=int, default=12)
    p.add_argument('--canvas-view', choices=['stitch','fused'], default='stitch',
                   help='Which panorama to show in alignment_preview.mp4 and panorama_final.png')
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stitch = make_stitch(args)
    scope = ScopeMask(args.scope_mask, args.width, args.height, args.mask_margin)
    tracker = TapirTracker(args.tapir_checkpoint, args.device, args.tapir_resolution,
                           args.tapir_spacing, args.tapir_refresh)
    canvas = coverage = H_canvas = H_stitch_prev = None
    rows = []
    preview = None
    prev_shape = None
    total = expected_frame_count(args.video, args.max_frames, args.stride)
    with tqdm(total=total, desc='Integrated', unit='frame', dynamic_ncols=True, smoothing=0.1) as progress:
      for i, rgb, fps in frame_iterator(args.video, args.width, args.height, args.max_frames, args.stride):
            t0 = time.perf_counter()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame_u = stitch.preprocess_frame(bgr)
            frame = image_numpy(frame_u)
            h, w = frame.shape[:2]
            if prev_shape is not None and prev_shape != (h,w):
                raise RuntimeError('Stitch preprocessing changed ROI size; a fixed ROI is required')
            prev_shape = (h,w)
            valid_mask = scope.crop(stitch, frame.shape)
            mask = scope.eroded(valid_mask)
            if args.scope_mask:
                scope.inject_stitch(stitch, valid_mask, args.device)
            p_prev, p_curr, ntracks = tracker.update(frame, mask)
            H_tapir, inliers, tapir_error = estimate_h(p_prev, p_curr, w, h, args.ransac_threshold)
            try:
                stitch.step_canvas(frame_u)
                H_stitch_abs = normalize_h(stitch._current_to_canvas_h())
            except (RuntimeError, ValueError) as exc:
                print(f'frame {i}: Stitch failed: {exc}; using TAPIR if possible')
                H_stitch_abs = None
            H_stitch_rel = None
            if H_stitch_prev is not None and H_stitch_abs is not None:
                try:
                    H_stitch_rel = normalize_h(np.linalg.inv(H_stitch_prev) @ H_stitch_abs)
                except np.linalg.LinAlgError: pass
            # Relative Stitch transform maps current -> previous frame.
            if not valid_h(H_stitch_rel, w, h): H_stitch_rel = None
            disagreement = stitch_error(H_stitch_rel, np.eye(3), p_prev, p_curr)
            if np.isfinite(disagreement) and disagreement > args.max_disagreement and H_tapir is not None:
                H_stitch_rel = None
            H_rel, source = refine_h(H_stitch_rel, H_tapir, p_prev, p_curr, w, h, args.stitch_weight)
            if canvas is None:
                canvas, coverage, H_canvas = canvas_init(w, h, args.canvas_scale)
                source = 'initialize'
            elif valid_h(H_rel, w, h):
                candidate = normalize_h(H_canvas @ H_rel)
                if valid_h(candidate, w, h): H_canvas = candidate
                else: source = 'hold_invalid'
            else:
                source = 'hold_no_motion'
            compose(canvas, coverage, frame, H_canvas, mask)
            if preview is None and not args.no_preview:
                preview = AlignmentVideo(out / 'alignment_preview.mp4', fps / max(args.stride,1), args.preview_width, args.preview_height, args.preview_every)
            if preview is not None:
                if args.canvas_view == 'stitch' and stitch.canvas is not None:
                    preview.write(rgb, stitch.canvas, H_stitch_abs, frame.shape, i,
                                  f'Stitch renderer | fusion={source} | tracks={ntracks}',
                                  coverage=getattr(stitch, 'canvas_mask', None))
                else:
                    preview.write(rgb, canvas, H_canvas, frame.shape, i,
                                  f'Fused renderer | {source} | tracks={ntracks}', coverage=coverage)
            if H_stitch_abs is not None: H_stitch_prev = H_stitch_abs
            rows.append(dict(frame=i, seconds=time.perf_counter()-t0, source=source,
                             tapir_tracks=ntracks, tapir_inliers=inliers,
                             tapir_reprojection_px=tapir_error, stitch_tapir_disagreement_px=disagreement,
                             stitch_valid=int(H_stitch_abs is not None), coverage_pixels=int(np.count_nonzero(coverage)),
                             h00=H_canvas[0,0],h01=H_canvas[0,1],h02=H_canvas[0,2],
                             h10=H_canvas[1,0],h11=H_canvas[1,1],h12=H_canvas[1,2],
                             h20=H_canvas[2,0],h21=H_canvas[2,1],h22=H_canvas[2,2]))
            if args.save_every and len(rows) % args.save_every == 0:
                chosen = image_numpy(stitch.canvas) if args.canvas_view == 'stitch' and stitch.canvas is not None else canvas
                save_rgb(out / f'panorama_{i:06d}.png', chosen)
            progress.set_postfix(source=source, tracks=ntracks, refresh=False)
            progress.update(1)
    if preview is not None: preview.close()
    if canvas is not None:
        save_rgb(out / 'panorama_fused_final.png', canvas)
        if stitch.canvas is not None:
            save_rgb(out / 'panorama_stitch_final.png', image_numpy(stitch.canvas))
        chosen = image_numpy(stitch.canvas) if args.canvas_view == 'stitch' and stitch.canvas is not None else canvas
        save_rgb(out / 'panorama_final.png', chosen)
        cv2.imwrite(str(out / 'coverage.png'), coverage)
    write_metrics(out / 'metrics.csv', rows)
    print('Saved', out)

if __name__ == '__main__': main()
