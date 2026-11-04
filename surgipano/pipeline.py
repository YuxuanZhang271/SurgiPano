"""Shared independent TAPIR/Stitch and EMDQ-inspired TAPIR/Stitch runners."""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm.auto import tqdm
from common import (expected_frame_count, add_arguments, frame_iterator, make_stitch,
                    image_numpy, normalize_h, estimate_h, transform, valid_h,
                    canvas_init, compose, stitch_error, write_metrics, save_rgb)
from tapir_adapter import TapirTracker
from mask_utils import ScopeMask
from video_preview import AlignmentVideo
from nonrigid import fit_residual_field, compose_nonrigid
from scipy.optimize import least_squares


def refine_h(H_stitch, H_tapir, prev_pts, curr_pts, w, h, stitch_weight=.25):
    if H_tapir is None:
        return H_stitch, 'stitch' if H_stitch is not None else 'hold_no_motion'
    if H_stitch is None:
        return H_tapir, 'tapir'
    errors = np.linalg.norm(transform(H_tapir, curr_pts) - prev_pts, axis=1)
    good = errors < 4.5
    if good.sum() < 4:
        return H_stitch, 'stitch'
    src, dst = curr_pts[good], prev_pts[good]
    anchors = np.float32([[0,0],[w/2,0],[w,0],[0,h/2],[w/2,h/2],[w,h/2],[0,h],[w/2,h],[w,h]])
    targets = transform(H_stitch, anchors)
    def encode(H):
        return np.r_[H[0], H[1], H[2,:2]]
    def decode(x):
        return np.array([[x[0],x[1],x[2]], [x[3],x[4],x[5]], [x[6],x[7],1.]])
    def residual(x):
        H = decode(x)
        try:
            return np.r_[(transform(H, src)-dst).ravel(),
                         np.sqrt(max(0.,stitch_weight))*(transform(H, anchors)-targets).ravel()]
        except cv2.error:
            return np.full(2*(len(src)+len(anchors)), 1e5)
    result = least_squares(residual, encode(H_tapir), loss='soft_l1', f_scale=3., max_nfev=35)
    H = normalize_h(decode(result.x))
    return (H, 'fused') if valid_h(H, w, h) else (H_tapir, 'tapir')


def motion_quality(H, w, h, max_corner_motion=90., max_scale=1.4):
    """Reject unstable relative projective updates before cumulative composition."""
    if not valid_h(H, w, h):
        return False
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    q = transform(H, corners)
    if not np.isfinite(q).all():
        return False
    if np.max(np.linalg.norm(q-corners, axis=1)) > max_corner_motion:
        return False
    a = cv2.contourArea(q.astype(np.float32), oriented=True)
    base = float((w-1)*(h-1))
    if not (base / max_scale**2 <= a <= base * max_scale**2):
        return False
    if not cv2.isContourConvex(q.astype(np.float32)):
        return False
    return True


def tapir_quality(H, batch, w, h, threshold=3., min_inliers=16,
                  min_ratio=.35, min_span=.25, max_motion=90.):
    """Require consensus, geographic support, and plausible frame-to-frame motion."""
    n = len(batch.curr)
    if H is None or n < min_inliers or not motion_quality(H, w, h, max_motion):
        return False, 0, float('nan')
    err = np.linalg.norm(transform(H, batch.curr)-batch.prev, axis=1)
    ok = np.isfinite(err) & (err < threshold)
    k = int(ok.sum())
    if k < min_inliers or k / n < min_ratio:
        return False, k, float(np.median(err[ok])) if k else float('nan')
    src = batch.curr[ok]
    if np.ptp(src[:, 0]) < min_span*w or np.ptp(src[:, 1]) < min_span*h:
        return False, k, float(np.median(err[ok]))
    return True, k, float(np.median(err[ok]))


def add_pipeline_arguments(p, nonrigid):
    add_arguments(p)
    p.add_argument('--tapir-checkpoint', required=True)
    p.add_argument('--tapir-resolution', type=int, default=256)
    p.add_argument('--tapir-spacing', type=int, default=48)
    p.add_argument('--tapir-max-points', type=int, default=192)
    p.add_argument('--tapir-min-confidence', type=float, default=.5)
    p.add_argument('--tapir-max-missing', type=int, default=20)
    p.add_argument('--tapir-replenish-every', type=int, default=5)
    p.add_argument('--ransac-threshold', type=float, default=3.)
    p.add_argument('--stitch-weight', type=float, default=.25)
    p.add_argument('--max-disagreement', type=float, default=25.)
    p.add_argument('--min-tapir-inliers', type=int, default=16)
    p.add_argument('--min-tapir-ratio', type=float, default=.35)
    p.add_argument('--max-frame-motion', type=float, default=90.)
    p.add_argument('--max-frame-scale', type=float, default=1.4)
    p.add_argument('--preview-width', type=int, default=1600)
    p.add_argument('--preview-height', type=int, default=720)
    p.add_argument('--preview-every', type=int, default=1)
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--mask-margin', type=int, default=12)
    p.add_argument('--canvas-view', choices=['stitch','fused'], default='fused')
    if nonrigid:
        p.add_argument('--deform-sigma', type=float, default=65.)
        p.add_argument('--deform-max-px', type=float, default=20.)
        p.add_argument('--deform-min-points', type=int, default=6)
        p.add_argument('--deform-grid-step', type=int, default=8)
        p.add_argument('--deform-support', type=float, default=.15)


def run(nonrigid=False):
    p = argparse.ArgumentParser(description='SurgiPano: ' + ('EMDQ-inspired nonrigid' if nonrigid else 'persistent TAPIR') + ' fusion')
    add_pipeline_arguments(p, nonrigid)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stitch = make_stitch(args)
    scope = ScopeMask(args.scope_mask, args.width, args.height, args.mask_margin)
    tracker = TapirTracker(args.tapir_checkpoint, args.device, args.tapir_resolution,
                           args.tapir_spacing, max_points=args.tapir_max_points,
                           min_confidence=args.tapir_min_confidence,
                           max_missing=args.tapir_max_missing,
                           replenish_every=args.tapir_replenish_every)
    canvas = coverage = H_canvas = H_stitch_prev = None
    rows, preview = [], None
    prev_shape = None
    total = expected_frame_count(args.video, args.max_frames, args.stride)
    try:
        with tqdm(total=total, desc='EMDQ+TAPIR+Stitch' if nonrigid else 'TAPIR+Stitch', unit='frame') as progress:
            for i, rgb, fps in frame_iterator(args.video, args.width, args.height, args.max_frames, args.stride):
                t0 = time.perf_counter()
                frame_u = stitch.preprocess_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frame = image_numpy(frame_u)
                h, w = frame.shape[:2]
                if prev_shape is not None and prev_shape != (h, w):
                    raise RuntimeError('Stitch ROI size changed; fixed ROI required')
                prev_shape = (h, w)
                valid_mask = scope.crop(stitch, frame.shape)
                mask = scope.eroded(valid_mask)
                if args.scope_mask:
                    scope.inject_stitch(stitch, valid_mask, args.device)
                batch = tracker.update(frame, mask)
                H_tapir_raw, raw_inliers, raw_error = estimate_h(
                    batch.prev, batch.curr, w, h, args.ransac_threshold)
                tapir_ok, inliers, tapir_error = tapir_quality(
                    H_tapir_raw, batch, w, h, args.ransac_threshold,
                    args.min_tapir_inliers, args.min_tapir_ratio,
                    max_motion=args.max_frame_motion)
                H_tapir = H_tapir_raw if tapir_ok else None
                try:
                    stitch.step_canvas(frame_u)
                    H_stitch_abs = normalize_h(stitch._current_to_canvas_h())
                except (RuntimeError, ValueError, cv2.error) as exc:
                    print(f'frame {i}: Stitch failed: {exc}')
                    H_stitch_abs = None
                H_stitch_rel = None
                if H_stitch_prev is not None and H_stitch_abs is not None:
                    try:
                        H_stitch_rel = normalize_h(np.linalg.inv(H_stitch_prev) @ H_stitch_abs)
                    except np.linalg.LinAlgError:
                        pass
                stitch_ok = motion_quality(H_stitch_rel, w, h,
                    args.max_frame_motion, args.max_frame_scale)
                if not stitch_ok:
                    H_stitch_rel = None
                disagreement = stitch_error(H_stitch_rel, np.eye(3), batch.prev, batch.curr)
                # Never discard a plausible Stitch transform on the basis of
                # an untrusted TAPIR hypothesis. When both disagree, keep Stitch.
                if H_stitch_rel is not None:
                    if (H_tapir is not None and np.isfinite(disagreement)
                            and disagreement <= args.max_disagreement):
                        H_rel, source = refine_h(H_stitch_rel, H_tapir,
                            batch.prev, batch.curr, w, h, args.stitch_weight)
                        if not motion_quality(H_rel, w, h,
                                args.max_frame_motion, args.max_frame_scale):
                            H_rel, source = H_stitch_rel, 'stitch'
                    else:
                        H_rel, source = H_stitch_rel, 'stitch'
                elif H_tapir is not None:
                    H_rel, source = H_tapir, 'tapir'
                else:
                    H_rel, source = None, 'hold_no_motion'
                previous_canvas_H = H_canvas.copy() if H_canvas is not None else None
                if canvas is None:
                    canvas, coverage, H_canvas = canvas_init(w, h, args.canvas_scale)
                    source = 'initialize'
                elif H_rel is not None:
                    candidate = normalize_h(H_canvas @ H_rel)
                    if candidate is not None and valid_h(candidate, w, h):
                        H_canvas = candidate
                    else:
                        source = 'hold_invalid'
                else:
                    source = 'hold_no_motion'
                # A held pose is NOT a valid observation. Never paint an
                # unregistered frame onto the panorama at the previous pose.
                registration_ok = source not in ('hold_invalid', 'hold_no_motion')
                field, deform_inliers, deform_rmse = None, 0, float('nan')
                if (nonrigid and previous_canvas_H is not None and
                        registration_ok and tapir_ok and H_stitch_rel is not None and
                        np.isfinite(disagreement) and disagreement <= args.max_disagreement and
                        len(batch.curr)):
                    # Target is the preceding panorama coordinate. This correctly
                    # accounts for projective changes of basis in the residual.
                    target = transform(previous_canvas_H, batch.prev)
                    field, deform_inliers, deform_rmse = fit_residual_field(
                        H_canvas, target, batch.curr, batch.confidence,
                        sigma=args.deform_sigma, max_residual=args.deform_max_px,
                        min_points=args.deform_min_points, support=args.deform_support)
                if not registration_ok:
                    pass  # Keep existing canvas unchanged.
                elif nonrigid and field is not None:
                    try:
                        compose_nonrigid(canvas, coverage, frame, H_canvas, field, mask,
                                         grid_step=args.deform_grid_step)
                    except (np.linalg.LinAlgError, cv2.error, ValueError):
                        field = None
                        compose(canvas, coverage, frame, H_canvas, mask)
                else:
                    compose(canvas, coverage, frame, H_canvas, mask)
                if registration_ok:
                    tracker.remember_canvas(H_canvas, batch)
                if preview is None and not args.no_preview:
                    preview = AlignmentVideo(out/'alignment_preview.mp4', fps/max(1,args.stride),
                        args.preview_width, args.preview_height, args.preview_every)
                if preview is not None:
                    if args.canvas_view == 'stitch' and stitch.canvas is not None:
                        preview.write(rgb, stitch.canvas, H_stitch_abs, frame.shape, i,
                            f'Stitch | {source} | tracks={len(batch.curr)}',
                            coverage=getattr(stitch, 'canvas_mask', None))
                    else:
                        preview.write(rgb, canvas, H_canvas, frame.shape, i,
                            f'Fusion | {source} | tracks={len(batch.curr)} | deform={deform_inliers}', coverage=coverage)
                if H_stitch_abs is not None:
                    H_stitch_prev = H_stitch_abs
                row = dict(frame=i, seconds=time.perf_counter()-t0, source=source,
                    tapir_tracks=len(batch.curr), tapir_inliers=inliers,
                    tapir_reprojection_px=tapir_error, stitch_tapir_disagreement_px=disagreement,
                    tapir_active=batch.active, tapir_born=batch.born,
                    tapir_occluded=batch.occluded, tapir_retired=batch.retired,
                    tapir_median_age=float(np.median(batch.age)) if len(batch.age) else float('nan'),
                    tapir_mean_confidence=float(batch.confidence.mean()) if len(batch.confidence) else float('nan'),
                    stitch_valid=int(H_stitch_abs is not None),
                    stitch_motion_valid=int(stitch_ok), tapir_motion_valid=int(tapir_ok),
                    tapir_inlier_ratio=float(inliers / max(1, len(batch.curr))),
                    registration_accepted=int(registration_ok),
                    coverage_pixels=int(np.count_nonzero(coverage)),
                    deform_active=int(field is not None), deform_inliers=deform_inliers,
                    deform_rmse_px=deform_rmse,
                    **{f'h{r}{c}':float(H_canvas[r,c]) for r in range(3) for c in range(3)})
                rows.append(row)
                if args.save_every and len(rows) % args.save_every == 0:
                    chosen = image_numpy(stitch.canvas) if args.canvas_view == 'stitch' and stitch.canvas is not None else canvas
                    save_rgb(out/f'panorama_{i:06d}.png', chosen)
                progress.set_postfix(source=source, tracks=len(batch.curr), born=batch.born, deform=deform_inliers, refresh=False)
                progress.update(1)
    finally:
        if preview is not None:
            preview.close()
        if canvas is not None:
            save_rgb(out/'panorama_fused_final.png', canvas)
            if stitch.canvas is not None:
                save_rgb(out/'panorama_stitch_final.png', image_numpy(stitch.canvas))
            chosen = image_numpy(stitch.canvas) if args.canvas_view == 'stitch' and stitch.canvas is not None else canvas
            save_rgb(out/'panorama_final.png', chosen)
            cv2.imwrite(str(out/'coverage.png'), coverage)
        write_metrics(out/'metrics.csv', rows)
    print('Saved', out)
