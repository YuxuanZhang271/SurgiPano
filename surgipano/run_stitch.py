"""Native Stitch-seg panorama with scope mask and side-by-side video."""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm.auto import tqdm
from common import (expected_frame_count, add_arguments, frame_iterator,
                    make_stitch, image_numpy, normalize_h, write_metrics, save_rgb)
from video_preview import AlignmentVideo
from mask_utils import ScopeMask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--mask-margin', type=int, default=12)
    p.add_argument('--preview-width', type=int, default=1600)
    p.add_argument('--preview-height', type=int, default=720)
    p.add_argument('--preview-every', type=int, default=1)
    p.add_argument('--no-preview', action='store_true')
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stitch = make_stitch(args)
    scope = ScopeMask(args.scope_mask, args.width, args.height, args.mask_margin)
    rows, preview = [], None
    total = expected_frame_count(args.video, args.max_frames, args.stride)
    try:
        with tqdm(total=total, desc='Stitch', unit='frame', dynamic_ncols=True) as bar:
            for i, rgb, fps in frame_iterator(args.video, args.width, args.height, args.max_frames, args.stride):
                t0 = time.perf_counter()
                frame_u = stitch.preprocess_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cropped = image_numpy(frame_u)
                valid = scope.crop(stitch, cropped.shape)
                if args.scope_mask:
                    scope.inject_stitch(stitch, valid, args.device)
                stitch.step_canvas(frame_u)
                H = normalize_h(stitch._current_to_canvas_h())
                canvas = image_numpy(stitch.canvas)
                if preview is None and not args.no_preview:
                    preview = AlignmentVideo(out/'alignment_preview.mp4', fps/max(args.stride,1),
                                             args.preview_width, args.preview_height, args.preview_every)
                if preview is not None and canvas is not None:
                    preview.write(rgb, canvas, H, cropped.shape, i, 'Stitch native canvas',
                                  coverage=getattr(stitch, 'canvas_mask', None))
                rows.append(dict(frame=i, seconds=time.perf_counter()-t0,
                                 stitch_valid=int(H is not None),
                                 canvas_pixels=int(np.count_nonzero(np.any(canvas != 0, axis=2))) if canvas is not None else 0))
                if args.save_every and len(rows) % args.save_every == 0 and canvas is not None:
                    save_rgb(out/f'panorama_{i:06d}.png', canvas)
                bar.update(1)
    finally:
        if preview is not None: preview.close()
    if stitch.canvas is not None:
        save_rgb(out/'panorama_final.png', image_numpy(stitch.canvas))
    write_metrics(out/'metrics.csv', rows)
    print('Saved', out)

if __name__ == '__main__': main()
