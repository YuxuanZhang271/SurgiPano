"""Compare outputs. Optional ground truth CSV: frame,x,y (canvas pixel location)."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2


def summarize(path, gt=None):
    path = Path(path)
    df = pd.read_csv(path / 'metrics.csv')
    out = {'frames': int(len(df)), 'mean_fps': float(len(df) / df.seconds.sum()),
           'median_latency_ms': float(df.seconds.median()*1000),
           'p95_latency_ms': float(df.seconds.quantile(.95)*1000)}
    if 'stitch_valid' in df: out['stitch_valid_fraction'] = float(df.stitch_valid.mean())
    if 'tapir_tracks' in df:
        out['median_tapir_tracks'] = float(df.tapir_tracks.median())
        out['median_tapir_inliers'] = float(df.tapir_inliers.median())
        out['median_tapir_reprojection_px'] = float(df.tapir_reprojection_px.median())
        out['median_stitch_tapir_disagreement_px'] = float(df.stitch_tapir_disagreement_px.median())
        out['motion_sources'] = df.source.value_counts().to_dict()
    pano = cv2.imread(str(path / 'panorama_final.png'))
    if pano is not None:
        out['panorama_width'] = pano.shape[1]
        out['panorama_height'] = pano.shape[0]
    mask = cv2.imread(str(path / 'coverage.png'), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        out['coverage_pixels'] = int(np.count_nonzero(mask))
        out['coverage_fraction_of_canvas'] = float(np.count_nonzero(mask)/mask.size)
    if gt is not None and all(c in df for c in ('h00','h01','h02','h10','h11','h12','h20','h21','h22')):
        g = pd.read_csv(gt)
        if not {'frame','x','y'}.issubset(g.columns):
            raise ValueError('Ground truth must contain frame,x,y in the same canvas pixel coordinate system')
        merged = df.merge(g, on='frame')
        if len(merged):
            # The current ROI center is not stored: use supplied query coordinates
            # if present, otherwise origin (0,0). Explicitly a 2D landmark test.
            if not {'query_x','query_y'}.issubset(merged.columns):
                raise ValueError('Ground truth also needs query_x,query_y (point in current cropped frame)')
            u,v = merged.query_x.to_numpy(), merged.query_y.to_numpy()
            den = merged.h20.to_numpy()*u + merged.h21.to_numpy()*v + merged.h22.to_numpy()
            valid = abs(den) > 1e-8
            pred_x = (merged.h00.to_numpy()*u+merged.h01.to_numpy()*v+merged.h02.to_numpy())/np.where(valid,den,1)
            pred_y = (merged.h10.to_numpy()*u+merged.h11.to_numpy()*v+merged.h12.to_numpy())/np.where(valid,den,1)
            err = np.hypot(pred_x-merged.x.to_numpy(), pred_y-merged.y.to_numpy())[valid]
            if len(err): out['gt_landmark_rmse_px'] = float(np.sqrt(np.mean(err**2)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--integrated', required=True)
    p.add_argument('--ground-truth', default=None, help='Optional CSV frame,query_x,query_y,x,y in shared canvas coordinates')
    p.add_argument('--output', default='evaluation.json')
    a = p.parse_args()
    result = {'baseline': summarize(a.baseline), 'integrated': summarize(a.integrated, a.ground_truth)}
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    print('NOTE: latency and map coverage are comparable; no ground-truth registration accuracy is inferred from TAPIR self-consistency.')

if __name__ == '__main__': main()
