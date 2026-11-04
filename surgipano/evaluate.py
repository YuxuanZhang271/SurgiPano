"""Compare three SurgiPano pipelines; landmark ground truth is optional."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def summarize(path, gt=None):
    path = Path(path)
    df = pd.read_csv(path/'metrics.csv')
    if df.empty:
        return {'frames':0}
    seconds = pd.to_numeric(df.seconds, errors='coerce')
    result = dict(frames=int(len(df)), mean_fps=float(len(df)/seconds.sum()) if seconds.sum()>0 else None,
                  median_latency_ms=float(seconds.median()*1000),
                  p95_latency_ms=float(seconds.quantile(.95)*1000))
    for col in ('stitch_valid', 'tapir_tracks', 'tapir_inliers', 'tapir_reprojection_px',
                'stitch_tapir_disagreement_px', 'tapir_active', 'tapir_median_age',
                'tapir_mean_confidence', 'tapir_born', 'tapir_retired',
                'deform_active', 'deform_inliers', 'deform_rmse_px'):
        if col in df:
            result['mean_'+col] = float(pd.to_numeric(df[col], errors='coerce').mean())
    if 'source' in df:
        result['motion_sources'] = {str(k):int(v) for k,v in df.source.value_counts().items()}
    pano = cv2.imread(str(path/'panorama_final.png'))
    if pano is not None:
        result['panorama_width'], result['panorama_height'] = pano.shape[1], pano.shape[0]
    mask = cv2.imread(str(path/'coverage.png'), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        result['coverage_pixels'] = int(np.count_nonzero(mask))
        result['coverage_fraction_of_canvas'] = float(np.count_nonzero(mask)/mask.size)
    if gt is not None:
        required = ['h'+str(r)+str(c) for r in range(3) for c in range(3)]
        if not set(required).issubset(df.columns):
            result['gt_note'] = 'No frame-to-canvas homography recorded for this run'
        else:
            g = pd.read_csv(gt)
            if not {'frame','query_x','query_y','x','y'}.issubset(g):
                raise ValueError('Ground truth requires frame,query_x,query_y,x,y')
            merged = df.merge(g, on='frame')
            if len(merged):
                u,v = merged.query_x.to_numpy(), merged.query_y.to_numpy()
                den = merged.h20.to_numpy()*u+merged.h21.to_numpy()*v+merged.h22.to_numpy()
                valid = np.isfinite(den)&(abs(den)>1e-8)
                x = (merged.h00.to_numpy()*u+merged.h01.to_numpy()*v+merged.h02.to_numpy())/np.where(valid,den,1)
                y = (merged.h10.to_numpy()*u+merged.h11.to_numpy()*v+merged.h12.to_numpy())/np.where(valid,den,1)
                error = np.hypot(x-merged.x.to_numpy(), y-merged.y.to_numpy())[valid]
                if len(error):
                    result['gt_homography_landmark_rmse_px'] = float(np.sqrt(np.mean(error**2)))
                result['gt_note'] = 'Homography-only GT evaluation; local deformation not included'
    return result


def clean(obj):
    if isinstance(obj, dict):
        return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--tapir-stitch', '--integrated', dest='tapir_stitch', required=True)
    p.add_argument('--emdq-tapir-stitch', default=None)
    p.add_argument('--ground-truth', default=None)
    p.add_argument('--output', default='evaluation.json')
    args = p.parse_args()
    results = {'baseline':summarize(args.baseline),
               'tapir_stitch':summarize(args.tapir_stitch, args.ground_truth)}
    if args.emdq_tapir_stitch:
        results['emdq_tapir_stitch'] = summarize(args.emdq_tapir_stitch, args.ground_truth)
    results = clean(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps(results, indent=2, allow_nan=False))
    print('TAPIR self-consistency is not registration ground truth; deformation-aware landmark evaluation requires saved dense maps.')

if __name__ == '__main__':
    main()
