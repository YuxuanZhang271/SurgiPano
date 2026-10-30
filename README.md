# SurgiPano

Surgical and endoscopic video panorama stitching built on top of
[Stitch-seg](https://github.com/JmeesInc/Stitch-seg), enhanced with
[TAPIR](https://github.com/google-deepmind/tapnet) point tracking for motion
sanity checks, and [YOLO](https://github.com/ultralytics/ultralytics)
segmentation for tool/port masking.

## What it does

- **Alignment (primary):** Stitch-seg's ALIKED + LightGlue homography.
- **Alignment (optional hint):** causal TAPIR tracking of a fixed grid,
  used to validate homography translation and to trigger early pipeline
  resets when the scene is static but the homography is large.
- **Segmentation:** YOLO-seg masks for tool and port regions, injected
  through Stitch-seg's existing `load_masking_model` / `load_masking_model2`
  API. The stitcher's masking contract is unchanged.
- **Benchmarking:** `evaluate.py` runs a 4-way matrix
  (baseline / tapir / yolo / tapir+yolo) and reports stitch latency,
  TAPIR latency, pipeline resets, wall time, canvas size, and ROI per
  configuration.

## Design

The two branches are strictly independent:

| Branch | Component | Toggle |
|---|---|---|
| Alignment | Stitch-seg homography | always on |
| Alignment hint | TAPIR causal tracker | `--enable_tapir 0/1` |
| Segmentation | YOLO tool/port masks | `--yolo_tool`, `--yolo_port` (paths) |

YOLO never participates in pose estimation, and TAPIR never replaces the
homography. This keeps the failure modes separable and makes the
benchmark attribution clean.

## Repository layout

    tapir_stitch/          # this package
      run_integrated.py    # demo with --enable_tapir / --yolo_* flags
      evaluate.py          # 4-way benchmark → JSON
      yolo_mask.py         # torch.nn.Module wrapper (N,1,H,W) output
      masking_setup.py     # patches Stitch-seg's mask loaders
      weights/             # YOLO-seg checkpoints
    # siblings, not vendored:
    ../tapnet/             # provides tapnet.torch + causal checkpoint
    ../Stitch-seg/         # provides stitch_seg package + weights

## Acknowledgements

- Stitch-seg (JmeesInc) — stitching pipeline and masking API.
- TAPIR / BootsTAPIR (Google DeepMind) — causal point tracking.
- YOLO (Ultralytics) — segmentation masks.