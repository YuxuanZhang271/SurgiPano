# SurgiPano

Surgical and endoscopic video panorama stitching built on
[Stitch-seg](https://github.com/JmeesInc/Stitch-seg), enhanced with
[TAPIR](https://github.com/google-deepmind/tapnet) causal point tracking for
geometric refinement.

This repository contains an experiment harness with three scripts:

| Script | Purpose |
|---|---|
| `surgipano/run_stitch.py` | Stitch baseline — native Stitch-seg canvas |
| `surgipano/run_integrated.py` | Continuous TAPIR + Stitch geometric fusion |
| `surgipano/evaluate.py` | Metrics comparison (baseline vs. integrated) |

## What it does

- **Alignment (primary):** Stitch-seg's ALIKED + LightGlue homography.
- **Alignment (refinement):** causal TAPIR tracking of a fixed query grid,
  used to validate the Stitch transform and to robustly refine it against a
  soft Stitch prior.
- **Masking:** an optional scope mask (white = valid endoscope region) is
  cropped to the Stitch ROI and injected into the stitcher, and is also used
  to reject TAPIR queries that fall outside the visible scope.
- **Benchmarking:** `evaluate.py` compares runtime, TAPIR self-consistency,
  and output metadata between the two pipelines.

## Design

The two pipelines are strictly independent branches:

| Branch | Component | Toggle |
|---|---|---|
| Alignment | Stitch-seg homography | always on |
| Alignment refinement | TAPIR causal tracker | `--tapir-checkpoint` (required for integrated) |
| Masking | Scope mask | `--scope-mask` |

TAPIR never replaces the homography outright and never participates in
Stitch's internal matcher. The integrated pipeline composes its **own**
independent panorama outside `StitchInferencer`, using a transform selected
and refined from Stitch's candidate plus TAPIR correspondences. This keeps
failure modes separable and makes benchmark attribution clean.

### Motion source selection

Each frame in the integrated pipeline records a `source` in `metrics.csv`:

| `source` | Meaning |
|---|---|
| `initialize` | First frame; canvas transform is seeded |
| `fused` | Robust refinement of TAPIR tracks with a soft Stitch prior |
| `stitch` | Stitch transform used alone (no usable TAPIR solution) |
| `tapir` | TAPIR transform used alone (no usable Stitch candidate) |
| `hold_invalid` | Candidate transform failed validity checks; previous pose held |
| `hold_no_motion` | No valid transform; previous pose held |

## Installation

The upstream `stitch_seg` and `tapnet` packages are **not** provided by
`requirements.txt`. They live in `third_party/` as git submodules and must be
installed separately, following their upstream READMEs.

```bash
# 1. Submodules
git submodule update --init --recursive

# 2. Upstream packages (editable installs)
pip install -e third_party/tapnet[torch]
pip install -e third_party/Stitch-seg

# 3. This project's own dependencies
pip install -r requirements.txt

# 4. Verify
python -c 'from stitch_seg import StitchInferencer; from tapnet.torch import tapir_model; print("OK")'
```

### Weights

Model weights are large and are not committed. Fetch them with the helper
script:

```bash
python scripts/fetch_weights.py               # TAPIR + ALIKED + LightGlue
python scripts/fetch_weights.py --verify      # check sizes only
python scripts/fetch_weights.py --force       # re-download
```

This populates:

| File | Destination |
|---|---|
| `causal_bootstapir_checkpoint.pt` | `third_party/tapnet/checkpoints/` |
| `aliked-n16.pth` | `third_party/Stitch-seg/weights/` |
| `aliked_lightglue_v0-1_arxiv.pth` | `third_party/Stitch-seg/weights/` |

`--yolo_placeholder` additionally pulls a stock YOLO11n-seg for smoke
testing. YOLO weights are otherwise supplied by the user into
`surgipano/weights/`.

## Usage

All commands are run from the repository root. Replace the dataset paths
with your own.

### Baseline (Stitch only)

```bash
python surgipano/run_stitch.py \
  --video ../../Datasets/002_S8a_videos_480p/S01_10_00.mp4 \
  --scope-mask ../../Datasets/002_S8a_videos_480p/S01_mask.png \
  --output-dir outputs/stitch \
  --aliked-weights third_party/Stitch-seg/weights/aliked-n16.pth \
  --lightglue-weights third_party/Stitch-seg/weights/aliked_lightglue_v0-1_arxiv.pth \
  --preview-width 1600 --preview-height 720 \
  --max-frames 300
```

### Integrated (TAPIR + Stitch)

```bash
python surgipano/run_integrated.py \
  --video ../../Datasets/002_S8a_videos_480p/S01_10_00.mp4 \
  --scope-mask ../../Datasets/002_S8a_videos_480p/S01_mask.png \
  --output-dir outputs/integrated \
  --aliked-weights third_party/Stitch-seg/weights/aliked-n16.pth \
  --lightglue-weights third_party/Stitch-seg/weights/aliked_lightglue_v0-1_arxiv.pth \
  --tapir-checkpoint third_party/tapnet/checkpoints/causal_bootstapir_checkpoint.pt \
  --preview-width 1600 --preview-height 720 \
  --canvas-view stitch \
  --max-frames 300
```

### Evaluation

```bash
python surgipano/evaluate.py --baseline outputs/stitch --integrated outputs/integrated \
  --output outputs/evaluation.json
```

### Common options

Options shared by both pipelines (defined in `surgipano/common.py`):

| Option | Default | Description |
|---|---|---|
| `--video` | *required* | Input video path |
| `--output-dir` | *required* | Output directory |
| `--width` / `--height` | `640` / `480` | Frame resize before processing |
| `--max-frames` | all | Frame limit |
| `--stride` | `1` | Process every Nth frame |
| `--device` | `cuda` if available | Torch device |
| `--scope-mask` | none | Endoscope scope mask (white = valid) |
| `--aliked-weights` | `weights/aliked-n16.pth` | ALIKED feature extractor |
| `--lightglue-weights` | `weights/aliked_lightglue_v0-1_arxiv.pth` | LightGlue matcher |
| `--canvas-scale` | `3` | Canvas size multiplier |
| `--save-every` | `100` | Save a panorama every N frames |

Integrated-only options:

| Option | Default | Description |
|---|---|---|
| `--tapir-checkpoint` | *required* | Causal TAPIR checkpoint |
| `--tapir-resolution` | `256` | Tracker input resolution |
| `--tapir-spacing` | `48` | Query grid spacing in pixels |
| `--tapir-refresh` | `30` | Frames between tracker re-initialization |
| `--ransac-threshold` | `3.0` | RANSAC threshold (pixels) for TAPIR homography |
| `--stitch-weight` | `0.25` | Weight of the soft Stitch prior during refinement |
| `--max-disagreement` | `25.0` | Discard the Stitch transform above this disagreement (px) |
| `--canvas-view` | `stitch` | Which panorama to render: `stitch` or `fused` |

Both pipelines accept `--mask-margin`, `--preview-width`, `--preview-height`,
`--preview-every`, and `--no-preview`.

### Outputs

Each run writes to its `--output-dir`:

| File | Contents |
|---|---|
| `metrics.csv` | Per-frame metrics |
| `panorama_final.png` | Final panorama (`--canvas-view` selected) |
| `panorama_<frame>.png` | Periodic panorama snapshots |
| `alignment_preview.mp4` | Side-by-side source frame and panorama |
| `coverage.png` | Canvas coverage mask (integrated only) |

The integrated run additionally writes `panorama_fused_final.png` and
`panorama_stitch_final.png`, giving both renderings side by side regardless
of `--canvas-view`.

## Important implementation boundaries

- TAPIR runs on **every** processed frame, not only when Stitch fails. TAPIR
  is **sparse point tracking**, not dense optical flow.
- Fusion occurs **outside** `StitchInferencer`: the independent panorama is
  composited from the geometric transform selected/refined using Stitch and
  TAPIR. This does not alter Stitch's internal matcher or its own panorama.
- Stitch's `_current_to_canvas_h()` is an internal API and can change across
  revisions. The implementation assumes its coordinates correspond to the
  cropped/preprocessed frame, and that successive Stitch transforms share a
  stable canvas coordinate system. A Stitch reset can invalidate this
  assumption. For rigorous testing, instrument the Stitch internals to
  expose reset events and raw match correspondences.
- The scope mask is resized to the **full input frame**, then cropped using
  `StitchInferencer.roi` so it aligns with `preprocess_frame()` output. The
  crop raises if the ROI falls outside the mask bounds.
- The baseline follows the upstream demo's missing-tool-detector behavior
  (zero masks) but **does** inject the scope mask when `--scope-mask` is
  given, so masking behaviour is comparable across both pipelines.
- The integrated output is a demonstration of **prior + robust geometric
  refinement**, not a trained model or a joint neural fusion network.
- Canvas images are RGB converted to BGR on disk. The baseline uses Stitch's
  own canvas, while the integrated pipeline uses a separately constructed
  canvas; their coverage should not be compared as an accuracy metric.
- `evaluate.py` can compare runtime, TAPIR internal reprojection, and output
  metadata. **It cannot establish improved registration accuracy without
  independent landmark/camera-pose ground truth.** Ground truth is optional
  and needs a CSV with `frame,query_x,query_y,x,y`, where `x,y` is the
  landmark in the integrated canvas coordinate system and `query_x,query_y`
  is the same point in the current cropped frame.
- For endoscopy, rigid homographies may fail under parallax, tissue
  deformation, forward camera translation, occlusion, and specular
  reflections.

## Suggested validation

1. First run 50 frames to check tensor shapes and coordinate conventions for
   your installed upstream versions.
2. Compare the first 50 frames visually against the original Stitch demo.
3. Inspect the `metrics.csv` `source` column: `initialize`, `fused`, `tapir`,
   `stitch`, `hold_invalid`, or `hold_no_motion`.
4. Validate transforms against simulator landmark positions or known camera
   poses, not TAPIR's own predictions.
5. For production integration, expose raw ALIKED/LightGlue matches and use a
   joint RANSAC estimator, plus reset-aware keyframe management.

## Repository layout

    SurgiPano/
      README.md
      scripts/
        fetch_weights.py        # downloads TAPIR / ALIKED / LightGlue weights
      surgipano/
        run_stitch.py           # Stitch baseline
        run_integrated.py       # TAPIR + Stitch geometric fusion
        evaluate.py             # metrics comparison
        common.py               # shared args, homography helpers, canvas ops
        tapir_adapter.py        # causal BootsTAPIR wrapper
        mask_utils.py           # scope mask crop / erosion / injection
        video_preview.py        # side-by-side preview encoder
        requirements.txt
        weights/                # YOLO-seg checkpoints (gitignored)
      outputs/                  # run artifacts (gitignored)
      third_party/              # git submodules, not vendored
        tapnet/                 # provides tapnet.torch + causal checkpoint
        Stitch-seg/             # provides stitch_seg package + weights

## Acknowledgements

- Stitch-seg (JmeesInc) — stitching pipeline and masking API.
- TAPIR / BootsTAPIR (Google DeepMind) — causal point tracking.
