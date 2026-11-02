# SurgiPano experiment harness

Three scripts: `run_stitch.py` (Stitch baseline), `run_integrated.py` (continuous TAPIR + Stitch external geometric fusion), `evaluate.py` (metrics).

## Installation

Use an environment where **both** upstream `stitch_seg` and `tapnet` are importable and their own dependencies are installed. These packages are **not** installed by `requirements.txt` alone. For example, install upstream repos in editable mode from their respective checkouts, following their READMEs. The TAPIR demo checkpoint should be `causal_bootstapir_checkpoint.pt` or a compatible causal checkpoint. Verify imports:

```bash
python -c 'from stitch_seg import StitchInferencer; from tapnet.torch import tapir_model; print("OK")'
pip install -r requirements.txt
```

Run from the folder containing these scripts, and replace paths:

```bash
python surgipano/run_stitch.py \
  --video ../../Datasets/002_S8a_videos_480p/S01_10_00.mp4 \
  --scope-mask ../../Datasets/002_S8a_videos_480p/S01_mask.png \
  --output-dir outputs/stitch \
  --aliked-weights third_party/Stitch-seg/weights/aliked-n16.pth \
  --lightglue-weights third_party/Stitch-seg/weights/aliked_lightglue_v0-1_arxiv.pth \
  --preview-width 1600 --preview-height 720 \
  --max-frames 300

python surgipano/run_integrated.py \
  --video ../../Datasets/002_S8a_videos_480p/S01_10_00.mp4 \
  --scope-mask ../../Datasets/002_S8a_videos_480p/S01_mask.png \
  --output-dir outputs/integrated \
  --aliked-weights third_party/Stitch-seg/weights/aliked-n16.pth \
  --lightglue-weights third_party/Stitch-seg/weights/aliked_lightglue_v0-1_arxiv.pth \
  --tapir-checkpoint third_party/tapnet/checkpoints/causal_bootstapir_checkpoint_prev.pt \
  --preview-width 1600 --preview-height 720 \
  --canvas-view stitch \
  --max-frames 300

python surgipano/evaluate.py --baseline outputs/stitch --integrated outputs/integrated \
  --output outputs/evaluation.json
```

## Important implementation boundaries

- TAPIR runs on every processed frame, not just when Stitch fails. TAPIR is **sparse point tracking**, not dense optical flow.
- Fusion occurs **outside** `StitchInferencer`: the independent panorama is composited from the geometric transform selected/refined using Stitch and TAPIR. This does not alter Stitch's internal matcher or its own panorama.
- Stitch's `_current_to_canvas_h()` is an internal API and can change across revisions. The implementation assumes its coordinates correspond to the cropped/preprocessed frame, and that successive Stitch transforms share a stable canvas coordinate system. A Stitch reset can invalidate this assumption. For rigorous testing, instrument the Stitch internals to expose reset events and raw match correspondences.
- `--scope-mask` in the integrated pipeline is resized to the *cropped* frame. For precise masking, crop it with `inferencer.roi` before passing it or modify the script to perform the ROI crop.
- The baseline intentionally follows the uploaded demo's missing-tool-detector behavior (zero masks). It does not currently inject a scope mask; to make scope masking identical, integrate the `run_demo.py` mask-injection logic into both scripts.
- The integrated output is a demonstration of **prior + robust geometric refinement**, not a trained model or a joint neural fusion network.
- Canvas images are RGB converted to BGR on disk. The baseline uses Stitch's own canvas, while integrated uses a separately constructed canvas; their coverage should not be compared as an accuracy metric.
- `evaluate.py` can compare runtime, TAPIR internal reprojection, and output metadata. **It cannot establish improved registration accuracy without independent landmark/camera-pose ground truth.** Ground-truth CSV is optional and needs `frame,query_x,query_y,x,y` with `x,y` in the integrated canvas coordinate system.
- For endoscopy, rigid homographies may fail under parallax, tissue deformation, forward camera translation, occlusion, and specular reflections.

## Suggested validation

1. First run 50 frames to check tensor shapes and coordinate conventions for your installed upstream versions.
2. Compare first 50 frames visually against the original Stitch demo.
3. Inspect `metrics.csv` source column: `initialize`, `fused`, `tapir`, `stitch`, or hold states.
4. Validate transforms against simulator landmark positions or known camera poses, not TAPIR's own predictions.
5. For production integration, expose raw ALIKED/LightGlue matches and use a joint RANSAC estimator, plus reset-aware keyframe management.
