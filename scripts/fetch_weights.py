#!/usr/bin/env python3
# scripts/fetch_weights.py
"""Download the external weights SurgiPano needs.

By default this fetches:
  * TAPIR / BootsTAPIR causal checkpoint   -> third_party/tapnet/checkpoints/
  * ALIKED feature extractor               -> third_party/Stitch-seg/weights/
  * LightGlue (ALIKED) matcher             -> third_party/Stitch-seg/weights/

YOLO segmentation weights are NOT downloaded automatically because they
are task-specific (your tool/port classes). Place them under
surgipano/weights/ manually, or pass --yolo_placeholder to pull a stock
Ultralytics YOLO-seg model just for wiring/testing.

Usage:
    python scripts/fetch_weights.py                 # fetch all upstream weights
    python scripts/fetch_weights.py --skip_tapir    # only stitcher weights
    python scripts/fetch_weights.py --skip_stitch   # only TAPIR
    python scripts/fetch_weights.py --yolo_placeholder
    python scripts/fetch_weights.py --force         # re-download even if present
    python scripts/fetch_weights.py --verify        # check sizes/checksums only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (must match the submodule layout from the bootstrap instructions)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
THIRD = ROOT / "third_party"
TAPNET = THIRD / "tapnet"
STITCH = THIRD / "Stitch-seg"
SURGI = ROOT / "surgipano"

TAPIR_DST = TAPNET / "checkpoints" / "causal_bootstapir_checkpoint.pt"
ALIKED_DST = STITCH / "weights" / "aliked-n16.pth"
LIGHTGLUE_DST = STITCH / "weights" / "aliked_lightglue_v0-1_arxiv.pth"
YOLO_PLACEHOLDER_DST = SURGI / "weights" / "yolo11n-seg.pt"

# ---------------------------------------------------------------------------
# Source URLs
#
# BootsTAPIR checkpoint: the official DM-TAPNet release. If this URL ever
# moves, check https://github.com/google-deepmind/tapnet#checkpoints
# ---------------------------------------------------------------------------
TAPIR_URL = (
    "https://storage.googleapis.com/dm-tapnet/bootstap/"
    "bootstapir_checkpoint_v2.pt"
)
ALIKED_URL = (
    "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"
)
LIGHTGLUE_URL = (
    "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/"
    "aliked_lightglue.pth"
)
# Ultralytics hosts its stock models on GitHub releases. YOLO() also
# auto-downloads, but going through this script keeps the layout explicit.
YOLO_PLACEHOLDER_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    "yolo11n-seg.pt"
)

# Expected minimum sizes (MB). Used by --verify to catch truncated downloads.
# Values are conservative — anything smaller is definitely broken.
MIN_MB = {
    TAPIR_DST: 30,
    ALIKED_DST: 10,
    LIGHTGLUE_DST: 3,
    YOLO_PLACEHOLDER_DST: 1,
}


def _fmt(mb: float) -> str:
    return f"{mb:.1f} MB"


def _download(url: str, dst: Path, force: bool = False) -> bool:
    """Download url -> dst atomically. Returns True if a fetch happened."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0 and not force:
        print(f"[skip] {dst.relative_to(ROOT)} "
              f"({_fmt(dst.stat().st_size / 1e6)})")
        return False

    print(f"[get ] {url}")
    print(f"       -> {dst.relative_to(ROOT)}")
    try:
        # Download to a sibling temp file, then rename — avoids leaving a
        # half-written checkpoint if the connection drops.
        with tempfile.NamedTemporaryFile(
                dir=dst.parent, prefix=dst.name, suffix=".part",
                delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(url) as resp:
                total = resp.headers.get("Content-Length")
                total_b = int(total) if total else None
                done = 0
                chunk = 1 << 20  # 1 MiB
                last_print = 0.0
                import time as _t
                t0 = _t.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    tmp.write(buf)
                    done += len(buf)
                    now = _t.time()
                    if now - last_print > 1.0:
                        if total_b:
                            pct = 100.0 * done / total_b
                            print(f"       {_fmt(done/1e6)} / "
                                  f"{_fmt(total_b/1e6)} ({pct:5.1f}%)",
                                  end="\r")
                        else:
                            print(f"       {_fmt(done/1e6)}", end="\r")
                        last_print = now
                dt = max(_t.time() - t0, 1e-6)
                print(f"\n       done: {_fmt(done/1e6)} "
                      f"in {dt:.1f}s ({done/1e6/dt:.1f} MB/s)")
        os.replace(tmp_path, dst)
    except Exception:
        # Clean up partial file on failure.
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass
        raise
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path) -> bool:
    if not path.is_file():
        print(f"[miss] {path.relative_to(ROOT)}")
        return False
    size_mb = path.stat().st_size / 1e6
    min_mb = MIN_MB.get(path, 0)
    if size_mb < min_mb:
        print(f"[bad ] {path.relative_to(ROOT)} — {_fmt(size_mb)} "
              f"< expected >= {_fmt(min_mb)} (truncated?)")
        return False
    print(f"[ ok ] {path.relative_to(ROOT):<60s} "
          f"{_fmt(size_mb):>10s}  sha256={_sha256(path)[:16]}…")
    return True


def _fetch_tapir(force: bool) -> None:
    print("\n--- TAPIR / BootsTAPIR checkpoint ---")
    if not TAPNET.is_dir():
        print(f"[error] {TAPNET.relative_to(ROOT)} missing — did you run\n"
              f"        git submodule update --init --recursive ?")
        sys.exit(2)
    _download(TAPIR_URL, TAPIR_DST, force=force)


def _fetch_stitch(force: bool) -> None:
    print("\n--- Stitch-seg feature / matcher weights ---")
    if not STITCH.is_dir():
        print(f"[error] {STITCH.relative_to(ROOT)} missing — did you run\n"
              f"        git submodule update --init --recursive ?")
        sys.exit(2)
    _download(ALIKED_URL, ALIKED_DST, force=force)
    _download(LIGHTGLUE_URL, LIGHTGLUE_DST, force=force)


def _fetch_yolo_placeholder(force: bool) -> None:
    print("\n--- YOLO-seg placeholder (for wiring/tests only) ---")
    print("[note] This is a COCO YOLO11n-seg model. It will happily run "
          "inside SurgiPano,")
    print("       but it has NO tool/port classes. Replace it with your own "
          "weights at")
    print(f"       {SURGI.relative_to(ROOT)}/tool_yolo_seg.pt and "
          f"{SURGI.relative_to(ROOT)}/port_yolo_seg.pt")
    _download(YOLO_PLACEHOLDER_URL, YOLO_PLACEHOLDER_DST, force=force)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch external weights for SurgiPano.")
    ap.add_argument("--skip_tapir", action="store_true",
                    help="Skip the TAPIR / BootsTAPIR checkpoint.")
    ap.add_argument("--skip_stitch", action="store_true",
                    help="Skip ALIKED and LightGlue weights.")
    ap.add_argument("--yolo_placeholder", action="store_true",
                    help="Also fetch a stock YOLO11n-seg for smoke-testing.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the file already exists.")
    ap.add_argument("--verify", action="store_true",
                    help="Only check existing files; do not download.")
    args = ap.parse_args()

    if not args.skip_tapir:
        if args.verify:
            print("\n--- TAPIR / BootsTAPIR checkpoint ---")
            _verify(TAPIR_DST)
        else:
            _fetch_tapir(args.force)

    if not args.skip_stitch:
        if args.verify:
            print("\n--- Stitch-seg feature / matcher weights ---")
            _verify(ALIKED_DST)
            _verify(LIGHTGLUE_DST)
        else:
            _fetch_stitch(args.force)

    if args.yolo_placeholder:
        if args.verify:
            print("\n--- YOLO-seg placeholder ---")
            _verify(YOLO_PLACEHOLDER_DST)
        else:
            _fetch_yolo_placeholder(args.force)

    # Final report so the user immediately sees what's ready.
    print("\n=== status ===")
    checks = [
        ("TAPIR      ", TAPIR_DST),
        ("ALIKED     ", ALIKED_DST),
        ("LightGlue  ", LIGHTGLUE_DST),
    ]
    if args.yolo_placeholder:
        checks.append(("YOLO-seg   ", YOLO_PLACEHOLDER_DST))
    ok = True
    for label, path in checks:
        if path.is_file() and path.stat().st_size > 0:
            print(f"  {label} {path.relative_to(ROOT)} "
                  f"({_fmt(path.stat().st_size / 1e6)})")
        else:
            print(f"  {label} MISSING  {path.relative_to(ROOT)}")
            ok = False

    # Point out user-supplied YOLO weights if present.
    for name in ("tool_yolo_seg.pt", "port_yolo_seg.pt"):
        p = SURGI / "weights" / name
        if p.is_file():
            print(f"  user       {p.relative_to(ROOT)} "
                  f"({_fmt(p.stat().st_size / 1e6)})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())