# surgipano/__init__.py
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../SurgiPano/surgipano
_ROOT = os.path.dirname(_HERE)                          # .../SurgiPano
_THIRD = os.path.join(_ROOT, "third_party")

for p in (
    _ROOT,
    _THIRD,
    os.path.join(_THIRD, "tapnet"),
    os.path.join(_THIRD, "Stitch-seg"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")