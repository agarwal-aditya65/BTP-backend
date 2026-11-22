"""Sanity-check trained weights BEFORE starting the server.

Run from the backend/ folder:
    python scripts/check_weights.py                      # checks weights only
    python scripts/check_weights.py path/to/test.jpg     # + full pipeline, saves out.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import config  # noqa: E402
from app.models.classifier_model import WeatherClassifier  # noqa: E402
from app.models.enhancement_model import build_generator  # noqa: E402
from app.services.classifier import ClassifierService  # noqa: E402
from app.services.enhancer import EnhancerService  # noqa: E402
from app.utils.checkpoint import load_weights  # noqa: E402

ok = True
jobs = [("classifier", config.CLASSIFIER_WEIGHTS, WeatherClassifier(config.NUM_CLASSES))]
jobs += [(f"{k} enhancer", c["weights"], build_generator(c["norm"])) for k, c in config.ENHANCERS.items()]

for name, fname, model in jobs:
    path = config.WEIGHTS_DIR / fname
    if not path.exists():
        print(f"[MISSING] {name:18s} {path}")
        ok = False
        continue
    try:
        load_weights(model, path, torch.device("cpu"))
        print(f"[ OK ]    {name:18s} {fname}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL]    {name:18s} {fname}\n{exc}\n")
        ok = False

if ok and len(sys.argv) > 1:
    img = Image.open(sys.argv[1]).convert("RGB")
    clf, enh = ClassifierService(), EnhancerService()
    clf.load(); enh.load()
    label, conf = clf.classify_with_confidence(img)
    print(f"\nPredicted: {label} ({conf:.1%})  | class order = {config.CLASS_NAMES}")
    enh.enhance(img, label).save("out.png")
    print("Saved enhanced image to out.png")

sys.exit(0 if ok else 1)
