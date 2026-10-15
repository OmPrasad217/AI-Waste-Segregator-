"""
Generate a small SYNTHETIC dataset so the whole pipeline can be tested
without a webcam and without downloading anything.

The images are procedurally drawn textures:
    dry     -> bright paper / plastic sheets with fold lines
    wet     -> moist green-brown organic blobs
    e-waste -> dark-green circuit board with silver traces, chips, gold pins

IMPORTANT: these are NOT photographs.  A model trained on them only learns
colour + texture cues; it is meant to prove that the software pipeline
(collect -> train -> evaluate -> live demo) works end-to-end.  For the real
project, collect real photos with scripts/capture_dataset.py and retrain.

Usage
-----
    python scripts/make_synthetic_dataset.py
    python scripts/make_synthetic_dataset.py --per-class 400 --out dataset
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the project root importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATASET_DIR, SEED  # noqa: E402

IMG = 128  # generated image size (train.py resizes to whatever the model needs)


# --------------------------------------------------------------------------
# texture painters (all colours are BGR)
# --------------------------------------------------------------------------
def _wet(rng: np.random.Generator) -> np.ndarray:
    """Wet / organic waste: moist kitchen-waste blobs in greens & browns."""
    img = np.zeros((IMG, IMG, 3), np.uint8)
    img[:] = rng.choice([(45, 55, 35), (60, 50, 35), (35, 50, 30), (70, 60, 40)])

    # organic blobs: greens, yellow-greens, browns
    blob_colours = [(50, 120, 45), (45, 95, 70), (30, 70, 110), (35, 80, 140),
                    (70, 140, 60), (25, 60, 120), (85, 120, 70), (40, 100, 30)]
    for _ in range(rng.integers(6, 14)):
        colour = blob_colours[int(rng.integers(len(blob_colours)))]
        centre = tuple(int(v) for v in rng.integers([5, 5], [IMG - 5, IMG - 5]))
        axes = tuple(int(v) for v in rng.integers(IMG // 14, IMG // 3, 2))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(img, centre, axes, angle, 0, 360, colour, -1)

    # moist dark spots
    for _ in range(rng.integers(10, 26)):
        centre = tuple(int(v) for v in rng.integers([0, 0], [IMG, IMG]))
        axes = tuple(int(v) for v in rng.integers(2, 9, 2))
        cv2.ellipse(img, centre, axes, int(rng.integers(0, 180)), 0, 360,
                    (20, 30, 20), -1)

    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(1.5, 4.0)))
    return img


def _dry(rng: np.random.Generator) -> np.ndarray:
    """Dry waste: bright paper / plastic sheets with folds and printed patches."""
    base_tones = [(235, 232, 225), (240, 238, 235), (225, 225, 230),
                  (212, 216, 226), (232, 226, 214), (222, 228, 238)]
    img = np.zeros((IMG, IMG, 3), np.uint8)
    img[:] = base_tones[int(rng.integers(len(base_tones)))]

    # large overlapping "sheets" (quads) with slightly different tones
    for _ in range(rng.integers(2, 5)):
        pts = np.array([[rng.integers(-20, IMG + 20), rng.integers(-20, IMG + 20)]
                        for _ in range(4)], np.int32)
        tone = np.clip(np.array(img[0, 0], float)
                       + rng.normal(0, 12, 3), 0, 255).astype(np.uint8)
        cv2.fillPoly(img, [pts], tuple(int(c) for c in tone))
        cv2.polylines(img, [pts], True, (150, 148, 145), 1)  # crease/edge

    # fold lines
    for _ in range(rng.integers(3, 9)):
        p1 = (int(rng.integers(0, IMG)), int(rng.integers(0, IMG)))
        p2 = (int(rng.integers(0, IMG)), int(rng.integers(0, IMG)))
        shade = int(rng.choice([170, 180, 195, 210, 225]))
        cv2.line(img, p1, p2, (shade, shade, shade), 1)

    # occasional printed packaging patch
    for _ in range(int(rng.integers(0, 4))):
        x, y = int(rng.integers(5, IMG - 40)), int(rng.integers(5, IMG - 40))
        wq, hq = int(rng.integers(15, 45)), int(rng.integers(15, 45))
        colour = (int(rng.integers(40, 220)),) * 3 if rng.random() < 0.4 else \
                 tuple(int(v) for v in rng.integers(40, 220, 3))
        cv2.rectangle(img, (x, y), (x + wq, y + hq), colour, -1)

    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.5)))
    return img


def _ewaste(rng: np.random.Generator) -> np.ndarray:
    """E-waste: dark-green PCB with silver traces, vias, chips and gold pins."""
    pcb_greens = [(45, 95, 25), (40, 85, 30), (55, 105, 20), (35, 75, 35), (60, 100, 45)]
    img = np.zeros((IMG, IMG, 3), np.uint8)
    img[:] = pcb_greens[int(rng.integers(len(pcb_greens)))]

    # silver PCB traces: L-shaped, axis-aligned polylines
    silver = (185, 190, 195)
    for _ in range(rng.integers(15, 35)):
        x, y = int(rng.integers(0, IMG)), int(rng.integers(0, IMG))
        pts = [(x, y)]
        for _ in range(int(rng.integers(2, 5))):
            step = int(rng.integers(-25, 25))
            if rng.random() < 0.5:
                x = int(np.clip(x + step, 0, IMG - 1))
            else:
                y = int(np.clip(y + step, 0, IMG - 1))
            pts.append((x, y))
        cv2.polylines(img, [np.array(pts, np.int32)], False, silver,
                      int(rng.integers(1, 3)))

    # vias / pads: tiny silver or gold circles
    for _ in range(rng.integers(15, 40)):
        centre = (int(rng.integers(0, IMG)), int(rng.integers(0, IMG)))
        colour = silver if rng.random() < 0.7 else (40, 160, 210)  # gold in BGR
        cv2.circle(img, centre, int(rng.integers(1, 4)), colour, -1)

    # chips: black rectangles with a lighter border
    for _ in range(int(rng.integers(2, 7))):
        wq, hq = int(rng.integers(18, 55)), int(rng.integers(12, 40))
        x, y = int(rng.integers(0, IMG - wq - 1)), int(rng.integers(0, IMG - hq - 1))
        cv2.rectangle(img, (x, y), (x + wq, y + hq), (25, 25, 28), -1)
        cv2.rectangle(img, (x, y), (x + wq, y + hq), (120, 125, 130), 1)
        # gold pins along one edge
        pins = max(2, wq // 7)
        for i in range(pins):
            px = x + 2 + i * (wq // pins)
            cv2.rectangle(img, (px, y + hq + 1), (px + 2, y + hq + 4), (40, 160, 210), -1)

    # connector / gold pad strip
    if rng.random() < 0.7:
        x, y = int(rng.integers(0, IMG - 30)), int(rng.integers(0, IMG - 8))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(18, 40)), y + 5), (40, 160, 210), -1)

    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.2)))
    return img


PAINTERS = {"dry": _dry, "wet": _wet, "e-waste": _ewaste}


# --------------------------------------------------------------------------
# common random augmentation
# --------------------------------------------------------------------------
def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    # small rotation
    angle = float(rng.uniform(-20, 20))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, float(rng.uniform(0.9, 1.1)))
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # mirror
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    # brightness / contrast jitter
    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-18, 18))
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # gaussian noise
    noise = rng.normal(0, float(rng.uniform(3, 9)), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-class", type=int, default=250,
                   help="images per class (default: %(default)s)")
    p.add_argument("--out", type=Path, default=DATASET_DIR,
                   help="output dataset folder (default: ./dataset)")
    p.add_argument("--seed", type=int, default=SEED,
                   help="random seed for reproducibility (default: %(default)s)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for cls in CLASSES:
        folder = args.out / cls
        folder.mkdir(parents=True, exist_ok=True)
        painter = PAINTERS[cls]
        for i in range(args.per_class):
            img = augment(painter(rng), rng)
            cv2.imwrite(str(folder / f"synth_{i:04d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {cls:>8}: {args.per_class} images -> {folder}")

    print(f"\nSynthetic dataset ready in {args.out}")
    print("NOTE: synthetic images are for pipeline testing only.")
    print("Next step:  python src/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
