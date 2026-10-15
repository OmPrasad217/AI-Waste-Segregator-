"""
REAL-TIME waste classification from the webcam.

Hold a waste item inside the square.  The CNN predicts the class of every
frame, the last few predictions are averaged for stability, and the app
tells you which bin to use (blue = dry, green = wet, red = e-waste).

Usage
-----
    python src/predict_webcam.py                          # your trained model, or the demo model
    python src/predict_webcam.py --model models/waste_segregator_demo.keras
    python src/predict_webcam.py --image photo.jpg        # single photo, no webcam needed
    python src/predict_webcam.py --model model.h5 --labels labels.txt   # Teachable Machine export
    python src/predict_webcam.py --normalize              # only for models WITHOUT a rescaling layer

Keys (in the camera window)
---------------------------
    q / ESC   quit          s   save a snapshot to captures/
    m         mirror on/off
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make the project root importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CAPTURE_DIR, CLASS_INFO, CLASSES, CONFIDENCE_THRESHOLD,  # noqa: E402
                    DEMO_MODEL_FILENAME, MODEL_DIR, MODEL_FILENAME,
                    ROI_FRACTION, SMOOTHING_FRAMES)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time AI waste segregator (webcam).")
    p.add_argument("--model", type=Path, default=None,
                   help="path to a .keras/.h5 model (default: your trained model, else the demo model)")
    p.add_argument("--labels", type=Path, default=None,
                   help="text file with one class name per line (e.g. Teachable Machine labels.txt)")
    p.add_argument("--source", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--image", type=Path, default=None,
                   help="classify a single image file instead of the live camera")
    p.add_argument("--show", action="store_true",
                   help="with --image: also open a window with the result")
    p.add_argument("--normalize", action="store_true",
                   help="divide pixels by 255 before feeding the model "
                        "(only needed if the model has NO built-in rescaling)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def resolve_model_path(arg: Path) -> Path:
    if arg is not None:
        if not arg.exists():
            raise SystemExit(f"ERROR: model not found: {arg}")
        return arg
    own = MODEL_DIR / MODEL_FILENAME
    if own.exists():
        return own
    demo = MODEL_DIR / DEMO_MODEL_FILENAME
    if demo.exists():
        print(f"NOTE: using the built-in demo model ({demo.name}).")
        print("      For the real project, collect data + run train.py first!\n")
        return demo
    raise SystemExit("ERROR: no model found in models/. Run  python src/train.py  first.")


def load_class_names(model_path: Path, labels_file: Path, n_outputs: int) -> list:
    """Decide the class-name order that matches the model's output neurons."""
    if labels_file is not None:                      # 1. explicit --labels file
        names = [l.strip() for l in labels_file.read_text().splitlines() if l.strip()]
    else:
        meta = model_path.with_name(model_path.stem + "_metadata.json")
        if meta.exists():                            # 2. metadata written by train.py
            names = json.loads(meta.read_text())["class_names"]
        else:                                        # 3. config fallback
            names = CLASSES
    if len(names) != n_outputs:
        raise SystemExit(f"ERROR: model has {n_outputs} outputs but {len(names)} "
                         f"class names were given ({names}).")
    return names


def load_model(path: Path):
    from tensorflow import keras
    return keras.models.load_model(path)


def preprocess(crop: np.ndarray, img_size, normalize: bool) -> np.ndarray:
    """Centre-square crop -> resize -> float32 batch of 1."""
    x = cv2.resize(crop, (img_size[1], img_size[0]))          # (w, h) for OpenCV
    if x.shape[2] == 1:                                        # just in case
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    x = x.astype("float32")
    if normalize:
        x = x / 255.0
    return x[None, ...]


def draw_overlay(frame, box, smoothed, class_names, fps=None):
    """Draw the ROI box, the prediction and the bin recommendation."""
    (x1, y1, x2, y2) = box
    label_idx = int(np.argmax(smoothed))
    confidence = float(smoothed[label_idx])
    certain = confidence >= CONFIDENCE_THRESHOLD

    if certain:
        cls = class_names[label_idx]
        info = CLASS_INFO.get(cls, {"display": cls.upper(), "bin": "check locally",
                                    "color": (200, 200, 200)})
        colour = info["color"]
        line1 = f"{info['display']}  {confidence:0.0%}"
        line2 = f"Dispose in: {info['bin']}"
    else:
        colour = (120, 120, 120)
        line1 = "NOT SURE YET"
        line2 = "centre the item, get closer, add light"

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)

    # banner above the box
    (tw1, _), _ = cv2.getTextSize(line1, FONT, 0.9, 2)
    (tw2, _), _ = cv2.getTextSize(line2, FONT, 0.6, 1)
    bw = max(tw1, tw2) + 20
    bh = 74
    by = max(0, y1 - bh - 6)
    cv2.rectangle(frame, (x1, by), (x1 + bw, by + bh), (25, 25, 25), -1)
    cv2.rectangle(frame, (x1, by), (x1 + bw, by + bh), colour, 2)
    cv2.putText(frame, line1, (x1 + 10, by + 28), FONT, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, line2, (x1 + 10, by + 58), FONT, 0.6, (220, 220, 220), 1)

    if fps is not None:
        cv2.putText(frame, f"{fps:4.1f} FPS", (10, 22), FONT, 0.55, (0, 255, 255), 1)
    return certain


def open_camera(index: int):
    """Open a webcam, retrying with the DirectShow backend on Windows."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cap


# ---------------------------------------------------------------------------
# single-image mode
# ---------------------------------------------------------------------------
def run_image(args, model, class_names, img_size) -> int:
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"ERROR: could not read image {args.image}")
        return 1
    h, w = img.shape[:2]
    side = int(min(h, w) * ROI_FRACTION)
    box = ((w - side) // 2, (h - side) // 2,
           (w - side) // 2 + side, (h - side) // 2 + side)
    crop = img[box[1]:box[3], box[0]:box[2]]

    probs = model(preprocess(crop, img_size, args.normalize), training=False).numpy()[0]
    draw_overlay(img, box, probs, class_names)

    idx = int(np.argmax(probs))
    cls = class_names[idx]
    print(f"Prediction : {CLASS_INFO.get(cls, {}).get('display', cls)}  ({probs[idx]:.1%})")
    print(f"Dispose in : {CLASS_INFO.get(cls, {}).get('bin', 'check locally')}")
    print("All probs  : " + ", ".join(f"{n}={p:.2f}" for n, p in zip(class_names, probs)))

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = CAPTURE_DIR / f"prediction_{args.image.stem}.jpg"
    cv2.imwrite(str(out), img)
    print(f"Annotated image saved to {out}")

    if args.show:
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# live webcam mode
# ---------------------------------------------------------------------------
def run_webcam(args, model, class_names, img_size) -> int:
    cap = open_camera(args.source)
    if not cap.isOpened():
        print("ERROR: could not open the webcam.")
        print(" * Close other apps using the camera, or try:  --source 1")
        print(" * No camera? Test with a photo instead:       --image photo.jpg")
        return 1

    history = deque(maxlen=SMOOTHING_FRAMES)
    mirror = True
    fps, last_t = 0.0, time.time()

    print("Camera ready - hold a waste item inside the square.")
    print("Keys: q/ESC quit | s snapshot | m mirror\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: could not read a frame from the camera.")
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        side = int(min(h, w) * ROI_FRACTION)
        x1, y1 = (w - side) // 2, (h - side) // 2
        box = (x1, y1, x1 + side, y1 + side)
        crop = frame[y1:y1 + side, x1:x1 + side]

        probs = model(preprocess(crop, img_size, args.normalize), training=False).numpy()[0]
        history.append(probs)
        smoothed = np.mean(history, axis=0)
        certain = draw_overlay(frame, box, smoothed, class_names, fps)

        # footer
        footer = "q=quit  s=snapshot  m=mirror  |  AI Waste Segregator"
        cv2.putText(frame, footer, (10, h - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if certain:
            cv2.putText(frame, "hold the item inside the square",
                        (w - 330, 22), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - last_t))
        last_t = now

        cv2.imshow("AI Waste Segregator - live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            idx = int(np.argmax(smoothed))
            name = f"snapshot_{class_names[idx]}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(CAPTURE_DIR / name), frame)
            print(f"  snapshot -> captures/{name}")
        elif key == ord("m"):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = load_model(model_path)

    img_size = tuple(int(v) for v in model.input_shape[1:3])   # (h, w) from the model
    class_names = load_class_names(model_path, args.labels, model.output_shape[-1])

    print(f"Model   : {model_path}")
    print(f"Classes : {class_names}")
    print(f"Input   : {img_size[0]}x{img_size[1]}\n")

    if args.image is not None:
        return run_image(args, model, class_names, img_size)
    return run_webcam(args, model, class_names, img_size)


if __name__ == "__main__":
    raise SystemExit(main())
