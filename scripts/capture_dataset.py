"""
Collect your own waste-image dataset straight from the webcam.

This is Step 1 of building the REAL model for your project.  Hold a waste
item inside the yellow square and save frames.  Aim for 100-200 images per
class, with different objects, angles, distances, backgrounds and lighting.

Usage
-----
    python scripts/capture_dataset.py --class dry --count 150
    python scripts/capture_dataset.py --class wet --auto 0.5      # auto-click every 0.5 s
    python scripts/capture_dataset.py --class e-waste --source 1 # second camera, if any

Keys (in the camera window)
---------------------------
    SPACE   save the current frame
    a       toggle auto-save mode
    n       switch to the next class (dry -> e-waste -> wet -> dry)
    q / ESC quit

Images are saved to  dataset/<class>/NNNN_<timestamp>.jpg
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Make the project root importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, CLASS_INFO, DATASET_DIR, ROI_FRACTION  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect waste images from the webcam.")
    p.add_argument("--class", dest="cls", default=CLASSES[0], choices=CLASSES,
                   help="which class to capture (default: %(default)s)")
    p.add_argument("--count", type=int, default=150,
                   help="target number of images for this class (default: %(default)s)")
    p.add_argument("--source", type=int, default=0,
                   help="camera index (default: 0)")
    p.add_argument("--auto", type=float, default=0.0, metavar="SECONDS",
                   help="auto-save one frame every SECONDS (0 = off, press 'a' to toggle)")
    return p.parse_args()


def open_camera(index: int):
    """Open a webcam, retrying with the DirectShow backend on Windows."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cap


def existing_count(cls: str) -> int:
    folder = DATASET_DIR / cls
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])


def save_frame(frame, cls: str, index: int) -> Path:
    """Crop the centre square (what the model will actually see) and save it."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * ROI_FRACTION)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame[top:top + side, left:left + side]
    crop = cv2.resize(crop, (256, 256))  # plenty for a 96x96 or 224x224 model

    folder = DATASET_DIR / cls
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{index:04d}_{stamp}.jpg"
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def main() -> int:
    args = parse_args()
    cap = open_camera(args.source)
    if not cap.isOpened():
        print("ERROR: could not open the webcam.")
        print(" * Check that no other app is using the camera.")
        print(" * Try a different index:   python scripts/capture_dataset.py --source 1")
        return 1

    cls = args.cls
    saved = existing_count(cls)
    auto_mode = args.auto > 0
    auto_interval = args.auto if auto_mode else 0.5
    last_save = time.time()
    flash_until = 0.0

    print(f"Capturing '{cls}' | already have {saved} images | target {args.count}")
    print("Keys: SPACE=save  a=auto-save  n=next class  q=quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: could not read a frame from the camera.")
            break
        frame = cv2.flip(frame, 1)  # mirror view, feels natural
        h, w = frame.shape[:2]

        # Centre square ROI
        side = int(min(h, w) * ROI_FRACTION)
        top, left = (h - side) // 2, (w - side) // 2
        colour = CLASS_INFO[cls]["color"]
        cv2.rectangle(frame, (left, top), (left + side, top + side), (0, 255, 255), 2)

        # Header band
        band = frame[:64]
        band[:] = tuple(int(c) for c in colour)
        cv2.putText(frame, f"CLASS: {cls}", (10, 26), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"saved: {saved}/{args.count}   auto: {'ON' if auto_mode else 'OFF'}",
                    (10, 52), FONT, 0.55, (255, 255, 255), 1)

        # Footer help
        cv2.putText(frame, "SPACE=save  a=auto  n=next class  q=quit", (10, h - 12),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # "Saved!" flash
        if time.time() < flash_until:
            cv2.putText(frame, "Saved!", (left + side // 2 - 60, top - 12),
                        FONT, 0.8, (0, 255, 0), 2)

        if auto_mode and time.time() - last_save >= auto_interval:
            saved += 1
            save_frame(frame, cls, saved)
            last_save = time.time()
            flash_until = time.time() + 0.25

        cv2.imshow("Collect dataset - AI Waste Segregator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            saved += 1
            save_frame(frame, cls, saved)
            last_save = time.time()
            flash_until = time.time() + 0.35
            print(f"  saved #{saved}")
        elif key == ord("a"):
            auto_mode = not auto_mode
            last_save = time.time()
            print(f"  auto-save {'ON' if auto_mode else 'OFF'}")
        elif key == ord("n"):
            cls = CLASSES[(CLASSES.index(cls) + 1) % len(CLASSES)]
            saved = existing_count(cls)
            print(f"\nSwitched to '{cls}' | already have {saved} images")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Dataset folder: {DATASET_DIR}")
    print("Next step:  python src/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
