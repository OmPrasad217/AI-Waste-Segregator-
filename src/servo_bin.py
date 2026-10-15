"""
OPTIONAL hardware extension: a smart bin whose lid opens for the predicted
class - the "exhibition prototype" part of the project.  Intended for a
Raspberry Pi with three servos (one lid per bin: blue / green / red).

    python src/servo_bin.py --simulate     # demo on any PC, no hardware needed
    python src/servo_bin.py                # on a Raspberry Pi (uses gpiozero)

Wiring (example pins - any PWM-capable GPIO pins work)
------------------------------------------------------
    lid servo, DRY (blue)    -> GPIO 12
    lid servo, E-WASTE (red) -> GPIO 13
    lid servo, WET (green)   -> GPIO 18
    Servo power (red wire)   -> external 5 V supply (NOT the Pi's 5 V pin)
    Servo GND (brown wire)   -> common ground with the Pi
    Servo signal (orange)    -> the GPIO pin above

Safety: always power servos from their own 5 V supply and join the grounds.
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make the project root importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CLASS_INFO, CONFIDENCE_THRESHOLD, DEMO_MODEL_FILENAME,  # noqa: E402
                    MODEL_DIR, MODEL_FILENAME, ROI_FRACTION, SMOOTHING_FRAMES)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SERVO_PINS = {"dry": 12, "e-waste": 13, "wet": 18}   # BCM pin numbers
TRIGGER_FRAMES = 12    # consecutive confident frames before a lid opens
OPEN_SECONDS = 3.5     # how long the lid stays open
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart bin lid controller (servo hardware).")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--simulate", action="store_true",
                   help="run without any hardware (prints the actions instead)")
    return p.parse_args()


class ServoLid:
    """One lid driven by one servo. Falls back to printing in simulate mode."""

    def __init__(self, name: str, pin: int, simulate: bool):
        self.name, self.simulate = name, simulate
        self.servo = None
        if not simulate:
            try:
                from gpiozero import AngularServo
                self.servo = AngularServo(pin, min_angle=0, max_angle=180)
                self.servo.angle = 0            # closed
            except Exception as exc:            # noqa: BLE001 - give a friendly message
                raise SystemExit(
                    f"Could not set up the servo on GPIO {pin}: {exc}\n"
                    "Are you on a Raspberry Pi?  For a no-hardware demo use --simulate.")

    def open(self) -> None:
        if self.servo:
            self.servo.angle = 110              # lift the lid
        print(f"  >>> OPENING {self.name.upper()} LID")

    def close(self) -> None:
        if self.servo:
            self.servo.angle = 0
        print(f"  <<< closing {self.name} lid")


def main() -> int:
    from config import CLASSES

    args = parse_args()
    simulate = args.simulate

    # Try to set up real hardware; fall back to simulation with a notice.
    try:
        lids = {name: ServoLid(name, pin, simulate)
                for name, pin in SERVO_PINS.items()}
    except SystemExit as exc:
        print(exc)
        print("-> falling back to SIMULATION mode (no hardware will move)\n")
        lids = {name: ServoLid(name, pin, True) for name, pin in SERVO_PINS.items()}


    # --- model ----------------------------------------------------------
    model_path = args.model or (MODEL_DIR / MODEL_FILENAME
                                if (MODEL_DIR / MODEL_FILENAME).exists()
                                else MODEL_DIR / DEMO_MODEL_FILENAME)
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    img_size = tuple(int(v) for v in model.input_shape[1:3])
    class_names = sorted(CLASSES)
    print(f"Model: {model_path.name} | classes {class_names} | "
          f"{'SIMULATION (no hardware)' if simulate else 'servos on GPIO ' + str(SERVO_PINS)}")

    # --- camera ---------------------------------------------------------
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("ERROR: could not open the webcam (try --source 1).")
        return 1

    history = deque(maxlen=SMOOTHING_FRAMES)
    confident_streak = 0
    open_until = 0.0
    active_lid = None

    print("Smart bin running - show an item to the camera. Ctrl+C to stop.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            side = int(min(h, w) * ROI_FRACTION)
            y1, x1 = (h - side) // 2, (w - side) // 2
            crop = frame[y1:y1 + side, x1:x1 + side]

            x = cv2.resize(crop, (img_size[1], img_size[0])).astype("float32")[None]
            probs = model(x, training=False).numpy()[0]
            history.append(probs)
            smoothed = np.mean(history, axis=0)

            idx = int(np.argmax(smoothed))
            cls = class_names[idx] if idx < len(class_names) else "?"
            conf = float(smoothed[idx])

            # simple status line (this demo is about the hardware)
            cv2.putText(frame, f"{cls} {conf:.0%}", (10, 30), FONT, 0.9,
                        CLASS_INFO.get(cls, {}).get("color", (200, 200, 200)), 2)
            cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 255), 2)
            cv2.imshow("Smart bin - AI Waste Segregator", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            now = time.time()
            if now < open_until:
                continue                          # lid already open, wait
            if active_lid is not None:
                active_lid.close()
                active_lid = None

            if conf >= CONFIDENCE_THRESHOLD:
                confident_streak += 1
            else:
                confident_streak = 0

            if confident_streak >= TRIGGER_FRAMES:
                lid = lids.get(cls)
                if lid:
                    lid.open()
                    active_lid = lid
                    open_until = now + OPEN_SECONDS
                    confident_streak = 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if active_lid is not None:
            active_lid.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
