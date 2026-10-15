"""
Evaluate a trained model: classification report + confusion matrix.

By default the model is evaluated on the SAME validation split that
train.py held out (same seed -> identical split).  You can also point it
at a separate folder of test photos you never trained on - that is the
more honest test!

Usage
-----
    python src/evaluate.py
    python src/evaluate.py --model models/waste_segregator_demo.keras
    python src/evaluate.py --test-dir path/to/test_photos

Outputs
-------
    docs/plots/confusion_matrix.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make the project root importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (BATCH_SIZE, DATASET_DIR, IMG_SIZE, MODEL_DIR,  # noqa: E402
                    MODEL_FILENAME, PLOT_DIR, SEED, VALIDATION_SPLIT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained waste classifier.")
    p.add_argument("--model", type=Path,
                   default=MODEL_DIR / MODEL_FILENAME,
                   help="path to the .keras model to evaluate")
    p.add_argument("--test-dir", type=Path, default=None,
                   help="optional folder of unseen test images (one sub-folder per class)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def load_class_names(model_path: Path) -> list:
    """Class order that matches the model's output neurons.

    Priority: sibling <model>_metadata.json (written by train.py)
              -> sorted sub-folders of the dataset.
    """
    meta_path = model_path.with_name(model_path.stem + "_metadata.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())["class_names"]
    if DATASET_DIR.exists():
        return sorted(d.name for d in DATASET_DIR.iterdir() if d.is_dir())
    raise SystemExit("ERROR: no metadata next to the model and no ./dataset folder - "
                     "cannot determine the class order. Re-run train.py or pass --test-dir.")


def gather_predictions(model, dataset, class_names):
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_pred.extend(int(np.argmax(p)) for p in probs)
        y_true.extend(int(l) for l in labels.numpy())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(cm, class_names, accuracy, out_path: Path) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), class_names, rotation=20)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix (accuracy {accuracy:.1%})")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13)
    fig.colorbar(im)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    import tensorflow as tf
    from tensorflow import keras

    args = parse_args()
    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        print("Train one first:  python src/train.py")
        return 1

    class_names = load_class_names(args.model)
    print(f"Model   : {args.model}")
    print(f"Classes : {class_names}")

    model = keras.models.load_model(args.model)
    img_size = tuple(int(v) for v in model.input_shape[1:3])  # read from the model itself
    print(f"Input   : {img_size[0]}x{img_size[1]}")

    if args.test_dir:
        dataset = keras.utils.image_dataset_from_directory(
            args.test_dir, image_size=img_size,
            batch_size=args.batch_size, shuffle=False)
        if dataset.class_names != class_names:
            print(f"WARNING: test folder classes {dataset.class_names} "
                  f"differ from model classes {class_names}")
    else:
        # reproduce exactly the validation split used during training
        dataset = keras.utils.image_dataset_from_directory(
            DATASET_DIR, validation_split=VALIDATION_SPLIT, subset="validation",
            seed=SEED, image_size=img_size, batch_size=args.batch_size)

    y_true, y_pred = gather_predictions(model, dataset, class_names)
    accuracy = float((y_true == y_pred).mean())
    print(f"\nOverall accuracy: {accuracy:.1%}  ({len(y_true)} images)\n")

    try:
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    except ImportError:
        print("TIP: pip install scikit-learn for a detailed per-class report.")
        for i, name in enumerate(class_names):
            mask = y_true == i
            correct = int((y_pred[mask] == i).sum())
            print(f"  {name:>8}: {correct}/{int(mask.sum())} correct")

    cm = np.zeros((len(class_names), len(class_names)), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    out = PLOT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, accuracy, out)
    print(f"Confusion matrix saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
