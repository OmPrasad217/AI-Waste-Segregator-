"""
Train the waste-classification CNN.

Expects  dataset/<class>/*.jpg  with one sub-folder per class
(see scripts/capture_dataset.py for real data, or
 scripts/make_synthetic_dataset.py for a quick synthetic test set).

Usage
-----
    python src/train.py                    # train on ./dataset
    python src/train.py --epochs 40 --batch-size 16
    python src/train.py --tflite           # also export a .tflite model (for Raspberry Pi)

Outputs
-------
    models/waste_segregator.keras           best model (early stopping)
    models/waste_segregator_metadata.json   class names, input size, metrics
    models/waste_segregator.tflite          (only with --tflite)
    docs/plots/training_curves.png          accuracy / loss curves
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieten TensorFlow logs

# Make the project root importable no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (BATCH_SIZE, DATASET_DIR, EPOCHS, IMG_SIZE, LEARNING_RATE,  # noqa: E402
                    MODEL_DIR, MODEL_FILENAME, PLOT_DIR, SEED, VALIDATION_SPLIT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # save figures to files, never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the waste classifier.")
    p.add_argument("--dataset", type=Path, default=DATASET_DIR,
                   help="dataset folder with one sub-folder per class")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=IMG_SIZE[0],
                   help="square input size of the network (default: %(default)s)")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--model-name", type=str, default=MODEL_FILENAME,
                   help="file name of the saved model inside models/")
    p.add_argument("--tflite", action="store_true",
                   help="also export a TensorFlow Lite model (for Raspberry Pi)")
    return p.parse_args()


def build_model(img_size: int, n_classes: int):
    """A compact CNN that trains fast on a CPU.

    The Rescaling + data-augmentation layers live INSIDE the model, so at
    inference time you can feed raw 0-255 pixels and the augmentation layers
    automatically switch themselves off.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    # --- data augmentation (active only while training) ---
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    # --- convolutional blocks ---
    # momentum=0.9 (instead of the default 0.99) so the BatchNorm running
    # statistics converge quickly - important on small datasets where the
    # number of training steps per epoch is small. Otherwise the model can
    # look perfect during training and fail badly during inference.
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    # --- classifier head ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="waste_segregator_cnn")


def plot_history(history, out_path: Path) -> None:
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, hist["accuracy"], "o-", label="train")
    ax1.plot(epochs, hist["val_accuracy"], "s-", label="validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, hist["loss"], "o-", label="train")
    ax2.plot(epochs, hist["val_loss"], "s-", label="validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("AI Waste Segregator - training curves")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    import tensorflow as tf
    from tensorflow import keras

    args = parse_args()

    # ---- checks -------------------------------------------------------
    if not args.dataset.exists():
        print(f"ERROR: dataset folder not found: {args.dataset}")
        print("Create one first, e.g.:  python scripts/capture_dataset.py --class dry")
        return 1
    class_names = sorted(d.name for d in args.dataset.iterdir() if d.is_dir())
    if len(class_names) < 2:
        print("ERROR: the dataset needs at least two class sub-folders.")
        return 1
    for cls in class_names:
        n = len(list((args.dataset / cls).glob("*")))
        if n < 10:
            print(f"WARNING: class '{cls}' has only {n} images - accuracy will suffer.")

    tf.random.set_seed(SEED)
    print(f"TensorFlow {tf.__version__} | classes: {class_names} | input {args.img_size}x{args.img_size}")

    # ---- data ---------------------------------------------------------
    # NOTE: label numbers follow sorted() order of the folder names, which
    # is exactly the order of `class_names` above.
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        args.dataset,
        validation_split=VALIDATION_SPLIT,
        subset="both",
        seed=SEED,                      # same seed -> same split every run
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    # ---- model --------------------------------------------------------
    model = build_model(args.img_size, len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- training -----------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)

    # ---- save ---------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / args.model_name
    model.save(model_path)

    best_val_acc = float(max(history.history["val_accuracy"]))
    metadata = {
        "class_names": class_names,
        "img_size": [args.img_size, args.img_size],
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs_trained": len(history.history["loss"]),
        "dataset": str(args.dataset),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = model_path.with_name(model_path.stem + "_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    curves = PLOT_DIR / "training_curves.png"
    plot_history(history, curves)

    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    print(f"Model saved to     {model_path}")
    print(f"Metadata saved to  {meta_path}")
    print(f"Curves saved to    {curves}")

    # ---- optional TFLite export (Raspberry Pi / phone) ---------------
    if args.tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = model_path.with_suffix(".tflite")
        tflite_path.write_bytes(tflite_model)
        print(f"TFLite model saved to {tflite_path} "
              f"({len(tflite_model) / 1e6:.1f} MB)")

    print("\nNext steps:")
    print("  python src/evaluate.py")
    print("  python src/predict_webcam.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
