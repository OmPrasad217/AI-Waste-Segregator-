"""
Central configuration for the AI-Powered Smart Waste Segregator.

Every script in the project (data collection, synthetic data, training,
evaluation, webcam demo, servo bin) imports its settings from this single
file so everything stays consistent.  Change a value here and it applies
everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"        # training images: one sub-folder per class
MODEL_DIR = BASE_DIR / "models"           # saved models + metadata
CAPTURE_DIR = BASE_DIR / "captures"       # snapshots saved from the webcam demo
PLOT_DIR = BASE_DIR / "docs" / "plots"    # generated figures (curves, matrices)

# ---------------------------------------------------------------------------
# Classes
#
# The order is ALPHABETICAL on purpose: tf.keras.utils.image_dataset_from_directory
# assigns label numbers in sorted folder order, and sorted() gives:
#     "dry" < "e-waste" < "wet"
# So model output index 0 = dry, 1 = e-waste, 2 = wet.
# ---------------------------------------------------------------------------
CLASSES = ["dry", "e-waste", "wet"]

# Display information for each class.
# NOTE: "color" is in BGR order (Blue, Green, Red) because OpenCV stores
# images that way.  The colours follow the Indian waste-bin convention:
# dry -> blue bin, wet -> green bin, e-waste -> red / authorised collector.
CLASS_INFO = {
    "dry": {
        "display": "DRY WASTE",
        "examples": "paper, cardboard, plastic, metal, glass",
        "bin": "BLUE bin (recyclable)",
        "color": (246, 120, 52),      # blue in BGR
    },
    "wet": {
        "display": "WET WASTE",
        "examples": "vegetable peels, food scraps, leaves, flowers",
        "bin": "GREEN bin (compostable)",
        "color": (60, 160, 46),       # green in BGR
    },
    "e-waste": {
        "display": "E-WASTE",
        "examples": "old phones, chargers, batteries, circuit boards",
        "bin": "RED bin / authorised e-waste collector",
        "color": (60, 60, 230),       # red in BGR
    },
}

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE = (96, 96)            # input size (height, width) of the demo CNN
BATCH_SIZE = 32
EPOCHS = 30                    # early stopping usually ends training sooner
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2         # fraction of images kept aside for validation
SEED = 42                      # fixed seed -> reproducible train/val split

# Names of the files produced by src/train.py
MODEL_FILENAME = "waste_segregator.keras"            # your own trained model
DEMO_MODEL_FILENAME = "waste_segregator_demo.keras"  # small demo model shipped in the repo

# ---------------------------------------------------------------------------
# Webcam demo settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.50    # below this the demo says "not sure yet"
SMOOTHING_FRAMES = 10          # predictions are averaged over the last N frames
ROI_FRACTION = 0.65            # side of the centre square, as a fraction of frame height
