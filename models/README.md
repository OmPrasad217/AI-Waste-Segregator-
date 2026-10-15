# Models folder

| File | What it is |
|---|---|
| `waste_segregator_demo.keras` | A small **demo** model shipped with the repo. |
| `waste_segregator_demo_metadata.json` | Class order + input size for the demo model. |
| `waste_segregator.keras` | *Your* model — created by `python src/train.py` (not committed). |

## About the demo model

It was trained on **synthetic, procedurally generated images**
(`scripts/make_synthetic_dataset.py`) — NOT real photographs. It learned
colour/texture cues:

- bright paper-like tones -> **dry**
- moist green/brown organic textures -> **wet**
- dark-green circuit-board textures -> **e-waste**

It exists so that `python src/predict_webcam.py` works the moment the repo
is cloned, proving the whole pipeline runs. **It will NOT reliably
recognise real objects.**

## Train the real model

```bash
# 1. collect ~150 real photos per class with your webcam
python scripts/capture_dataset.py --class dry
python scripts/capture_dataset.py --class e-waste
python scripts/capture_dataset.py --class wet

# 2. train (this overwrites models/waste_segregator.keras)
python src/train.py

# 3. the demo app automatically prefers your model over the demo one
python src/predict_webcam.py
```

Or train in the browser with Google Teachable Machine — see
`docs/TEACHABLE_MACHINE.md`.
