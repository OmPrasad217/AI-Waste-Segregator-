# Training with Google Teachable Machine (the no-code path)

If you prefer not to write training code at all, Google's **Teachable
Machine** (https://teachablemachine.withgoogle.com) trains an image
classifier in the browser for free. You can then plug the exported model
straight into this project's webcam app.

## Step 1 - Collect samples

1. Open https://teachablemachine.withgoogle.com and choose **Image Project → Standard image model**.
2. Rename the classes to exactly:
   - `dry`
   - `e-waste`
   - `wet`
3. For each class click **Webcam → Hold to record** and capture **100–200+
   samples** of different objects, angles, distances and backgrounds.
   * Add a few "nothing / table / hand" pictures to every class? No — instead
     you can add a 4th class later; but for this homework keep the 3 classes.

   Tips
   - same objects from different sides
   - different lighting (bright window, dim room)
   - some close-ups, some from ~50 cm
   - different backgrounds (table, floor, white sheet)

## Step 2 - Train

Click **Train Model**. Keep the tab open while training. Play with the
preview to check that each class is recognised reliably.

## Step 3 - Export as a Keras model

1. Click **Export Model**.
2. Choose **TensorFlow → Keras**.
3. Click **Download model** - you get a zip containing:
   - `model.h5` - the trained model
   - `labels.txt` - the class names, in output order
   - (ignore the tfjs-files folder)

## Step 4 - Use it with this project's webcam app

Unzip and copy the two files into the `models/` folder of this repo, then:

```bash
python src/predict_webcam.py --model models/model.h5 --labels models/labels.txt
```

The app reads the input size (Teachable Machine uses 224x224) from the
model itself, so nothing else to configure.

> If predictions look random, the exported model may expect pixel values
> scaled to 0–1 instead of 0–255. Add the `--normalize` flag:
>
> ```bash
> python src/predict_webcam.py --model models/model.h5 --labels models/labels.txt --normalize
> ```

## Which path should I choose for the exhibition?

| | Teachable Machine | `src/train.py` (this repo) |
|---|---|---|
| Coding needed | none | one command |
| Control over the model | little | full (architecture, augmentation, epochs) |
| Reproducible / documented | browser only | yes - everything is in the repo |
| Judges' impression | quick demo | "we trained our own CNN" + training curves |

Both produce a Keras model that works with `src/predict_webcam.py` and
`src/servo_bin.py`, so you can start with Teachable Machine and switch to
`train.py` later without changing anything else.
