# MediScan AI — Pneumonia Detection from Chest X-Rays

A deep learning system that classifies chest X-ray images as **NORMAL** or
**PNEUMONIA** using transfer learning on a pretrained EfficientNet-B0
backbone. Built to train and run entirely on a CPU — no GPU required.

The project includes a full pipeline: data loading, training, evaluation,
single-image prediction, Grad-CAM explainability, and a Streamlit web app.

> ⚠️ **Disclaimer**: This project is for educational and portfolio purposes
> only. It is not a certified medical device and must never be used for
> real clinical diagnosis or treatment decisions.

---

## Project Structure

```
medical-image-classifier/
├── app.py                      # Streamlit web application
├── train.py                    # Model training with early stopping
├── evaluate.py                 # Test-set evaluation & metrics
├── predict.py                  # Single-image inference
├── gradcam.py                  # Grad-CAM explainability
├── llm_explainer.py             # Optional: AI-generated plain-language Grad-CAM explanations (via Groq)
├── requirements.txt
├── models/
│   └── best_model.pth          # Saved best checkpoint (created by train.py)
├── outputs/
│   ├── training_loss.png       # Created by train.py
│   ├── training_accuracy.png   # Created by train.py
│   ├── confusion_matrix.png    # Created by evaluate.py
│   ├── metrics.json            # Created by evaluate.py
│   └── gradcam/                # Created by gradcam.py
└── utils/
    ├── __init__.py
    ├── dataset.py               # Dataset loading & directory resolution
    └── transforms.py            # Image preprocessing pipelines
```

The dataset itself is **not included** in this repository (it's the public
Kaggle "Chest X-Ray Images (Pneumonia)" dataset, ~2 GB). You must download
and extract it separately — see [Dataset Setup](#dataset-setup) below.

---

## 1. Installation

### Requirements
- Python 3.9–3.12
- No GPU needed (CPU-only PyTorch works fine, training is just slower)

### Steps

```bash
# 1. Clone or unzip the project, then move into it
cd medical-image-classifier

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Setup

1. Download the dataset from Kaggle: **"Chest X-Ray Images (Pneumonia)"**
   (search for `paultimothymooney/chest-xray-pneumonia` on Kaggle).
2. Extract the zip file. After extraction, you should have a `chest_xray/`
   folder with this structure:

   ```
   chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

3. Place the `chest_xray/` folder inside the project root, next to
   `train.py`. (If it ends up nested, e.g. `chest_xray/chest_xray/train`,
   that's fine — `utils/dataset.py` automatically detects and handles this
   common Kaggle extraction quirk, and also ignores any stray `__MACOSX`
   folders or hidden `.DS_Store` / `._*` files.)

If your dataset lives somewhere else, every script accepts a `--data-dir`
flag instead of moving files around — see usage examples below.

---

## 3. Training

Train the model with default settings (10 epochs, batch size 32, learning
rate 0.001, early stopping patience 3):

```bash
python train.py
```

### Custom options

```bash
python train.py --epochs 15 --batch-size 16 --lr 0.0005 --patience 5
python train.py --data-dir /path/to/chest_xray
```

### What happens during training
- Loads an EfficientNet-B0 pretrained on ImageNet.
- Freezes the entire convolutional backbone — only a new classifier head
  (`Dropout → Linear(1280, 2)`) is trained.
- Automatically uses a GPU if one is available (`torch.cuda.is_available()`),
  otherwise runs on CPU with no code changes needed.
- After every epoch, prints training/validation loss and accuracy.
- Saves the model to `models/best_model.pth` every time validation accuracy
  improves — so the file on disk is always the best checkpoint seen so far.
- Stops early if validation accuracy doesn't improve for 3 consecutive
  epochs (configurable via `--patience`).
- Saves two plots at the end: `outputs/training_loss.png` and
  `outputs/training_accuracy.png`.

### Expected runtime
On a typical laptop CPU (4–8 cores, no GPU), one epoch over the ~5,200
training images usually takes a few minutes, since only the classifier head
is being trained — the frozen backbone's forward pass is the main cost.
Total training time for 10 epochs is typically well under an hour.

---

## 4. Evaluation

Once you have a trained model, evaluate it on the held-out test set:

```bash
python evaluate.py
```

### Custom options

```bash
python evaluate.py --data-dir /path/to/chest_xray --model-path models/best_model.pth
```

### What it produces
- Prints **Accuracy**, **Precision**, **Recall**, and **F1 Score** (with
  PNEUMONIA treated as the positive class) to the console, along with a
  full scikit-learn classification report.
- Saves a confusion matrix plot to `outputs/confusion_matrix.png`.
- Saves a `outputs/metrics.json` summary, which the Streamlit app reads to
  display metrics instantly without re-running inference.

---

## 5. Predicting on a Single Image

Run inference on one X-ray image from the command line:

```bash
python predict.py --image path/to/xray.jpeg
```

This prints the predicted class (`NORMAL` or `PNEUMONIA`), an overall
confidence score, and the probability assigned to each class. The same
preprocessing pipeline used during validation/testing (resize to 224×224,
ImageNet normalization) is applied automatically.

You can also use it programmatically:

```python
from predict import predict_from_path

result = predict_from_path("path/to/xray.jpeg")
print(result.predicted_class, result.confidence)
```

---

## 6. Grad-CAM Explainability

Generate a visual explanation of why the model made its prediction:

```bash
python gradcam.py --image path/to/xray.jpeg
```

This saves two files to `outputs/gradcam/`:
- `<filename>_heatmap.png` — the raw Grad-CAM activation heatmap.
- `<filename>_overlay.png` — the heatmap blended on top of the original
  X-ray, which is generally the more clinically useful view.

Grad-CAM targets EfficientNet-B0's final convolutional block and explains
whichever class the model actually predicted for that image.

### Optional: AI-generated plain-language explanations

`app.py` includes an optional "Explain this heatmap in plain language"
button in the Grad-CAM section. This uses an LLM (hosted on
[Groq](https://groq.com)'s inference API) to turn objectively measured
heatmap statistics — which region the model focused on, and whether that
focus sits over the spine/midline rather than the lung fields — into a
plain-language explanation. The LLM is never shown the image itself; it
only receives pre-computed numbers, so it's describing facts rather than
guessing from pixels.

This feature is entirely optional and the app works fully without it:

1. Get a free API key at [console.groq.com](https://console.groq.com).
2. Set it as an environment variable before launching the app:

   ```bash
   # macOS/Linux
   export GROQ_API_KEY="your-key-here"

   # Windows PowerShell
   $env:GROQ_API_KEY="your-key-here"
   ```

3. Run `streamlit run app.py` as usual. If the key isn't set, the button
   simply doesn't appear — everything else in the app works as normal.

---

## 7. Running the Web App

Launch the full interactive Streamlit application:

```bash
streamlit run app.py
```

The sidebar lets you switch between two pages:

- **🔬 Detection** — the main workflow:
  - **Upload** — drag-and-drop or browse for a PNG/JPG/JPEG X-ray.
  - **Prediction** — the uploaded image alongside the predicted class and a
    confidence readout.
  - **Explainability** — the Grad-CAM heatmap and overlay for the uploaded
    image, plus an optional AI-generated plain-language explanation (see
    above).
  - **Model Metrics** — Accuracy, Precision, Recall, and F1 Score from your
    most recent `evaluate.py` run, plus the confusion matrix.
- **📚 Learn About Pneumonia** — static educational content: what pneumonia
  is, general prevention measures, when to seek medical care, and links to
  further reading. This page is independent of the model and does not react
  to any prediction.

The app will tell you directly if `models/best_model.pth` or
`outputs/metrics.json` don't exist yet, and which script to run to produce
them — so the steps above (train → evaluate → run app) are meant to be
followed in order.


---

## 8. Demo
[Medical_Image_Classifier](https://medical-image-classifier-menigrdhfned9fbm3zonbt.streamlit.app/)
---

## Tech Stack

| Component         | Library                          |
|--------------------|-----------------------------------|
| Model & training   | PyTorch, torchvision (EfficientNet-B0) |
| Image processing   | OpenCV, Pillow                    |
| Metrics            | scikit-learn                      |
| Plotting           | Matplotlib                        |
| Web app            | Streamlit                         |
| Data handling      | NumPy, Pandas                     |

---

## Recommended Workflow Summary

```bash
# One-time setup
pip install -r requirements.txt

# 1. Train
python train.py

# 2. Evaluate
python evaluate.py

# 3. (Optional) Try a single prediction or Grad-CAM from the CLI
python predict.py --image chest_xray/test/PNEUMONIA/some_image.jpeg
python gradcam.py --image chest_xray/test/PNEUMONIA/some_image.jpeg

# 4. Launch the app
streamlit run app.py
```
