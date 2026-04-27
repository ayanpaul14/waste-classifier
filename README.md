# WasteAI — Smart Waste Image Classifier

A production-ready deep learning web application that classifies waste images into 6 categories in real time and provides step-by-step recycling guidance for each waste type.

Built with **MobileNetV3** (transfer learning) + **Flask** + animated **HTML/CSS/JS** frontend.

---

## Live Demo

Coming Soon..

---

## Features

- **Real-time classification** — drag and drop any waste image and get instant results
- **6 waste categories** — cardboard, glass, metal, paper, plastic, trash
- **Confidence score** — color-coded high/medium/low confidence display
- **Low confidence warning** — automatic yellow warning when model is uncertain
- **Recycling process guide** — 5-step animated recycling instructions per waste type
- **Global recycling rate** — animated bar showing real-world recycling statistics
- **Disposal bin recommendation** — tells you exactly which bin to use
- **Fun facts** — educational recycling facts for each material
- **Animated background** — floating waste icons canvas animation
- **Frosted glass UI** — modern dark theme with backdrop blur cards

---

## Model Performance

| Class | Accuracy |
|---|---|
| Cardboard | 90.2% |
| Glass | 82.9% |
| Metal | 91.9% |
| Paper | 91.1% |
| Plastic | 83.6% |
| Trash | 77.3% |
| **Overall Test** | **87.24%** |
| **Best Val** | **90.45%** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | MobileNetV3-Small (PyTorch) |
| Training | Transfer learning + two-phase fine-tuning |
| Backend | Flask (Python) |
| Frontend | HTML / CSS / JavaScript (no framework) |
| Dataset | TrashNet / Kaggle Garbage Classification |
| Evaluation | scikit-learn, per-class accuracy, confusion matrix |

---

## Project Structure

```
waste-classifier/
├── app.py                  — Flask server + inference pipeline
├── train.py                — Model training (transfer learning)
├── evaluate.py             — Confusion matrix + Grad-CAM
├── prepare_dataset.py      — Train/val/test split utility
├── check2.py               — Per-class accuracy checker
├── requirements.txt        — Python dependencies
├── templates/
│   └── index.html          — Web UI (drag-and-drop classifier)
├── static/
│   └── uploads/            — Uploaded images
├── models/
│   ├── waste_classifier.pth        — Trained weights
│   ├── training_history.json       — Loss/accuracy history
│   └── training_curve.png          — Training plots
└── dataset/
    ├── train/   val/   test/       — Split dataset
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/waste-classifier.git
cd waste-classifier
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask werkzeug pillow matplotlib seaborn scikit-learn tqdm
```

### 4. Download dataset

```bash
pip install kaggle
kaggle datasets download -d asdasdasasdas/garbage-classification
```

Unzip and split:
```bash
# Windows
Expand-Archive -Path garbage-classification.zip -DestinationPath raw_dataset

# Mac/Linux
unzip garbage-classification.zip -d raw_dataset
```

```bash
python prepare_dataset.py --src "raw_dataset/Garbage classification/Garbage classification" --dst dataset
```

### 5. Train the model

```bash
python train.py
```

Expected output:
```
Training on: cpu
Train: 1766 | Val: 377 | Test: 384
Classes: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

── Phase 1: Train head only (5 epochs) ──
Epoch  1 | Train 0.623 | Val 0.701
...
── Phase 2: Fine-tune full model (15 epochs) ──
Epoch  6 | Train 0.812 | Val 0.834
...
Test Accuracy : 0.8724
Best Val Acc  : 0.9045
```

### 6. Run the web app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Model Architecture

```
Input Image (any size)
      ↓
Preprocessing → Resize 256 → CenterCrop 224 → Normalize
      ↓
MobileNetV3-Small backbone (pretrained on ImageNet)
      ↓
Global Average Pooling
      ↓
Hardswish → Linear(576→1024) → Hardswish → Dropout(0.2)
      ↓
Linear(1024→6)   ← Custom head for 6 waste classes
      ↓
Softmax → Predicted Class + Confidence Score
```

### Training Strategy — Two-Phase Transfer Learning

**Phase 1 (Epochs 1–5):** Backbone frozen, only classifier head trained. Prevents destroying pretrained ImageNet features.

**Phase 2 (Epochs 6–20):** Full model unfrozen, fine-tuned with 10× lower learning rate. Allows backbone to adapt to waste image domain.

---

## Engineering Decisions and Challenges

### Challenge 1 — Class imbalance
The trash class had only 95 images vs paper's 594. This caused lower trash accuracy (77%). Solution: weighted loss functions or SMOTE oversampling could improve this in future iterations.

### Challenge 2 — Organic class experiment
Attempted to add a 7th organic class using an external dataset. Discovered that the dataset labelled mixed recyclable and non-recyclable items as organic, causing cross-class contamination. The model began misclassifying cardboard and plastic as organic due to visual similarity in the training data. Diagnosed using per-class accuracy analysis and reverted to 6 reliable classes — maintaining 90% validation accuracy.

### Challenge 3 — Domain gap
Stock images with transparent/white backgrounds and no labels are harder to classify than real-world photos. The training dataset uses studio-style photos which differ from real-world dirty/occluded waste. Future work: add augmentation with varied backgrounds.

---

## Future Improvements

- Add object detection (YOLO) to handle multiple waste items per image
- Mobile app deployment (Flutter + TFLite export)
- Real-time webcam stream classification using OpenCV
- Docker containerisation for cloud deployment
- Active learning loop — human corrections improve the model over time
- Add organic class with a higher quality, properly labelled dataset

---

## Dataset

**TrashNet / Kaggle Garbage Classification**
- 2,527 images across 6 classes
- Split: 70% train / 15% val / 15% test
- Image size: ~512×384px, RGB, JPEG format
- Source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

---

## Requirements

```
torch==2.3.0
torchvision==0.18.0
flask==3.0.3
Pillow==10.3.0
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
tqdm==4.66.4
werkzeug==3.0.3
```

---

## Author

**Ayan Paul**
- GitHub: [@ayanpaul14](https://github.com/ayanpaul14)

---

## License

MIT License — feel free to use this project for learning and portfolio purposes.
