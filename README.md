# 🎭 Deepfake Detection System
### Spatial-Temporal CNN + LSTM Pipeline

A deep learning system that detects deepfake videos using a two-stage architecture:
- **EfficientNet-B0** (CNN) — extracts spatial features from individual frames
- **LSTM** — analyzes temporal inconsistencies across frame sequences

---

## 📁 Project Structure

```
deepfake-detection/
├── notebooks/
│   └── deepfake_detection.ipynb   ← Main Colab notebook (all phases)
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

- Go to [colab.research.google.com](https://colab.research.google.com)
- File → Open Notebook → GitHub → paste this repo URL
- Open `notebooks/deepfake_detection.ipynb`
- Runtime → Change Runtime Type → **T4 GPU**

### 2. Run cells top to bottom

Each phase is clearly labeled. Follow the instructions in each cell.

---

## 🧠 Architecture

```
Video Input
    │
    ▼
Frame Extraction (10 frames per video)
    │
    ▼
EfficientNet-B0 (CNN)
→ Spatial features per frame (1280-dim vector)
    │
    ▼
LSTM (2 layers, 256 hidden units)
→ Temporal patterns across frames
    │
    ▼
Fully Connected Classifier
→ REAL / FAKE + confidence score
```

---

## 📊 Dataset

**FaceForensics++** from Kaggle  
- 500 real videos + 500 fake videos (subset)
- 10 frames extracted per video
- 80/20 train/validation split

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| CNN | EfficientNet-B0 (timm) |
| Temporal Model | PyTorch LSTM |
| Training | Google Colab (T4 GPU) |
| Demo UI | Gradio |
| Dataset | FaceForensics++ (Kaggle) |

---

## 👥 Team

- [Your Name]
- [Friend's Name]
