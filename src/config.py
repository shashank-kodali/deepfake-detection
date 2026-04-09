import os
import torch

# ─── Paths ───────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
FRAMES_PATH  = os.path.join(BASE_DIR, 'frames')
MODELS_PATH  = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for path in [DATASET_PATH, FRAMES_PATH, MODELS_PATH,
             os.path.join(FRAMES_PATH, 'real'),
             os.path.join(FRAMES_PATH, 'fake')]:
    os.makedirs(path, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────
REAL_VIDEOS = os.path.join(DATASET_PATH, 'real')
FAKE_VIDEOS = os.path.join(DATASET_PATH, 'fake')
MAX_VIDEOS  = 500
NUM_FRAMES  = 10

# ─── Model ───────────────────────────────────────────────
IMG_SIZE    = 224
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT     = 0.3

# ─── Training ────────────────────────────────────────────
BATCH_SIZE    = 8
NUM_EPOCHS    = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
TRAIN_SPLIT   = 0.8

# ─── Device ──────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')