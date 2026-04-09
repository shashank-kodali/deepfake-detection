import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config


# ─── Frame Extraction ────────────────────────────────────
def extract_frames(video_path, output_dir, num_frames=config.NUM_FRAMES):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    saved = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (config.IMG_SIZE, config.IMG_SIZE))
            save_path = os.path.join(output_dir, f'frame_{i:03d}.jpg')
            Image.fromarray(frame_resized).save(save_path)
            saved += 1

    cap.release()
    return saved


def process_video_folder(video_folder, frames_output, label, max_videos=config.MAX_VIDEOS):
    """Process all videos in a folder and extract frames."""
    video_files = glob.glob(os.path.join(video_folder, '**', '*.mp4'), recursive=True)[:max_videos]
    print(f'Processing {len(video_files)} {label} videos...')

    for vf in tqdm(video_files):
        video_name = os.path.splitext(os.path.basename(vf))[0]
        out_dir = os.path.join(frames_output, label, video_name)
        os.makedirs(out_dir, exist_ok=True)

        # Skip if already extracted
        if len(os.listdir(out_dir)) == config.NUM_FRAMES:
            continue

        extract_frames(vf, out_dir)

    print(f'✅ Done extracting {label} frames!')


# ─── Dataset Class ───────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Video sequence dataset.
    Each sample is a sequence of frames from one video.
    Shape: (num_frames, C, H, W)
    Label: 0 = real, 1 = fake
    """
    def __init__(self, frames_root, num_frames=config.NUM_FRAMES, transform=None):
        self.samples = []
        self.num_frames = num_frames
        self.transform = transform

        for label_name, label_idx in [('real', 0), ('fake', 1)]:
            label_dir = os.path.join(frames_root, label_name)
            if not os.path.exists(label_dir):
                continue
            for video_dir in os.listdir(label_dir):
                full_path = os.path.join(label_dir, video_dir)
                if os.path.isdir(full_path):
                    frames = glob.glob(os.path.join(full_path, '*.jpg'))
                    if len(frames) > 0:
                        self.samples.append((full_path, label_idx))

        print(f'Dataset: {len(self.samples)} videos found.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_files = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))

        # Pad or trim to exactly num_frames
        if len(frame_files) < self.num_frames:
            frame_files += [frame_files[-1]] * (self.num_frames - len(frame_files))
        else:
            frame_files = frame_files[:self.num_frames]

        frames = []
        for ff in frame_files:
            img = Image.open(ff).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


# ─── Transforms ──────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ─── Dataloader Builder ──────────────────────────────────
def get_dataloaders(frames_root=config.FRAMES_PATH):
    full_dataset = DeepfakeDataset(frames_root, transform=None)

    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'✅ Train: {len(train_ds)} | Val: {len(val_ds)}')
    return train_loader, val_loader