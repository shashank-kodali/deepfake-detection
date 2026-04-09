import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.dataset import get_dataloaders, process_video_folder
from src.model import get_model


def train_epoch(model, loader, optimizer, criterion, epoch):
    # Unfreeze CNN after epoch 5
    if epoch == 5:
        for param in model.cnn.parameters():
            param.requires_grad = True
        print('🔓 CNN unfrozen — full model training!')

    model.train()
    total_loss = 0

    for frames, labels in tqdm(loader, desc=f'Training Epoch {epoch+1}'):
        frames, labels = frames.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc='Evaluating'):
            frames, labels = frames.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), acc, auc


def main():
    print('=' * 50)
    print('   Deepfake Detection — Training')
    print('=' * 50)

    # ─── Step 1: Extract frames if not done ───────────
    real_frames = os.path.join(config.FRAMES_PATH, 'real')
    fake_frames = os.path.join(config.FRAMES_PATH, 'fake')

    if len(os.listdir(real_frames)) == 0:
        print('\n📽️  Extracting frames from videos...')
        process_video_folder(config.REAL_VIDEOS, config.FRAMES_PATH, 'real')
        process_video_folder(config.FAKE_VIDEOS, config.FRAMES_PATH, 'fake')
    else:
        print(f'✅ Frames already extracted — skipping.')

    # ─── Step 2: Build dataloaders ────────────────────
    print('\n📦 Building dataloaders...')
    train_loader, val_loader = get_dataloaders()

    # ─── Step 3: Build model ──────────────────────────
    print('\n🧠 Building model...')
    model = get_model()

    # Freeze CNN initially
    for param in model.cnn.parameters():
        param.requires_grad = False
    print('✅ CNN frozen — training LSTM + classifier first')

    # ─── Step 4: Loss, optimizer, scheduler ──────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_auc = 0

    # ─── Step 5: Training loop ────────────────────────
    print('\n🚀 Starting training...\n')
    for epoch in range(config.NUM_EPOCHS):
        print(f'\n🔄 Epoch {epoch+1}/{config.NUM_EPOCHS}')

        # Rebuild optimizer after unfreezing
        if epoch == 5:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.LEARNING_RATE * 0.1,
                weight_decay=config.WEIGHT_DECAY
            )

        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Acc: {val_acc:.4f} | AUC: {val_auc:.4f}')

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(),
                       os.path.join(config.MODELS_PATH, 'best_model.pth'))
            print(f'💾 Best model saved! AUC: {best_auc:.4f}')

    print(f'\n✅ Training complete! Best AUC: {best_auc:.4f}')
    return history


if __name__ == '__main__':
    main()