import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay)
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.dataset import get_dataloaders
from src.model import get_model


def evaluate_model(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_results(history, labels, preds, probs):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # ─── Loss Curve ───────────────────────────────────
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    # ─── Accuracy Curve ───────────────────────────────
    axes[1].plot(history['val_acc'], color='green')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)

    # ─── AUC Curve ────────────────────────────────────
    axes[2].plot(history['val_auc'], color='orange')
    axes[2].set_title('Validation AUC Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)

    # ─── ROC Curve ────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[3].plot(fpr, tpr, color='blue', label=f'AUC = {auc:.4f}')
    axes[3].plot([0, 1], [0, 1], 'k--')
    axes[3].set_title('ROC Curve')
    axes[3].set_xlabel('False Positive Rate')
    axes[3].set_ylabel('True Positive Rate')
    axes[3].legend()

    plt.tight_layout()
    save_path = os.path.join(config.MODELS_PATH, 'results.png')
    plt.savefig(save_path)
    plt.show()
    print(f'✅ Results saved to {save_path}')

    # ─── Confusion Matrix ─────────────────────────────
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['REAL', 'FAKE'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(config.MODELS_PATH, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.show()
    print(f'✅ Confusion matrix saved to {cm_path}')


def main():
    print('=' * 50)
    print('   Deepfake Detection — Evaluation')
    print('=' * 50)

    # Load model
    model = get_model()
    model_path = os.path.join(config.MODELS_PATH, 'best_model.pth')

    if not os.path.exists(model_path):
        print('❌ No trained model found! Run train.py first.')
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    print('✅ Model loaded!')

    # Load data
    _, val_loader = get_dataloaders()

    # Evaluate
    labels, preds, probs = evaluate_model(model, val_loader)

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    print(f'\n📊 Final Results:')
    print(f'   Accuracy : {acc:.4f} ({acc*100:.1f}%)')
    print(f'   AUC Score: {auc:.4f}')


if __name__ == '__main__':
    main()