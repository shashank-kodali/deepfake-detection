import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config


class DeepfakeDetector(nn.Module):
    """
    Spatial-Temporal Deepfake Detector
    ─────────────────────────────────────
    EfficientNet-B0 → extracts spatial features per frame
    LSTM            → learns temporal patterns across frames
    FC Head         → binary classification (real vs fake)
    """
    def __init__(self):
        super().__init__()

        # ─── CNN: EfficientNet-B0 pretrained on ImageNet ───
        self.cnn = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0       # remove classification head
        )
        cnn_out_features = self.cnn.num_features  # 1280 for B0

        # ─── LSTM: temporal reasoning across frames ────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT
        )

        # ─── Classifier head ──────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.LSTM_HIDDEN),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.LSTM_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x shape: (batch, num_frames, C, H, W)
        batch, num_frames, C, H, W = x.shape

        # Flatten batch and frames → run all through CNN
        x = x.view(batch * num_frames, C, H, W)
        features = self.cnn(x)                    # (batch*frames, 1280)

        # Reshape for LSTM → (batch, frames, features)
        features = features.view(batch, num_frames, -1)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(features)         # (batch, frames, hidden)
        last_out = lstm_out[:, -1, :]             # last timestep

        # Classification
        logits = self.classifier(last_out)        # (batch, 2)
        return logits


def get_model():
    model = DeepfakeDetector().to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✅ Model ready!')
    print(f'   Parameters: {total_params:,}')
    print(f'   Device: {config.DEVICE}')
    return model