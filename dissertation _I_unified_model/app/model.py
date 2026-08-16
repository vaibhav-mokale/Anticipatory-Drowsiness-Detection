from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torchvision


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.attention(lstm_out)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * lstm_out, dim=1)


class CNN_LSTM_ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v3_large(weights=None)
        self.cnn = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        cnn_dim = 960

        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = TemporalAttention(128)

        self.vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=0
        )
        vit_dim = 192

        self.classifier = nn.Sequential(
            nn.Linear(256 + vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        cnn_input = x.view(b * t, c, h, w)
        cnn_features = self.cnn(cnn_input)
        cnn_features = self.pool(cnn_features).flatten(1)
        cnn_features = cnn_features.view(b, t, -1)

        lstm_out, _ = self.lstm(cnn_features)
        temporal_features = self.attention(lstm_out)

        last_frame = x[:, -1]
        vit_features = self.vit(last_frame)

        fusion = torch.cat([temporal_features, vit_features], dim=1)
        return self.classifier(fusion)
