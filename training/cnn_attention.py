import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling1D(nn.Module):
    """Learned attention pooling that keeps informative time steps."""

    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        scores = self.score(self.norm(x)).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class CNNFeatureExtractor(nn.Module):
    """Two-stage CNN extractor with residual shortcuts."""

    def __init__(self, in_channels=10, stem_dropout=0.1):
        super().__init__()
        self.block1 = ConvBNAct(in_channels, 32, kernel_size=3, dropout=stem_dropout)
        self.res1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.block2 = ConvBNAct(32, 64, kernel_size=3, dropout=stem_dropout)
        self.res2 = nn.Conv1d(32, 64, kernel_size=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.block1(x) + self.res1(x)
        x = self.pool1(x)

        x = self.block2(x) + self.res2(x)
        x = self.pool2(x)
        return x


class CNNAttention(nn.Module):
    """CNN + attention pooling classifier for inputs of shape (B, 1000, 10)."""

    def __init__(
        self,
        input_shape=(1000, 10),
        num_classes=8,
        stem_dropout=0.1,
        feature_dropout=0.1,
        pool_dropout=0.1,
        head_dropout=0.25,
    ):
        super().__init__()
        seq_len, in_channels = input_shape
        if seq_len != 1000 or in_channels != 10:
            raise ValueError("Expected input_shape=(1000, 10) based on current feature design")

        self.feature_extractor = CNNFeatureExtractor(in_channels=in_channels, stem_dropout=stem_dropout)
        self.feature_dropout = nn.Dropout1d(feature_dropout)
        self.attention_pool = AttentionPooling1D(d_model=64, hidden_dim=64, dropout=pool_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 96),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.feature_dropout(x)
        x = x.transpose(1, 2)
        x = self.attention_pool(x) + x.mean(dim=1)
        return self.classifier(x)


def build_cnn_attention_model(
    input_shape=(1000, 10),
    num_classes=8,
    stem_dropout=0.1,
    feature_dropout=0.1,
    pool_dropout=0.1,
    head_dropout=0.25,
):
    return CNNAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        stem_dropout=stem_dropout,
        feature_dropout=feature_dropout,
        pool_dropout=pool_dropout,
        head_dropout=head_dropout,
    )