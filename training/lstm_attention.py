import torch
import torch.nn as nn


class SequenceProjection(nn.Module):
    def __init__(self, in_channels, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return self.drop(x)


class LSTMAttention(nn.Module):
    """Stacked LSTM classifier for inputs of shape (B, 1000, 10)."""

    def __init__(
        self,
        input_shape=(1000, 10),
        num_classes=8,
        hidden_size=128,
        num_lstm_layers=2,
        proj_dropout=0.1,
        lstm_dropout=0.15,
        feature_dropout=0.1,
        head_dropout=0.25,
    ):
        super().__init__()
        seq_len, in_channels = input_shape
        if seq_len != 1000 or in_channels != 10:
            raise ValueError("Expected input_shape=(1000, 10) based on current feature design")
        if num_lstm_layers < 1:
            raise ValueError("num_lstm_layers must be >= 1")
        if hidden_size < 32:
            raise ValueError("hidden_size must be >= 32")

        self.sequence_projection = SequenceProjection(in_channels=in_channels, d_model=hidden_size, dropout=proj_dropout)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )
        self.post_encoder_norm = nn.LayerNorm(hidden_size)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.sequence_projection(x)
        self.encoder.flatten_parameters()
        x, _ = self.encoder(x)
        x = self.feature_dropout(self.post_encoder_norm(x))
        return self.classifier(x[:, -1, :])


def build_lstm_attention_model(
    input_shape=(1000, 10),
    num_classes=8,
    hidden_size=128,
    num_lstm_layers=2,
    proj_dropout=0.1,
    lstm_dropout=0.15,
    feature_dropout=0.1,
    head_dropout=0.25,
):
    return LSTMAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        proj_dropout=proj_dropout,
        lstm_dropout=lstm_dropout,
        feature_dropout=feature_dropout,
        head_dropout=head_dropout,
    )