import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth for residual branches."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class BidirectionalLSTMLayer(nn.Module):
    """Bidirectional LSTM sequence encoder with a lightweight fusion head."""

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for bidirectional LSTM")

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return self.fusion(outputs)


class LSTMBlock(nn.Module):
    def __init__(self, d_model=64, dropout=0.15, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.rnn = BidirectionalLSTMLayer(d_model=d_model, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)
        self.layer_scale1 = nn.Parameter(torch.ones(d_model) * 1e-3)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.drop_path2 = DropPath(drop_path)
        self.layer_scale2 = nn.Parameter(torch.ones(d_model) * 1e-3)

    def forward(self, x):
        x = x + self.drop_path1(self.drop1(self.rnn(self.norm1(x)) * self.layer_scale1.view(1, 1, -1)))
        x = x + self.drop_path2(self.drop2(self.ffn(self.norm2(x)) * self.layer_scale2.view(1, 1, -1)))
        return x


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


class CNNBiLSTMAttention(nn.Module):
    """CNN + bidirectional LSTM classifier for inputs of shape (B, 1000, 10)."""

    def __init__(
        self,
        input_shape=(1000, 10),
        num_classes=8,
        num_recurrent_layers=2,
        recurrent_dropout=0.15,
        stem_dropout=0.1,
        head_dropout=0.25,
        drop_path_rate=0.1,
        feature_dropout=0.1,
        pool_dropout=0.1,
    ):
        super().__init__()
        seq_len, in_channels = input_shape
        if seq_len != 1000 or in_channels != 10:
            raise ValueError("Expected input_shape=(1000, 10) based on current feature design")
        if num_recurrent_layers not in (1, 2, 3):
            raise ValueError("num_recurrent_layers must be 1, 2, or 3")

        self.feature_extractor = CNNFeatureExtractor(in_channels=10, stem_dropout=stem_dropout)

        drop_path_values = torch.linspace(0.0, drop_path_rate, steps=num_recurrent_layers).tolist()
        self.recurrent_blocks = nn.ModuleList(
            [
                LSTMBlock(
                    d_model=64,
                    dropout=recurrent_dropout,
                    drop_path=drop_path_values[i],
                )
                for i in range(num_recurrent_layers)
            ]
        )

        self.post_recurrent_norm = nn.LayerNorm(64)
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

        for block in self.recurrent_blocks:
            x = block(x)

        x = self.post_recurrent_norm(x)
        x = self.attention_pool(x) + x.mean(dim=1)
        return self.classifier(x)


def build_cnn_bilstm_attention_model(
    input_shape=(1000, 10),
    num_classes=8,
    num_recurrent_layers=2,
    recurrent_dropout=0.15,
    stem_dropout=0.1,
    head_dropout=0.25,
    drop_path_rate=0.1,
    feature_dropout=0.1,
    pool_dropout=0.1,
):
    return CNNBiLSTMAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        num_recurrent_layers=num_recurrent_layers,
        recurrent_dropout=recurrent_dropout,
        stem_dropout=stem_dropout,
        head_dropout=head_dropout,
        drop_path_rate=drop_path_rate,
        feature_dropout=feature_dropout,
        pool_dropout=pool_dropout,
    )