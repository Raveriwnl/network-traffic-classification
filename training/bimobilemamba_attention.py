import torch
import torch.nn as nn

from cnn_bimobilemamba_attention import AttentionPooling1D, MobileMambaBlock


class BiMobileMambaAttention(nn.Module):
    """Bidirectional MobileMamba + attention classifier without a CNN stem."""

    def __init__(
        self,
        input_shape=(1000, 10),
        num_classes=8,
        d_model=64,
        num_mamba_layers=2,
        mamba_d_state=16,
        mamba_d_conv=3,
        mamba_expand=2,
        mamba_dropout=0.15,
        input_dropout=0.1,
        head_dropout=0.25,
        drop_path_rate=0.1,
        pool_dropout=0.1,
    ):
        super().__init__()
        seq_len, in_channels = input_shape
        if seq_len != 1000 or in_channels != 10:
            raise ValueError("Expected input_shape=(1000, 10) based on current feature design")
        if int(d_model) <= 0:
            raise ValueError("d_model must be a positive integer")
        if num_mamba_layers not in (1, 2, 3):
            raise ValueError("num_mamba_layers must be 1, 2, or 3")
        if int(mamba_d_state) <= 0:
            raise ValueError("mamba_d_state must be a positive integer")
        if int(mamba_d_conv) <= 0 or int(mamba_d_conv) % 2 == 0:
            raise ValueError("mamba_d_conv must be a positive odd integer")
        if float(mamba_expand) <= 0:
            raise ValueError("mamba_expand must be positive")

        d_model = int(d_model)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, d_model),
            nn.GELU(),
        )
        self.feature_dropout = nn.Dropout(input_dropout)

        drop_path_values = torch.linspace(0.0, drop_path_rate, steps=num_mamba_layers).tolist()
        self.mamba_blocks = nn.ModuleList(
            [
                MobileMambaBlock(
                    d_model=d_model,
                    d_state=int(mamba_d_state),
                    d_conv=int(mamba_d_conv),
                    expand=float(mamba_expand),
                    dropout=mamba_dropout,
                    drop_path=drop_path_values[i],
                )
                for i in range(num_mamba_layers)
            ]
        )

        self.post_mamba_norm = nn.LayerNorm(d_model)
        self.attention_pool = AttentionPooling1D(d_model=d_model, hidden_dim=d_model, dropout=pool_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(96, d_model)),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(max(96, d_model), num_classes),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.feature_dropout(x)

        for block in self.mamba_blocks:
            x = block(x)

        x = self.post_mamba_norm(x)
        x = self.attention_pool(x) + x.mean(dim=1)
        return self.classifier(x)


def build_bimobilemamba_attention_model(
    input_shape=(1000, 10),
    num_classes=8,
    d_model=64,
    num_mamba_layers=2,
    mamba_d_state=16,
    mamba_d_conv=3,
    mamba_expand=2,
    mamba_dropout=0.15,
    input_dropout=0.1,
    head_dropout=0.25,
    drop_path_rate=0.1,
    pool_dropout=0.1,
):
    return BiMobileMambaAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        d_model=d_model,
        num_mamba_layers=num_mamba_layers,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_dropout=mamba_dropout,
        input_dropout=input_dropout,
        head_dropout=head_dropout,
        drop_path_rate=drop_path_rate,
        pool_dropout=pool_dropout,
    )