import torch
import torch.nn as nn


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


class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, d_model, patch_size=4, dropout=0.0):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return self.drop(x)


class TokenDropout(nn.Module):
    """Drops patch tokens during training while preserving the CLS token."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training or x.size(1) <= 1:
            return x

        keep_prob = 1.0 - self.drop_prob
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        mask = keep_prob + torch.rand(
            patch_tokens.size(0),
            patch_tokens.size(1),
            1,
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )
        mask = mask.floor()
        patch_tokens = patch_tokens.div(keep_prob) * mask
        return torch.cat([cls_token, patch_tokens], dim=1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_multiplier=4,
        attention_dropout=0.1,
        encoder_dropout=0.1,
        drop_path=0.0,
        layer_scale_init=1e-4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(encoder_dropout)
        self.drop_path1 = DropPath(drop_path)
        self.layer_scale1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(encoder_dropout),
        )
        self.drop_path2 = DropPath(drop_path)
        self.layer_scale2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.drop_path1(self.attn_dropout(attn_output) * self.layer_scale1.view(1, 1, -1))
        x = x + self.drop_path2(self.mlp(self.norm2(x)) * self.layer_scale2.view(1, 1, -1))
        return x


class TransformerAttention(nn.Module):
    """Patch-based Transformer classifier for inputs of shape (B, 1000, 10)."""

    def __init__(
        self,
        input_shape=(1000, 10),
        num_classes=8,
        d_model=128,
        num_heads=4,
        num_encoder_layers=3,
        ff_multiplier=4,
        patch_size=4,
        embed_dropout=0.1,
        attention_dropout=0.1,
        encoder_dropout=0.15,
        token_dropout=0.0,
        drop_path_rate=0.0,
        layer_scale_init=1e-4,
        head_dropout=0.25,
    ):
        super().__init__()
        seq_len, in_channels = input_shape
        if seq_len != 1000 or in_channels != 10:
            raise ValueError("Expected input_shape=(1000, 10) based on current feature design")
        if seq_len % patch_size != 0:
            raise ValueError("patch_size must evenly divide the sequence length")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be >= 1")

        num_tokens = seq_len // patch_size
        self.patch_embed = PatchEmbedding1D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            dropout=embed_dropout,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))
        self.token_dropout = TokenDropout(token_dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embed, std=0.02)

        drop_path_values = torch.linspace(0.0, drop_path_rate, steps=num_encoder_layers).tolist()
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    attention_dropout=attention_dropout,
                    encoder_dropout=encoder_dropout,
                    drop_path=drop_path_values[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.post_encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embed[:, : x.size(1), :]
        x = self.token_dropout(x)
        for block in self.encoder:
            x = block(x)
        x = self.post_encoder_norm(x)
        return self.classifier(x[:, 0, :])


def build_transformer_attention_model(
    input_shape=(1000, 10),
    num_classes=8,
    d_model=128,
    num_heads=4,
    num_encoder_layers=3,
    ff_multiplier=4,
    patch_size=4,
    embed_dropout=0.1,
    attention_dropout=0.1,
    encoder_dropout=0.15,
    token_dropout=0.0,
    drop_path_rate=0.0,
    layer_scale_init=1e-4,
    head_dropout=0.25,
):
    return TransformerAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        ff_multiplier=ff_multiplier,
        patch_size=patch_size,
        embed_dropout=embed_dropout,
        attention_dropout=attention_dropout,
        encoder_dropout=encoder_dropout,
        token_dropout=token_dropout,
        drop_path_rate=drop_path_rate,
        layer_scale_init=layer_scale_init,
        head_dropout=head_dropout,
    )