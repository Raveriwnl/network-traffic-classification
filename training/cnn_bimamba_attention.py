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


class MambaS6Layer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        if int(d_state) <= 0:
            raise ValueError("d_state must be a positive integer")
        if int(d_conv) <= 0 or int(d_conv) % 2 == 0:
            raise ValueError("d_conv must be a positive odd integer")
        if float(expand) <= 0:
            raise ValueError("expand must be positive")

        self.d_model = d_model
        self.d_state = int(d_state)
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=int(d_conv),
            padding=int(d_conv) // 2,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(a_init.unsqueeze(0).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def selective_scan(self, u, delta, a, b, c):
        batch_size, seq_len, _ = u.shape
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :]
            b_t = b[:, t, :]
            c_t = c[:, t, :]
            u_t = u[:, t, :]

            d_a_t = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            d_b_t = delta_t.unsqueeze(-1) * b_t.unsqueeze(1)
            h = d_a_t * h + d_b_t * u_t.unsqueeze(-1)
            y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        combined = self.in_proj(x)
        u, x_gate = torch.chunk(combined, 2, dim=-1)

        u = self.conv1d(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)

        x_dbl = self.x_proj(u)
        delta, b, c = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        a = -torch.exp(self.A_log)
        y = self.selective_scan(u, delta, a, b, c)
        y = y + u * self.D.view(1, 1, -1)
        y = y * F.silu(x_gate)
        return self.out_proj(y)


class BidirectionalMambaLayer(nn.Module):
    """Fuse forward and backward Mamba passes for richer sequence context."""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.forward_mamba = MambaS6Layer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = MambaS6Layer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        forward_states = self.forward_mamba(x)

        reversed_x = torch.flip(x, dims=[1])
        backward_states = self.backward_mamba(reversed_x)
        backward_states = torch.flip(backward_states, dims=[1])

        return self.fusion(torch.cat([forward_states, backward_states], dim=-1))


class MambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=3, expand=2, dropout=0.15, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = BidirectionalMambaLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
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
        x = x + self.drop_path1(self.drop1(self.ssm(self.norm1(x)) * self.layer_scale1.view(1, 1, -1)))
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


class CNNBiMambaAttention(nn.Module):
    """CNN + bidirectional Mamba classifier for inputs of shape (B, 1000, 10)."""

    def __init__(
        self,
        input_shape=(1000, 10),
        num_classes=8,
        num_mamba_layers=2,
        mamba_d_state=16,
        mamba_d_conv=3,
        mamba_expand=2,
        mamba_dropout=0.15,
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
        if num_mamba_layers not in (1, 2, 3):
            raise ValueError("num_mamba_layers must be 1, 2, or 3")
        if int(mamba_d_state) <= 0:
            raise ValueError("mamba_d_state must be a positive integer")
        if int(mamba_d_conv) <= 0 or int(mamba_d_conv) % 2 == 0:
            raise ValueError("mamba_d_conv must be a positive odd integer")
        if float(mamba_expand) <= 0:
            raise ValueError("mamba_expand must be positive")

        # CNN module: (B, 10, 1000) -> (B, 64, 250)
        self.feature_extractor = CNNFeatureExtractor(in_channels=10, stem_dropout=stem_dropout)

        # Mamba module: process (B, 250, 64)
        drop_path_values = torch.linspace(0.0, drop_path_rate, steps=num_mamba_layers).tolist()
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=64,
                    d_state=int(mamba_d_state),
                    d_conv=int(mamba_d_conv),
                    expand=float(mamba_expand),
                    dropout=mamba_dropout,
                    drop_path=drop_path_values[i],
                )
                for i in range(num_mamba_layers)
            ]
        )

        self.post_mamba_norm = nn.LayerNorm(64)
        self.feature_dropout = nn.Dropout1d(feature_dropout)
        self.attention_pool = AttentionPooling1D(d_model=64, hidden_dim=64, dropout=pool_dropout)

        # Classification head: attention pooling with residual global context.
        self.classifier = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 96),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        # Input transform: (B, 1000, 10) -> (B, 10, 1000)
        x = x.transpose(1, 2)

        # CNN module
        x = self.feature_extractor(x)
        x = self.feature_dropout(x)

        # (B, 64, 250) -> (B, 250, 64)
        x = x.transpose(1, 2)

        # Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)

        x = self.post_mamba_norm(x)

        # Attention pooling over sequence length with a residual mean summary.
        x = self.attention_pool(x) + x.mean(dim=1)
        return self.classifier(x)


def build_cnn_bimamba_attention_model(
    input_shape=(1000, 10),
    num_classes=8,
    num_mamba_layers=2,
    mamba_d_state=16,
    mamba_d_conv=3,
    mamba_expand=2,
    mamba_dropout=0.15,
    stem_dropout=0.1,
    head_dropout=0.25,
    drop_path_rate=0.1,
    feature_dropout=0.1,
    pool_dropout=0.1,
):
    return CNNBiMambaAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        num_mamba_layers=num_mamba_layers,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_dropout=mamba_dropout,
        stem_dropout=stem_dropout,
        head_dropout=head_dropout,
        drop_path_rate=drop_path_rate,
        feature_dropout=feature_dropout,
        pool_dropout=pool_dropout,
    )
