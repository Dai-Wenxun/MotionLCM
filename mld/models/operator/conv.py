from typing import Optional

import torch
import torch.nn as nn


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in: int, n_state: int, dilation: int = 1, activation: str = 'silu', dropout: float = 0.2,
                 norm: Optional[str] = None, norm_groups: int = 32, norm_eps: float = 1e-6, time_dim: int = 512) -> None:
        super(ResConv1DBlock, self).__init__()

        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in, eps=norm_eps)
            self.norm2 = nn.LayerNorm(n_in, eps=norm_eps)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=norm_groups, num_channels=n_in, eps=norm_eps)
            self.norm2 = nn.GroupNorm(num_groups=norm_groups, num_channels=n_in, eps=norm_eps)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=norm_eps)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding=dilation, dilation=dilation)
        self.time_mlp = nn.Linear(time_dim, n_state)
        nn.init.zeros_(self.time_mlp.weight)
        nn.init.zeros_(self.time_mlp.bias)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation(x)

        x = self.conv1(x)

        if time_embed is not None:
            x = x + self.time_mlp(time_embed).unsqueeze(-1)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in: int, n_state: int, n_depth: int, reverse_dilation: bool = True,
                 dilation_growth_rate: int = 1, activation: str = 'relu', dropout: float = 0.2,
                 norm: Optional[str] = None, norm_groups: int = 32, norm_eps: float = 1e-6) -> None:
        super(Resnet1D, self).__init__()
        blocks = [ResConv1DBlock(n_in, n_state, dilation=dilation_growth_rate ** depth, activation=activation,
                                 dropout=dropout, norm=norm, norm_groups=norm_groups, norm_eps=norm_eps)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResEncoder(nn.Module):
    def __init__(self,
                 in_width: int = 263,
                 mid_width: int = 512,
                 out_width: int = 512,
                 down_t: int = 2,
                 stride_t: int = 2,
                 n_depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 norm: Optional[str] = None,
                 norm_groups: int = 32,
                 norm_eps: float = 1e-6,
                 double_z: bool = False) -> None:
        super(ResEncoder, self).__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(in_width, mid_width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(mid_width, mid_width, filter_t, stride_t, pad_t),
                Resnet1D(mid_width, mid_width, n_depth, reverse_dilation=True, dilation_growth_rate=dilation_growth_rate,
                         activation=activation, dropout=dropout, norm=norm, norm_groups=norm_groups, norm_eps=norm_eps))
            blocks.append(block)
        blocks.append(nn.Conv1d(mid_width, out_width * 2 if double_z else out_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.permute(0, 2, 1))  # B x C x T


class ResDecoder(nn.Module):
    def __init__(self,
                 in_width: int = 263,
                 mid_width: int = 512,
                 out_width: int = 512,
                 down_t: int = 2,
                 n_depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 norm: Optional[str] = None,
                 norm_groups: int = 32,
                 norm_eps: float = 1e-6) -> None:
        super(ResDecoder, self).__init__()
        blocks = [nn.Conv1d(out_width, mid_width, 3, 1, 1), nn.ReLU()]

        for i in range(down_t):
            block = nn.Sequential(
                Resnet1D(mid_width, mid_width, n_depth, reverse_dilation=True, dilation_growth_rate=dilation_growth_rate,
                         activation=activation, dropout=dropout, norm=norm, norm_groups=norm_groups, norm_eps=norm_eps),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(mid_width, mid_width, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(mid_width, mid_width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(mid_width, in_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).permute(0, 2, 1)  # B x T x C
