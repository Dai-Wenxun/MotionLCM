import numpy as np

import torch
import torch.nn as nn


class PositionEmbeddingSine1D(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500, batch_first: bool = False) -> None:
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return x


class PositionEmbeddingLearned1D(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500, batch_first: bool = False) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return x


def build_position_encoding(N_steps: int, position_embedding: str = "sine") -> nn.Module:
    if position_embedding == 'sine':
        position_embedding = PositionEmbeddingSine1D(N_steps)
    elif position_embedding == 'learned':
        position_embedding = PositionEmbeddingLearned1D(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding
