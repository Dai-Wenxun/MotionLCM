from typing import Optional

import torch
import torch.nn as nn

from mld.models.operator.attention import SkipTransformerEncoder, TransformerEncoderLayer
from mld.models.operator.position_encoding import build_position_encoding


class MldTrajEncoder(nn.Module):

    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 hidden_dim: Optional[int] = None,
                 force_post_proj: bool = False,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,
                 activation: str = "gelu",
                 norm_post: bool = True,
                 activation_post: Optional[str] = None,
                 position_embedding: str = "learned") -> None:
        super(MldTrajEncoder, self).__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim
        add_post_proj = force_post_proj or (hidden_dim is not None and hidden_dim != latent_dim[-1])
        self.latent_proj = nn.Linear(self.latent_dim, latent_dim[-1]) if add_post_proj else nn.Identity()

        self.skel_embedding = nn.Linear(nfeats * 3, self.latent_dim)

        self.query_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
            norm_eps
        )
        encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm, activation_post)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size, self.latent_dim))

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        bs, nframes, nfeats = features.shape
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos_encoder(xseq)
        global_token = self.encoder(xseq, src_key_padding_mask=~aug_mask)[0][:dist.shape[0]]
        global_token = self.latent_proj(global_token)
        global_token = global_token.permute(1, 0, 2)
        return global_token
