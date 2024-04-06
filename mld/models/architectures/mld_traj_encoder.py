from typing import Optional

import torch
import torch.nn as nn

from mld.models.operator.cross_attention import SkipTransformerEncoder, TransformerEncoderLayer
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask


class MldTrajEncoder(nn.Module):

    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned") -> None:

        super().__init__()
        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]

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
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size, self.latent_dim))

    def forward(self, features: torch.Tensor, lengths: Optional[list[int]] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if lengths is None and mask is None:
            lengths = [len(feature) for feature in features]
            mask = lengths_to_mask(lengths, features.device)

        bs, nframes, nfeats = features.shape

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        global_token = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        return global_token
