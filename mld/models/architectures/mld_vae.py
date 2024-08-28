from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.distribution import Distribution

from mld.models.operator.attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer
)
from mld.models.operator.conv import ResEncoder, ResDecoder
from mld.models.operator.position_encoding import build_position_encoding


class MldVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "encoder_decoder",
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,
                 activation: str = "gelu",
                 position_embedding: str = "learned") -> None:
        super(MldVae, self).__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.arch = arch

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
        encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        elif self.arch == 'encoder_decoder':
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)

            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                norm_eps
            )
            decoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            raise ValueError(f"Not support architecture: {self.arch}!")

        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, nfeats)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Distribution]:
        z, dist = self.encode(features, mask)
        feats_rst = self.decode(z, mask)
        return feats_rst, z, dist

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Distribution]:
        bs, nframes, nfeats = features.shape
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        # [latent_dim[0], batch_size, latent_dim] -> [batch_size, latent_dim[0], latent_dim[1]]
        latent = latent.permute(1, 0, 2)
        return latent, dist

    def decode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [batch_size, latent_dim[0], latent_dim[1]] -> [latent_dim[0], batch_size, latent_dim]
        z = z.permute(1, 0, 2)
        bs, nframes = mask.shape
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size), dtype=torch.bool, device=z.device)
            aug_mask = torch.cat((z_mask, mask), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq, src_key_padding_mask=~aug_mask)[z.shape[0]:]

        elif self.arch == "encoder_decoder":
            queries = self.query_pos_decoder(queries)
            output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        output[~mask.T] = 0
        feats = output.permute(1, 0, 2)
        return feats


class MldVaeV2(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: int,
                 down_t: int = 2,
                 stride_t: int = 2,
                 n_depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 norm: Optional[str] = None,
                 norm_groups: int = 32,
                 norm_eps: float = 1e-6,
                 clip_range: Optional[tuple[int]] = None) -> None:
        super(MldVaeV2, self).__init__()
        self.down_t = down_t
        self.stride_t = stride_t
        self.clip_range = clip_range

        self.encoder = ResEncoder(nfeats, latent_dim, latent_dim, down_t, stride_t,
                                  n_depth, dilation_growth_rate, activation=activation,
                                  dropout=dropout, norm=norm, norm_groups=norm_groups,
                                  norm_eps=norm_eps, double_z=True)
        self.decoder = ResDecoder(nfeats, latent_dim, latent_dim, down_t, n_depth,
                                  dilation_growth_rate, activation=activation, dropout=dropout,
                                  norm=norm, norm_groups=norm_groups, norm_eps=norm_eps)

        self.quant_conv = nn.Conv1d(latent_dim * 2, latent_dim * 2, 1)
        self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)

        self.T = None

    def encode(self, features: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, Distribution]:
        self.T = features.shape[1]
        padding_factor = self.stride_t ** self.down_t
        padding = (padding_factor - self.T % padding_factor) % padding_factor
        features = F.pad(features, (0, 0, 0, padding))

        h = self.encoder(features)
        moments = self.quant_conv(h)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        if self.clip_range is not None:
            logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        z = self.post_quant_conv(z)
        feats = self.decoder(z)
        feats = feats[:, :self.T, :]
        return feats
