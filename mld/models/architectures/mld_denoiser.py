from typing import Optional, Union

import torch
import torch.nn as nn

from mld.models.operator.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator.attention import (SkipTransformerEncoder,
                                           TransformerDecoder,
                                           TransformerDecoderLayer,
                                           TransformerEncoder,
                                           TransformerEncoderLayer)
from mld.models.operator.conv import ResConv1DBlock
from mld.models.operator.utils import get_clones
from mld.models.operator.position_encoding import build_position_encoding


class MldDenoiser(nn.Module):

    def __init__(self,
                 latent_dim: list = [1, 256],
                 alpha: int = 1,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: float = 0,
                 text_dim: int = 768,
                 time_dim: int = 768,
                 time_cond_proj_dim: Optional[int] = None,
                 is_controlnet: bool = False) -> None:
        super(MldDenoiser, self).__init__()

        self.latent_dim = latent_dim[-1] * alpha
        self.latent_pre = nn.Linear(latent_dim[-1], self.latent_dim) if alpha != 1 else nn.Identity()
        self.latent_post = nn.Linear(self.latent_dim, latent_dim[-1]) if alpha != 1 else nn.Identity()

        self.text_dim = text_dim

        self.arch = arch
        self.time_cond_proj_dim = time_cond_proj_dim

        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_dim, self.latent_dim, cond_proj_dim=time_cond_proj_dim, act_fn='silu')
        self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(text_dim, self.latent_dim))

        self.query_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)

        if self.arch == "trans_enc":
            encoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                norm_eps
            )
            encoder_norm = None if is_controlnet else nn.LayerNorm(self.latent_dim, eps=norm_eps)
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm,
                                                  return_intermediate=is_controlnet)
        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")

        self.is_controlnet = is_controlnet

        def zero_module(module: nn.Module) -> nn.Module:
            for p in module.parameters():
                nn.init.zeros_(p)
            return module

        if self.is_controlnet:
            self.controlnet_cond_embedding = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                zero_module(nn.Linear(self.latent_dim, self.latent_dim))
            )

            self.controlnet_down_mid_blocks = nn.ModuleList([
                zero_module(nn.Linear(self.latent_dim, self.latent_dim)) for _ in range(num_layers)])

    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                timestep_cond: Optional[torch.Tensor] = None,
                controlnet_cond: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None
                ) -> Union[torch.Tensor, list[torch.Tensor]]:
        # 0. dimension matching (pre)
        sample = sample.permute(1, 0, 2)
        sample = self.latent_pre(sample)

        # 1. check if controlnet
        if self.is_controlnet:
            controlnet_cond = controlnet_cond.permute(1, 0, 2)
            sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        # 2. time_embedding
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb, timestep_cond).unsqueeze(0)

        # 3. condition + time embedding
        # text_emb [seq_len, batch_size, text_dim] <= [batch_size, seq_len, text_dim]
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        # text embedding projection
        text_emb_latent = self.emb_proj(encoder_hidden_states)
        emb_latent = torch.cat((time_emb, text_emb_latent), 0)

        # 4. transformer
        if self.arch == "trans_enc":
            xseq = torch.cat((sample, emb_latent), axis=0)
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq, controlnet_residuals=controlnet_residuals)

            if self.is_controlnet:
                control_res_samples = []
                for res, block in zip(tokens, self.controlnet_down_mid_blocks):
                    r = block(res)
                    control_res_samples.append(r)
                return control_res_samples

            sample = tokens[:sample.shape[0]]
        else:
            raise TypeError(f"{self.arch} is not supported")

        # 5. dimension matching (post)
        sample = self.latent_post(sample)
        sample = sample.permute(1, 0, 2)
        return sample


class MldDenoiserV2(nn.Module):

    def __init__(self,
                 latent_dim: list = [1, 256],
                 alpha: int = 1,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 position_embedding: str = "learned",
                 freq_shift: float = 0,
                 text_dim: int = 768,
                 text_fusion_layers: int = 4,
                 time_embedding_dim: Optional[int] = None,
                 time_cond_proj_dim: Optional[int] = None,
                 **kwargs) -> None:
        super(MldDenoiserV2, self).__init__()

        self.latent_dim = latent_dim[-1]
        self.latent_pre = nn.Linear(latent_dim[-1], self.latent_dim) if alpha != 1 else nn.Identity()
        self.latent_post = nn.Linear(self.latent_dim, latent_dim[-1]) if alpha != 1 else nn.Identity()

        self.time_cond_proj_dim = time_cond_proj_dim

        self.query_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)

        self.embed_text = nn.Linear(text_dim, self.latent_dim)
        text_encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
            norm_eps
        )
        self.textTransEncoder = get_clones(text_encoder_layer, text_fusion_layers)
        self.text_ln = nn.LayerNorm(self.latent_dim, eps=norm_eps)

        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
            norm_eps
        )

        time_embed_dim = time_embedding_dim or self.latent_dim * 4

        res_layer = ResConv1DBlock(
            self.latent_dim, self.latent_dim, activation=activation, dropout=dropout,
            norm='GN', time_dim=time_embed_dim
        )

        self.layers_dec = get_clones(decoder_layer, num_layers)
        self.layers_mlp = get_clones(res_layer, num_layers)

        self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            self.latent_dim, time_embed_dim, act_fn=activation, cond_proj_dim=time_cond_proj_dim)

    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor, **kwargs) -> Union[torch.Tensor, list[torch.Tensor]]:
        sample = sample.permute(1, 0, 2)
        sample = self.latent_pre(sample)

        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb)

        encoder_hidden_states = self.embed_text(encoder_hidden_states)
        for layer in self.textTransEncoder:
            encoder_hidden_states = layer(encoder_hidden_states)
        encoder_hidden_states = self.text_ln(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)

        sample = self.query_pos(sample)
        for decoder_layer, res_layer in zip(self.layers_dec, self.layers_mlp):
            sample = sample.permute(1, 2, 0)
            sample = res_layer(sample, time_emb)
            sample = sample.permute(2, 0, 1)
            # scale, shift = time_emb_mlp.chunk(2, dim=-1)
            # sample = sample * (1 + scale) + shift
            sample = decoder_layer(sample, encoder_hidden_states)

        sample = self.latent_post(sample)
        sample = sample.permute(1, 0, 2)
        return sample
