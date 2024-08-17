from typing import Optional, Union

import torch
import torch.nn as nn

from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator.attention import (SkipTransformerEncoder,
                                           TransformerDecoder,
                                           TransformerDecoderLayer,
                                           TransformerEncoder,
                                           TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding


class MldDenoiser(nn.Module):

    def __init__(self,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: float = 0,
                 text_encoded_dim: int = 768,
                 time_cond_proj_dim: int = None,
                 is_controlnet: bool = False) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim

        self.arch = arch
        self.time_cond_proj_dim = time_cond_proj_dim

        self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(text_encoded_dim, self.latent_dim, cond_proj_dim=time_cond_proj_dim)
        if text_encoded_dim != self.latent_dim:
            self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        if self.arch == "trans_enc":
            encoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            encoder_norm = None if is_controlnet else nn.LayerNorm(self.latent_dim)
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm,
                                                  return_intermediate=is_controlnet)

        elif self.arch == "trans_dec":
            assert not is_controlnet, f"controlnet not supported in architecture: 'trans_dec'"
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)

            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )

        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")

        self.is_controlnet = is_controlnet

        def zero_module(module):
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

        # 0. dimension matching
        # [batch_size, latent_dim[0], latent_dim[1]] -> [latent_dim[0], batch_size, latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        # 1. check if controlnet
        if self.is_controlnet:
            controlnet_cond = controlnet_cond.permute(1, 0, 2)
            sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        # 2. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb, timestep_cond).unsqueeze(0)

        # 3. condition + time embedding
        # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
        # text embedding projection
        if self.text_encoded_dim != self.latent_dim:
            # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
            text_emb_latent = self.emb_proj(text_emb)
        else:
            text_emb_latent = text_emb
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

        elif self.arch == "trans_dec":
            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

        else:
            raise TypeError(f"{self.arch} is not supported")

        # 5. [latent_dim[0], batch_size, latent_dim[1]] -> [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2)
        return sample
