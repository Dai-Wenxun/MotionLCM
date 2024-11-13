from typing import Optional, Union

import torch
import torch.nn as nn

from mld.models.operator.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator.attention import (SkipTransformerEncoder,
                                           SkipTransformerDecoder,
                                           TransformerDecoder,
                                           TransformerDecoderLayer,
                                           TransformerEncoder,
                                           TransformerEncoderLayer)
from mld.models.operator.moe import MoeTransformerEncoderLayer, MoeTransformerDecoderLayer
from mld.models.operator.utils import get_clones, get_activation_fn, zero_module
from mld.models.operator.position_encoding import build_position_encoding


def load_balancing_loss_func(router_logits: tuple, num_experts: int = 4, topk: int = 2):
    router_logits = torch.cat(router_logits, dim=0)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, topk, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = num_experts * torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss


class MldDenoiser(nn.Module):

    def __init__(self,
                 latent_dim: list = [1, 256],
                 hidden_dim: Optional[int] = None,
                 text_dim: int = 768,
                 time_dim: int = 768,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,
                 activation: str = "gelu",
                 norm_post: bool = True,
                 activation_post: Optional[str] = None,
                 flip_sin_to_cos: bool = True,
                 freq_shift: float = 0,
                 time_act_fn: str = 'silu',
                 time_post_act_fn: Optional[str] = None,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 add_mem_pos: bool = True,
                 force_pre_post_proj: bool = False,
                 text_act_fn: str = 'relu',
                 time_cond_proj_dim: Optional[int] = None,
                 zero_init_cond: bool = True,
                 is_controlnet: bool = False,
                 controlnet_embed_dim: Optional[int] = None,
                 controlnet_act_fn: str = 'silu',
                 moe: bool = False,
                 moe_num_experts: int = 4,
                 moe_topk: int = 2,
                 moe_loss_weight: float = 1e-2,
                 moe_jitter_noise: Optional[float] = None
                 ) -> None:
        super(MldDenoiser, self).__init__()

        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim
        add_pre_post_proj = force_pre_post_proj or (hidden_dim is not None and hidden_dim != latent_dim[-1])
        self.latent_pre = nn.Linear(latent_dim[-1], self.latent_dim) if add_pre_post_proj else nn.Identity()
        self.latent_post = nn.Linear(self.latent_dim, latent_dim[-1]) if add_pre_post_proj else nn.Identity()

        self.arch = arch
        self.time_cond_proj_dim = time_cond_proj_dim

        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_loss_weight = moe_loss_weight

        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_dim, self.latent_dim, time_act_fn, post_act_fn=time_post_act_fn,
                                                cond_proj_dim=time_cond_proj_dim, zero_init_cond=zero_init_cond)
        self.emb_proj = nn.Sequential(get_activation_fn(text_act_fn), nn.Linear(text_dim, self.latent_dim))

        self.query_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)
        if self.arch == "trans_enc":
            if moe:
                encoder_layer = MoeTransformerEncoderLayer(
                    self.latent_dim, num_heads, moe_num_experts, moe_topk, ff_size,
                    dropout, activation, normalize_before, norm_eps, moe_jitter_noise)
            else:
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim, num_heads, ff_size, dropout,
                    activation, normalize_before, norm_eps)

            encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post and not is_controlnet else None
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm, activation_post,
                                                  is_controlnet=is_controlnet, is_moe=moe)

        elif self.arch == 'trans_dec':
            if add_mem_pos:
                self.mem_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)
            else:
                self.mem_pos = None
            if moe:
                decoder_layer = MoeTransformerDecoderLayer(
                    self.latent_dim, num_heads, moe_num_experts, moe_topk, ff_size,
                    dropout, activation, normalize_before, norm_eps, moe_jitter_noise)
            else:
                decoder_layer = TransformerDecoderLayer(
                    self.latent_dim, num_heads, ff_size, dropout,
                    activation, normalize_before, norm_eps)

            decoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post and not is_controlnet else None
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm, activation_post,
                                                  is_controlnet=is_controlnet, is_moe=moe)
        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")

        self.is_controlnet = is_controlnet
        if self.is_controlnet:
            embed_dim = controlnet_embed_dim if controlnet_embed_dim is not None else self.latent_dim
            modules = [
                nn.Linear(latent_dim[-1], embed_dim),
                get_activation_fn(controlnet_act_fn) if controlnet_act_fn else None,
                nn.Linear(embed_dim, embed_dim),
                get_activation_fn(controlnet_act_fn) if controlnet_act_fn else None,
                zero_module(nn.Linear(embed_dim, latent_dim[-1]))
            ]
            self.controlnet_cond_embedding = nn.Sequential(*[m for m in modules if m is not None])

            self.controlnet_down_mid_blocks = nn.ModuleList([
                zero_module(nn.Linear(self.latent_dim, self.latent_dim)) for _ in range(num_layers)])

    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                timestep_cond: Optional[torch.Tensor] = None,
                controlnet_cond: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None
                ) -> tuple:

        # 0. check if controlnet
        if self.is_controlnet:
            sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        # 1. dimension matching (pre)
        sample = sample.permute(1, 0, 2)
        sample = self.latent_pre(sample)

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
            tokens, intermediates, router_logits = self.encoder(xseq, controlnet_residuals=controlnet_residuals)
        elif self.arch == 'trans_dec':
            sample = self.query_pos(sample)
            if self.mem_pos:
                emb_latent = self.mem_pos(emb_latent)
            tokens, intermediates, router_logits = self.decoder(sample, emb_latent,
                                                                controlnet_residuals=controlnet_residuals)
        else:
            raise TypeError(f"{self.arch} is not supported")

        router_loss = None
        if router_logits is not None:
            router_loss = load_balancing_loss_func(router_logits, self.moe_num_experts, self.moe_topk)
            router_loss = self.moe_loss_weight * router_loss

        if self.is_controlnet:
            control_res_samples = []
            for res, block in zip(intermediates, self.controlnet_down_mid_blocks):
                r = block(res)
                control_res_samples.append(r)
            return control_res_samples, router_loss
        elif self.arch == "trans_enc":
            sample = tokens[:sample.shape[0]]
        elif self.arch == 'trans_dec':
            sample = tokens
        else:
            raise TypeError(f"{self.arch} is not supported")

        # 5. dimension matching (post)
        sample = self.latent_post(sample)
        sample = sample.permute(1, 0, 2)
        return sample, router_loss
