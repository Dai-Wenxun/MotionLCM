from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_clones, get_activation_fn


class SparseMoeMLP(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str) -> None:
        super(SparseMoeMLP, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(hidden_states))))


class SparseMoeBlock(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str,
                 num_experts: int, topk: int, jitter_noise: Optional[float] = None) -> None:
        super(SparseMoeBlock, self).__init__()
        self.topk = topk
        self.num_experts = num_experts
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(d_model, num_experts)
        self.experts = get_clones(SparseMoeMLP(d_model, dim_feedforward, dropout, activation), num_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        sequence_length, batch_size, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise is not None:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(sequence_length, batch_size, hidden_dim)
        return final_hidden_states, router_logits


class MoeTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, num_experts: int, topk: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", normalize_before: bool = False,
                 norm_eps: float = 1e-5, jitter_noise: Optional[float] = None) -> None:
        super(MoeTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.moe = SparseMoeBlock(
            d_model, dim_feedforward, dropout, activation,
            num_experts=num_experts, topk=topk, jitter_noise=jitter_noise
        )
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward_post(self,
                     src: torch.Tensor,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2, logits = self.moe(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, logits

    def forward_pre(self,
                    src: torch.Tensor,
                    src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2, logits = self.moe(src2)
        src = src + self.dropout2(src2)
        return src, logits

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask)
        return self.forward_post(src, src_mask, src_key_padding_mask)


class MoeTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, num_experts: int, topk: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", normalize_before: bool = False,
                 norm_eps: float = 1e-5, jitter_noise: Optional[float] = None) -> None:
        super(MoeTransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.moe = SparseMoeBlock(
            d_model, dim_feedforward, dropout, activation,
            num_experts=num_experts, topk=topk, jitter_noise=jitter_noise
        )

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward_post(self,
                     tgt: torch.Tensor,
                     memory: torch.Tensor,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2, logits = self.moe(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, logits

    def forward_pre(self,
                    tgt: torch.Tensor,
                    memory: torch.Tensor,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2, logits = self.moe(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        return tgt, logits

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> tuple:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
