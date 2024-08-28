import copy
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None, return_intermediate: bool = False) -> None:
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        x = src
        intermediate = []
        index = 0
        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            if controlnet_residuals is not None:
                x = x + controlnet_residuals[index]
                index += 1

            xs.append(x)

            if self.return_intermediate:
                intermediate.append(x)

        x = self.middle_block(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if controlnet_residuals is not None:
            x = x + controlnet_residuals[index]
            index += 1

        if self.return_intermediate:
            intermediate.append(x)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            if controlnet_residuals is not None:
                x = x + controlnet_residuals[index]
                index += 1

            if self.return_intermediate:
                intermediate.append(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        x = tgt
        intermediate = []
        index = 0
        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                       memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask)

            if controlnet_residuals is not None:
                x = x + controlnet_residuals[index]
                index += 1

            xs.append(x)

            if self.return_intermediate:
                intermediate.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        if controlnet_residuals is not None:
            x = x + controlnet_residuals[index]
            index += 1

        if self.return_intermediate:
            intermediate.append(x)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                       memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask)

            if controlnet_residuals is not None:
                x = x + controlnet_residuals[index]
                index += 1

            if self.return_intermediate:
                intermediate.append(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 return_intermediate: bool = False) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False, norm_eps: float = 1e-5) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self,
                     src: torch.Tensor,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src: torch.Tensor,
                    src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask)
        return self.forward_post(src, src_mask, src_key_padding_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False, norm_eps: float = 1e-5) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self,
                     tgt: torch.Tensor,
                     memory: torch.Tensor,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt: torch.Tensor,
                    memory: torch.Tensor,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)


def _get_clone(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
