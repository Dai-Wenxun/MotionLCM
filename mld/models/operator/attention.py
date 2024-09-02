from typing import Optional

import torch
import torch.nn as nn


from .utils import get_clone, get_clones, get_activation_fn


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 act: Optional[str] = None, return_intermediate: bool = False) -> None:
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)
        self.return_intermediate = return_intermediate
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(encoder_layer, num_block)
        self.middle_block = get_clone(encoder_layer)
        self.output_blocks = get_clones(encoder_layer, num_block)
        self.linear_blocks = get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

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

        if self.norm:
            x = self.act(self.norm(x))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 act: Optional[str] = None, return_intermediate: bool = False) -> None:
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)
        self.return_intermediate = return_intermediate
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(decoder_layer, num_block)
        self.middle_block = get_clone(decoder_layer)
        self.output_blocks = get_clones(decoder_layer, num_block)
        self.linear_blocks = get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

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

        if self.norm:
            x = self.act(self.norm(x))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 act: Optional[str] = None, return_intermediate: bool = False) -> None:
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm
        self.act = get_activation_fn(act)

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        output = src
        intermediate = []
        index = 0
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            if controlnet_residuals is not None:
                output = output + controlnet_residuals[index]
                index += 1

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm:
            output = self.act(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 act: Optional[str] = None, return_intermediate: bool = False) -> None:
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm
        self.act = get_activation_fn(act)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        output = tgt
        intermediate = []
        index = 0
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            if controlnet_residuals is not None:
                output = output + controlnet_residuals[index]
                index += 1

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm:
            output = self.act(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False, norm_eps: float = 1e-5) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.activation_name = activation
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward if activation != 'geglu' else dim_feedforward * 2)
        self.activation = get_activation_fn(activation) if activation != 'geglu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward_post(self,
                     src: torch.Tensor,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.activation_name == 'geglu':
            src2, gate = self.linear1(src).chunk(2, dim=-1)
            src2 = src2 * self.activation(gate)
        else:
            src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
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
        if self.activation_name == 'geglu':
            src2, gate = self.linear1(src2).chunk(2, dim=-1)
            src2 = src2 * self.activation(gate)
        else:
            src2 = self.activation(self.linear1(src2))
        src2 = self.linear2(self.dropout(src2))
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
        self.d_model = d_model
        self.activation_name = activation
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward if activation != 'geglu' else dim_feedforward * 2)
        self.activation = get_activation_fn(activation) if activation != 'geglu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

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
                     memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.activation_name == 'geglu':
            tgt2, gate = self.linear1(tgt).chunk(2, dim=-1)
            tgt2 = tgt2 * self.activation(gate)
        else:
            tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.linear2(self.dropout(tgt2))
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
        if self.activation_name == 'geglu':
            tgt2, gate = self.linear1(tgt2).chunk(2, dim=-1)
            tgt2 = tgt2 * self.activation(gate)
        else:
            tgt2 = self.activation(self.linear1(tgt2))
        tgt2 = self.linear2(self.dropout(tgt2))
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
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
