# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

import sys; sys.path.append("extensions/vggt")
from extensions.vggt.vggt.layers.attention import Attention
from extensions.vggt.vggt.layers.drop_path import DropPath
from extensions.vggt.vggt.layers.layer_scale import LayerScale
from extensions.vggt.vggt.layers.mlp import Mlp


# Modifications from VGGT:
#     1. Support AdaLN-Zero for time conditioning via `is_dit_block`
#     2. Support cross-attention


XFORMERS_AVAILABLE = False


class MyBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        cross_attn_class: Callable[..., nn.Module] = None,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        kv_dim: int =None,
        is_dit_block: bool = False,
    ) -> None:
        super().__init__()

        if kv_dim is None:
            kv_dim = dim

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if cross_attn_class is not None:
            self.norm_cross_attn = norm_layer(dim)
            self.cross_attn = cross_attn_class(
                dim,
                kv_dim=kv_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                qk_norm=qk_norm,
                fused_attn=fused_attn,
                rope=rope,
            )
            self.ls_cross_attn = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path_cross_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        else:
            self.norm_cross_attn = None
            self.cross_attn = None
            self.ls_cross_attn = None
            self.drop_path_cross_attn = None

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        self.is_dit_block = is_dit_block
        if is_dit_block:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )
            torch.nn.init.zeros_(self.adaLN_modulation[-1].weight)
            torch.nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, y: Tensor = None, pos=None, temb=None) -> Tensor:
        if self.is_dit_block:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=1)

            def attn_residual_func(x: Tensor, pos=None) -> Tensor:
                return (1. + gate_msa.unsqueeze(1)) * self.ls1(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos=pos))

            def ffn_residual_func(x: Tensor) -> Tensor:
                return (1. + gate_mlp.unsqueeze(1)) * self.ls2(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))

        else:
            def attn_residual_func(x: Tensor, pos=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), pos=pos))

            def ffn_residual_func(x: Tensor) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

        if self.cross_attn is not None:
            def cross_attn_residual_func(x: Tensor, y: Tensor, pos=None) -> Tensor:
                return self.ls_cross_attn(self.cross_attn(self.norm_cross_attn(x), y), pos=pos)

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            if self.cross_attn is not None:
                x = x + self.drop_path_cross_attn(cross_attn_residual_func(x, y, pos=pos))
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            if self.cross_attn is not None:
                x = x + self.drop_path_cross_attn(cross_attn_residual_func(x, y, pos=pos))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            if self.cross_attn is not None:
                x = x + cross_attn_residual_func(x, y, pos=pos)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


#################################################################################
#                                      DiT                                      #
#################################################################################

import math


def modulate(x: Tensor, shift: Tensor, scale: Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
