# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed, build_norm_layer
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmseg.registry import MODELS
from ..utils import LayerNorm2d


class HieraAttention(BaseModule):
    """Attention module for Hiera.
    
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
        drop_rate (float): Dropout rate.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class HieraBlock(BaseModule):
    """Hierarchical Vision Transformer Block for SAM2.
    
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
        norm_cfg (dict): Config dict for normalization layer.
        has_proj (bool): Whether this block has a projection layer.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 norm_cfg: dict = dict(type='LN'),
                 has_proj: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.has_proj = has_proj
        
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = HieraAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate)
        
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dims),
            nn.Dropout(drop_rate)
        )
        
        # Optional projection layer for hierarchical feature maps
        if has_proj:
            self.proj = nn.Linear(embed_dims, embed_dims)
    
    def forward(self, x):
        # x shape: [B, N, C]
        shortcut = x
        x = self.norm1(x)
        
        # Self-attention
        x = self.attn(x)
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        # Apply projection if needed
        if self.has_proj:
            x = self.proj(x)
            
        return x


@MODELS.register_module()
class HieraSAM2(BaseModule):
    """Hierarchical Vision Transformer (Hiera) backbone for SAM2.
    
    Args:
        img_size (int | tuple): Input image size.
        patch_size (int | tuple): Patch size.
        in_channels (int): Number of input channels.
        embed_dims (int): Embedding dimension.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        out_indices (Sequence | int): Output from which stages.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_cfg (dict): Config dict for normalization layer.
        patch_norm (bool): Whether to add normalization after patch embedding.
        frozen_stages (int): Stages to be frozen.
        proj_layers (Sequence[int]): Indices of layers with projection.
        use_pos_embed (bool): Whether to use position embedding.
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 1024,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 num_layers: int = 24,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 out_indices: Sequence[int] = [-1],
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN'),
                 patch_norm: bool = True,
                 frozen_stages: int = -1,
                 proj_layers: Sequence[int] = [2, 5, 21],  # Based on SAM2 Hiera
                 use_pos_embed: bool = True,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.frozen_stages = frozen_stages
        self.use_pos_embed = use_pos_embed
        
        # Convert output indices to list
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = out_indices
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None)
        
        # Position embedding
        if use_pos_embed:
            num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            self.pos_embed_window = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.pos_embed_window, std=0.02)
        else:
            self.pos_embed = None
            self.pos_embed_window = None
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # Build transformer layers
        self.layers = ModuleList()
        for i in range(num_layers):
            has_proj = i in proj_layers
            self.layers.append(
                HieraBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    qkv_bias=True,
                    norm_cfg=norm_cfg,
                    has_proj=has_proj))
        
        # Final norm layer
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        
        # Freeze stages if needed
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            
            if self.use_pos_embed and self.pos_embed is not None:
                self.pos_embed.requires_grad = False
                if self.pos_embed_window is not None:
                    self.pos_embed_window.requires_grad = False
        
        for i in range(self.frozen_stages):
            if i < len(self.layers):
                self.layers[i].eval()
                for param in self.layers[i].parameters():
                    param.requires_grad = False
    
    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        
        if not (isinstance(self.init_cfg, dict) and 
                self.init_cfg.get('type') == 'Pretrained'):
            # Initialize patch embedding
            if self.patch_embed.norm is not None:
                self.patch_embed.norm.weight.data.fill_(1.0)
                self.patch_embed.norm.bias.data.zero_()
            
            # Initialize position embedding
            if self.use_pos_embed and self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.pos_embed_window is not None:
                    trunc_normal_(self.pos_embed_window, std=0.02)
    
    def forward(self, x):
        """Forward function."""
        B = x.shape[0]
        
        # Patch embedding
        x, hw_shape = self.patch_embed(x)
        
        # Reshape to [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding if available
        if self.use_pos_embed and self.pos_embed is not None:
            # Resize position embeddings to match input sequence length
            pos_embed = F.interpolate(
                self.pos_embed.reshape(1, int(self.pos_embed.shape[1]**0.5), 
                                      int(self.pos_embed.shape[1]**0.5), self.embed_dims).permute(0, 3, 1, 2),
                size=(hw_shape[0], hw_shape[1]),
                mode='bicubic',
                align_corners=False).permute(0, 2, 3, 1).reshape(1, -1, self.embed_dims)
            x = x + pos_embed
        
        # Forward through transformer layers
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i in self.out_indices:
                # Reshape to [B, C, H, W]
                out = x.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2)
                outs.append(out)
        
        # Apply final norm if the last layer is not in out_indices
        if self.num_layers - 1 not in self.out_indices:
            x = self.norm(x)
            out = x.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2)
            outs.append(out)
        
        return tuple(outs)