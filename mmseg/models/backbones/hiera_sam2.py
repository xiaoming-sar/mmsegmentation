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


class HieraBlock(BaseModule):
    """Hierarchical Vision Transformer Block for SAM2.
    
    This is a simplified implementation of the Hiera block used in SAM2.
    It may need to be adjusted based on the actual implementation details.
    
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
        norm_cfg (dict): Config dict for normalization layer.
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
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=drop_rate, bias=qkv_bias, batch_first=True)
        
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=dict(type='GELU'))
    
    def forward(self, x):
        # x shape: [B, N, C]
        shortcut = x
        x = self.norm1(x)
        
        # Self-attention
        x = self.attn(x, x, x)[0]
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register_module()
class HieraSAM2(BaseModule):
    """Hierarchical Vision Transformer (Hiera) backbone for SAM2.
    
    This is a simplified implementation of the Hiera backbone used in SAM2.
    It may need to be adjusted based on the actual implementation details.
    
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
        init_cfg (dict, optional): Initialization config dict.
    """
    
    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 out_indices: Sequence[int] = [-1],
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN'),
                 patch_norm: bool = True,
                 frozen_stages: int = -1,
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
        
        # Position embedding is handled in the SAM2 Hiera model
        self.pos_embed = None
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # Build transformer layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                HieraBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg))
        
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
    
    def forward(self, x):
        """Forward function."""
        B = x.shape[0]
        
        # Patch embedding
        x, hw_shape = self.patch_embed(x)
        
        # Reshape to [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        
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