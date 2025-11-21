# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
# Keep LayerScale from timm as it's simple
from timm.layers import LayerScale, DropPath, to_2tuple, use_fused_attn
# from timm.layers import PatchEmbed as TimmPatchEmbed # Use custom or mmseg version

from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
# from ..utils import ConvPatchEmbed # Assuming you might have this, otherwise define below

# --- Helper Functions (from timm HieraDet) ---

def window_partition(x, window_size: Tuple[int, int]):
    """ Partition into non-overlapping windows. Assumes H, W are divisible by window size. """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_unpartition(windows: torch.Tensor, window_size: Tuple[int, int], hw: Tuple[int, int]):
    """ Unpartition windows back into sequences. Assumes H, W are divisible by window size. """
    H, W = hw
    C = windows.shape[-1]
    num_windows = H * W // (window_size[0] * window_size[1])
    B = windows.shape[0] // num_windows
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

def _calc_pad(H: int, W: int, window_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """ Calculates padding required for window partitioning. """
    pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - W % window_size[1]) % window_size[1]
    Hp, Wp = H + pad_h, W + pad_w
    return Hp, Wp, pad_h, pad_w

# --- Custom Patch Embedding (if not using mmseg.models.utils.PatchEmbed) ---
# Simplified version matching HieraPatchEmbed
class ConvPatchEmbed(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=7,
                 stride=4,
                 padding=3,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.proj = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=to_2tuple(kernel_size),
            stride=to_2tuple(stride),
            padding=to_2tuple(padding)
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            # Apply norm to BCHW format
            x = self.norm(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x, x.shape[1:3] # Return H, W as well

# --- Hiera Modules adapted for MMSegmentation ---

class MMSegMultiScaleAttention(BaseModule):
    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None, # Pass the instantiated pooling layer
        attn_drop_rate=0.,
        proj_drop_rate=0.,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        try:
            # Use timm's helper, requires timm to be installed
            self.fused_attn = use_fused_attn()
        except ImportError:
            self.fused_attn = False
            print_log("timm not found, disabling fused attention", logger='current')


        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.attn_drop = nn.Dropout(attn_drop_rate) # Added attn_drop
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop_rate) # Added proj_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        N = H * W # Number of tokens before pooling

        # qkv with shape (B, N, 3, nHead, C_head)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)

        # q, k, v with shape (B, N, nheads, C_head)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool is not None:
            q_pooled = q.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # to BCHW for pool
            q_pooled = self.q_pool(q_pooled).permute(0, 2, 3, 1) # BHWC_out
            H_out, W_out = q_pooled.shape[1:3]  # downsampled shape
            N_out = H_out * W_out
            q = q_pooled.reshape(B, N_out, self.num_heads, -1) # B, N_out, nheads, C_head
        else:
            H_out, W_out = H, W
            N_out = N

        # Torch's SDPA expects [B, nheads, N, C_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.fused_attn:
            # dropout_p is only applied during training
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-1, -2) # B, nheads, N_out, N
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # B, nheads, N_out, C_head

        # Transpose back: B, N_out, nheads, C_head -> B, H_out, W_out, C_out
        x = x.transpose(1, 2).reshape(B, H_out, W_out, self.dim_out)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MMSegMultiScaleBlock(BaseModule):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        q_stride: Optional[Tuple[int, int]] = None,
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        window_size: int = 0,
        init_values: Optional[float] = None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        with_cp=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.window_size = to_2tuple(window_size)
        self.is_windowed = any(w > 0 for w in self.window_size)
        self.dim = dim
        self.dim_out = dim_out
        self.q_stride = q_stride

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = nn.Identity()

        self.pool = None
        if self.q_stride:
            # Create the pooling layer here to be passed to attention
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride,
                stride=q_stride,
                ceil_mode=False, # Hiera uses floor mode (ceil_mode=False)
            )

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = MMSegMultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=deepcopy(self.pool), # Pass a copy to attention
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
        )
        # Use timm's LayerScale, simpler than finding mmcv equivalent
        self.ls1 = LayerScale(dim_out, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim_out)[1]
        ffn_hidden_dim = int(dim_out * mlp_ratio)
        self.ffn = FFN(
            embed_dims=dim_out,
            feedforward_channels=ffn_hidden_dim,
            num_fcs=2,
            ffn_drop=drop_rate, # FFN drop uses proj_drop_rate
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False, # We add identity outside FFN
            init_cfg=None
        )
        self.ls2 = LayerScale(dim_out, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        def _inner_forward(x):
            shortcut = x  # B, H, W, C
            x_norm = self.norm1(x)

            # Skip connection projection/pooling
            if self.dim != self.dim_out:
                shortcut = self.proj(x_norm) # Apply proj on normalized input like timm? Or non-normed? Let's stick to timm's non-normed.
                shortcut = self.proj(x)
                if self.pool is not None:
                    shortcut_pool = shortcut.permute(0, 3, 1, 2) # BCHW
                    shortcut_pool = self.pool(shortcut_pool)
                    shortcut = shortcut_pool.permute(0, 2, 3, 1) # BHWC
            elif self.pool is not None: # Need to pool shortcut even if dim doesn't change
                 shortcut_pool = shortcut.permute(0, 3, 1, 2) # BCHW
                 shortcut_pool = self.pool(shortcut_pool)
                 shortcut = shortcut_pool.permute(0, 2, 3, 1) # BHWC


            # Window partition
            H, W = x_norm.shape[1:3]
            window_size = self.window_size
            Hp, Wp, pad_h, pad_w = H, W, 0, 0 # Init for non-windowed case
            if self.is_windowed:
                Hp, Wp, pad_h, pad_w = _calc_pad(H, W, window_size)
                if pad_h > 0 or pad_w > 0:
                    x_norm = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h))
                x_windows = window_partition(x_norm, window_size)
                attn_in = x_windows
            else:
                attn_in = x_norm # Use the full feature map

            # Attention + Q Pooling (if stage change)
            attn_windows = self.attn(attn_in) # Output shape depends on pooling

            # Determine output H, W after potential pooling in attention
            if self.q_stride is not None:
                H_out, W_out = shortcut.shape[1:3]
                 # Window size needs adjustment if pooling happened inside windowed attention
                if self.is_windowed:
                    window_size = (self.window_size[0] // self.q_stride[0], self.window_size[1] // self.q_stride[1])
                    # Recalculate padding based on output size
                    Hp, Wp, pad_h, pad_w = _calc_pad(H_out, W_out, window_size)
            else:
                H_out, W_out = H, W
                # Use original padded size if no pooling
                Hp, Wp, pad_h, pad_w = Hp, Wp, pad_h, pad_w


            # Reverse window partition
            if self.is_windowed:
                x_attn = window_unpartition(attn_windows, window_size, (Hp, Wp))
                # Unpad if needed
                if pad_h > 0 or pad_w > 0:
                    x_attn = x_attn[:, :H_out, :W_out, :].contiguous()
            else:
                 x_attn = attn_windows # Output is already the full feature map


            # First residual connection
            x = shortcut + self.drop_path1(self.ls1(x_attn))

            # Second residual connection (MLP)
            x_norm2 = self.norm2(x)
            x_mlp = self.ffn(x_norm2)
            x = x + self.drop_path2(self.ls2(x_mlp))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class HieraSAM2(BaseModule):
    """ Hiera Vision Transformer with SAM2 Backbone for MMSegmentation

    Based on timm implementation: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hiera.py
    Reference: https://arxiv.org/abs/2306.00989
    """
    def __init__(
            self,
            in_channels: int = 3,
            embed_dims: int = 96,  # initial embed dim
            num_heads: int = 1,  # initial number of heads
            patch_kernel: Union[int, Tuple[int, int]] = 7,
            patch_stride: Union[int, Tuple[int, int]] = 4,
            patch_padding: Union[int, Tuple[int, int]] = 3,
            # patch_size: Optional[Union[int, Tuple[int, int]]] = None, # TODO: Add ViT-style patch embed option
            q_pool: int = 3,  # number of q_pool stages
            q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
            stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
            dim_mul: float = 2.0,  # dim_mul factor at stage shift
            head_mul: float = 2.0,  # head_mul factor at stage shift
            mlp_ratio: float = 4.0, # MLP expansion ratio
            global_pos_size: Tuple[int, int] = (7, 7), # Size for global part of pos embed
            # window size per stage, when not using global att.
            window_spec: Tuple[int, ...] = (8, 4, 14, 7),
            # global attn in these blocks (0-based index)
            global_att_blocks: Optional[Tuple[int, ...]] = (12, 16, 20),
            init_values: Optional[float] = None, # LayerScale init value
            drop_rate: float = 0.0, # General dropout for FFN, Attn proj
            attn_drop_rate: float = 0.0, # Attention map dropout
            drop_path_rate: float = 0.0,  # stochastic depth
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            with_cp: bool = False,
            pretrained: Optional[str] = None,
            init_cfg: Optional[Union[Dict, List[Dict]]] = None,
            out_indices: Sequence[int] = (0, 1, 2, 3),
            frozen_stages: int = -1,
            # fix_init: bool = True, # From timm, maybe apply later if needed
    ):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        self.num_stages = len(stages)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, self.num_stages + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        # Blocks where pooling happens (0-based index)
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]

        # TODO: Add ViT-style patch embed option if needed
        # Currently only Conv-based
        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            # norm_cfg=norm_cfg # Hiera doesn't norm patch embed
        )

        # Which blocks have global att? (convert to set for faster lookup)
        self.global_att_blocks = set(global_att_blocks) if global_att_blocks is not None else set()

        # Windowed positional embedding
        self.global_pos_size = global_pos_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dims, *self.global_pos_size))
        # Assume square window for pos embed window part
        # Use first stage window spec for this parameter
        win_size_pos = self.window_spec[0]
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dims, win_size_pos, win_size_pos))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        cur_stage_idx = 0
        cur_block_idx = 0
        current_dim = embed_dims
        current_heads = num_heads
        self.blocks = ModuleList()
        self.num_features = [] # Store dims for output norms

        for i in range(self.num_stages):
            stage_depth = stages[i]
            for j in range(stage_depth):
                is_global_att = cur_block_idx in self.global_att_blocks
                window_size = 0 if is_global_att else self.window_spec[i]

                dim_out = current_dim
                heads_out = current_heads
                block_q_stride = None

                # Check if this block is the start of a new stage (except the first block)
                # Pooling happens *at* the block index specified in q_pool_blocks
                if cur_block_idx in self.q_pool_blocks:
                    dim_out = int(current_dim * dim_mul)
                    heads_out = int(current_heads * head_mul)
                    block_q_stride = self.q_stride

                block = MMSegMultiScaleBlock(
                    dim=current_dim,
                    dim_out=dim_out,
                    num_heads=heads_out, # Use potentially updated head count
                    mlp_ratio=mlp_ratio,
                    q_stride=block_q_stride,
                    window_size=window_size,
                    init_values=init_values,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur_block_idx],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                )
                current_dim = dim_out
                current_heads = heads_out # Update heads for next block within stage if needed
                self.blocks.append(block)
                cur_block_idx += 1

            # Store feature dimension at the end of each stage
            self.num_features.append(current_dim)


        # Add normalization layers for each output index
        for i in self.out_indices:
            if i >= len(self.num_features):
                 raise ValueError(f"out_index {i} is out of range for {len(self.num_features)} stages.")
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self._freeze_stages()


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_embed_window.requires_grad = False

        # Freeze stages (blocks are indexed sequentially)
        # frozen_stages corresponds to the number of *stages* to freeze
        start_block_idx = 0
        for i in range(self.frozen_stages):
             end_block_idx = self.stage_ends[i] + 1 # stage_ends is 0-based index
             for blk_idx in range(start_block_idx, end_block_idx):
                 m = self.blocks[blk_idx]
                 m.eval()
                 for param in m.parameters():
                     param.requires_grad = False
             start_block_idx = end_block_idx

             # Freeze the output norm layer if the stage is frozen
             if i in self.out_indices:
                 norm_layer = getattr(self, f'norm{i}')
                 norm_layer.eval()
                 for param in norm_layer.parameters():
                     param.requires_grad = False


    def init_weights(self):
        """Initialize weights."""
        print_log(f'Initializing {self.__class__.__name__} weights')
        if self.init_cfg is None:
            print_log('No pretrained ckpt specified, init from scratch.')
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.pos_embed_window, std=0.02)
            self.apply(self._init_weights_fn)
            # Apply fix_init like timm? (Optional)
            # self.fix_init_weight()
        elif isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained':
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: # Common in timm checkpoints
                 state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # --- Checkpoint Key Adaptation ---
            # Handle potential prefixes (e.g., from SAM2)
            # Example prefix, adjust if needed based on actual checkpoint
            # sam2_prefix = 'image_encoder.trunk.'
            # if any(k.startswith(sam2_prefix) for k in state_dict.keys()):
            #      print_log(f"Removing prefix '{sam2_prefix}' from checkpoint keys.")
            #      state_dict = {k.replace(sam2_prefix, ''): v for k, v in state_dict.items() if k.startswith(sam2_prefix)}

            # Handle MLP layer renaming: timm 'mlp.fc1/fc2' vs mmcv 'ffn.layers.0.0/...'
            # This depends on the exact structure of FFN in mmcv
            # Assuming FFN has layers.0.0 for fc1 and layers.1 for fc2
            # adapted_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     new_k = k
            #     if 'mlp.fc1' in k:
            #         new_k = k.replace('mlp.fc1', 'ffn.layers.0.0') # Adjust if FFN structure differs
            #     elif 'mlp.fc2' in k:
            #         new_k = k.replace('mlp.fc2', 'ffn.layers.1') # Adjust if FFN structure differs
            #     adapted_state_dict[new_k] = v
            # state_dict = adapted_state_dict
            # --- End Checkpoint Key Adaptation ---


            # --- Positional Embedding Interpolation ---
            rel_pos_keys = [k for k in state_dict if "pos_embed" in k]
            for k in rel_pos_keys:
                if k in self.state_dict():
                    ckpt_pos_embed = state_dict[k]
                    model_pos_embed = self.state_dict()[k]
                    if ckpt_pos_embed.shape != model_pos_embed.shape:
                        print_log(f'Interpolating position embedding {k} from '
                                  f'{ckpt_pos_embed.shape} to {model_pos_embed.shape}')
                        # Assumes shape is (1, C, H, W) for interpolation
                        if ckpt_pos_embed.ndim == 4 and model_pos_embed.ndim == 4:
                             # Check if channel dimension matches
                             if ckpt_pos_embed.shape[1] != model_pos_embed.shape[1]:
                                 print_log(f'Channel mismatch for {k}, skipping interpolation.', level='warning')
                                 state_dict[k] = model_pos_embed # Use initialized weights instead
                                 continue

                             new_size = model_pos_embed.shape[2:]
                             state_dict[k] = F.interpolate(
                                 ckpt_pos_embed, size=new_size, mode='bicubic', align_corners=False)
                        else:
                             print_log(f'Unexpected shape for {k}, skipping interpolation.', level='warning')
                             state_dict[k] = model_pos_embed # Use initialized weights instead
            # --- End Positional Embedding Interpolation ---

            # Load the adapted state dict
            msg = self.load_state_dict(state_dict, strict=False)
            print_log(f"Load pretrained weights msg: {msg}", logger='current')
        else:
             raise TypeError("init_cfg must be None or a dict with type='Pretrained'")


    @staticmethod
    def _init_weights_fn(m):
        """Standard weight init."""
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
             # Optional: Init conv like ViT/MAE
             fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
             fan_in //= m.groups
             m.weight.data.normal_(0, math.sqrt(2.0 / fan_in))
             if m.bias is not None:
                 m.bias.data.zero_()

    # Optional: Implement fix_init_weight from timm if needed
    # def fix_init_weight(self):
    #     def rescale(param, _layer_id):
    #         param.div_(math.sqrt(2.0 * _layer_id))
    #     # Block index starts from 1 for scaling
    #     for layer_id, layer in enumerate(self.blocks, 1):
    #         rescale(layer.attn.proj.weight.data, layer_id)
    #         # Assuming FFN structure allows accessing fc2 weight directly
    #         if hasattr(layer.ffn, 'layers') and len(layer.ffn.layers) > 1:
    #              rescale(layer.ffn.layers[1].weight.data, layer_id)
    #         elif hasattr(layer.ffn, 'fc2'): # Fallback if structure is different
    #              rescale(layer.ffn.fc2.weight.data, layer_id)


    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Applies and interpolates positional embeddings."""
        B, H, W, C = x.shape
        pos_embed = self.pos_embed # 1, C, Hp, Wp (global size)
        window_embed = self.pos_embed_window # 1, C, Hw, Ww (window size)

        # Interpolate global part
        pos_embed_int = F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)

        # Tile window part
        win_h, win_w = window_embed.shape[-2:]
        if H % win_h != 0 or W % win_w != 0:
             # This case should ideally be handled by ensuring input size compatibility
             # or by padding/cropping window_embed before tiling.
             # For simplicity, we'll just use the interpolated global part if tiling doesn't fit.
             # print_log(f"Feature map size ({H},{W}) not divisible by window pos embed ({win_h},{win_w}). Using global pos embed only.", level='warning')
             final_pos_embed = pos_embed_int
        else:
             tile_h = H // win_h
             tile_w = W // win_w
             # Tile requires explicit expand before reshape to avoid ambiguity
             window_embed_tiled = window_embed.expand(B, -1, -1, -1) # Expand batch dim if needed (should be 1)
             window_embed_tiled = window_embed_tiled.repeat(1, 1, tile_h, tile_w) # Tile spatial dims
             # Add tiled window embed to interpolated global embed
             final_pos_embed = pos_embed_int + window_embed_tiled

        # Permute to B H W C and add to input
        final_pos_embed = final_pos_embed.permute(0, 2, 3, 1)
        return x + final_pos_embed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # Input x: BCHW
        x, hw = self.patch_embed(x) # Output x: BHWC, hw: (H, W)
        x = self._pos_embed(x) # Adds pos embed, keeps BHWC

        outs = []
        cur_block_idx = 0
        for i in range(self.num_stages):
            stage_end_idx = self.stage_ends[i]
            # Iterate through blocks of the current stage
            while cur_block_idx <= stage_end_idx:
                 x = self.blocks[cur_block_idx](x)
                 cur_block_idx += 1

            # After completing a stage, check if it's in out_indices
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                normed_x = norm_layer(x)
                # Convert to NCHW for standard MMSeg output format
                out = normed_x.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()