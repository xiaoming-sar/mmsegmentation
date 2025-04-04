# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) OpenMMLab. All rights reserved. # MMSeg Adaptation
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

import math
from functools import partial
from typing import List, Tuple, Callable, Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _ntuple # MMSeg Adaptation: For tuple handling

# MMSeg Adaptation: Necessary imports from mm* libraries
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmengine.logging import MMLogger
from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn import build_norm_layer

from timm.models.layers import DropPath, Mlp

# MMSeg Adaptation: Assume hiera_utils.py is in the same directory or PYTHONPATH
try:
    from .hiera_utils import pretrained_model, conv_nd, do_pool, do_masked_conv, Unroll, Reroll
except ImportError:
    # Provide dummy implementations or raise clearer error if utils are missing
    raise ImportError("hiera_utils.py not found. Please ensure it's in the correct path.")
    # # Dummy example (replace with actual utils):
    # class Unroll(nn.Module): def __init__(self, *args, **kwargs): super().__init__(); self.identity = nn.Identity() ; def forward(self, x): return self.identity(x)
    # class Reroll(nn.Module): def __init__(self, *args, **kwargs): super().__init__(); self.identity = nn.Identity() ; def forward(self, x, *args): return self.identity(x) # Note: Reroll needs specific logic
    # def conv_nd(dims): return {2: nn.Conv2d, 3: nn.Conv3d}[dims]
    # def do_pool(x, stride): return F.max_pool2d(x, kernel_size=stride, stride=stride) # Example for 2D
    # def do_masked_conv(x, conv, mask): return conv(x) # Simplified


# MMSeg Adaptation: Register with MMSeg's backbone registry
from mmseg.registry import MODELS

# --- Functions/Classes Moved Mostly As-Is (Dependencies handled) ---

class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.
    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See Unroll for more details.
    (Code identical to original, moved as is)
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0, # Flat window size
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        # Note: N here might be the number of *unmasked* tokens if MAE was used
        # In standard backbone usage N = total tokens in the current grid arrangement
        # window_size needs to be the *flat* size (e.g., 8*8=64)
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )
        if num_windows == 0 and self.use_mask_unit_attn:
             # Handle cases where N is smaller than q_stride * window_size, maybe fallback to global
             # This might happen if image size is small relative to mask units/strides
             # Forcing global attention:
             num_windows = 1
             # Warning or assertion could be useful here depending on expected behavior
             # print(f"Warning: N ({N}) < q_stride*window_size ({self.q_stride}*{self.window_size}). Forcing Global Attention.")


        # B, N, C -> B, N, 3*C_out -> B, N, num_windows, 3, H, C_head -> 3, B, H, num_windows, N', C_head
        # N' = N / num_windows
        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] # Each is [B, H, num_windows, N', C_head]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            # Maxpool over the q_stride dimension
            # q: [B, H, num_windows, q_stride * (N'/q_stride), C_head]
            #   -> [B, H, num_windows, q_stride, (N'/q_stride), C_head]
            # pool -> [B, H, num_windows, (N'/q_stride), C_head]
            if q.shape[3] % self.q_stride != 0:
                 raise ValueError(f"Shape {q.shape} dim 3 ({q.shape[3]}) is not divisible by q_stride {self.q_stride}")
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            ) # -> [B, H, num_windows, N'/q_stride, C_head]


        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
             # Reshape q,k,v for SDPA: Needs [B, H, Seq, C_head] like format
             # q: [B, H, num_windows, N_q, C_head] -> [B*H*num_windows, N_q, C_head]
             # k: [B, H, num_windows, N_k, C_head] -> [B*H*num_windows, N_k, C_head]
             # v: [B, H, num_windows, N_v, C_head] -> [B*H*num_windows, N_v, C_head]
             N_q = q.shape[-2]
             N_k = k.shape[-2]
             N_v = v.shape[-2] # N_k == N_v
             q = q.reshape(-1, N_q, self.head_dim)
             k = k.reshape(-1, N_k, self.head_dim)
             v = v.reshape(-1, N_v, self.head_dim)

             x = F.scaled_dot_product_attention(q, k, v) # Output: [B*H*num_windows, N_q, C_head]
             # Reshape back
             # [B*H*num_windows, N_q, C_head] -> [B, H, num_windows, N_q, C_head]
             # Transpose & reshape to final output format:
             # [B, H, num_windows, N_q, C_head] -> [B, num_windows, N_q, H, C_head]
             # -> [B, num_windows * N_q, H*C_head] = [B, N_out, C_out]
             x = x.view(B, self.heads, num_windows, N_q, self.head_dim)
             x = x.permute(0, 2, 3, 1, 4).reshape(B, num_windows * N_q, self.dim_out)

        else:
             # Manual Attention Calculation
             # q: [B, H, num_windows, N_q, C_head]
             # k: [B, H, num_windows, N_k, C_head] -> k.T: [B, H, num_windows, C_head, N_k]
             # attn = (q @ k.T) * scale
             attn = (q * self.scale) @ k.transpose(-1, -2) # [B, H, num_windows, N_q, N_k]
             attn = attn.softmax(dim=-1)
             # x = attn @ v
             # v: [B, H, num_windows, N_v, C_head] (N_v == N_k)
             x = (attn @ v) # [B, H, num_windows, N_q, C_head]
             # Reshape to final output format
             N_q = x.shape[-2]
             x = x.transpose(1, 3).reshape(B, num_windows * N_q, self.dim_out) # [B, N_out, C_out]

        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    """
    Single Hiera Transformer Block.
    (Code identical to original, but norm_layer now expects a callable, handled in Hiera main class)
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Callable = nn.LayerNorm, # MMSeg Adaptation: Expect callable
        act_layer: Callable = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0, # Flat window size
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        # MMSeg Adaptation: Use norm_layer directly
        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        # MMSeg Adaptation: Use norm_layer directly
        self.norm2 = norm_layer(dim_out)
        # MMSeg Adaptation: Use nn.GELU if act_layer is string? Or assume callable. Let's assume callable.
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            # MMSeg Adaptation: Define proj as identity if dims are same for consistent attribute access
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, N, C]
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm) # Output: [B, N_out, C_out]

        # Apply projection and pooling to input before residual connection if needed
        if self.dim != self.dim_out:
             # Pool the input to match the output spatial size reduction caused by q_stride
             # Assume do_pool handles token sequences [B, N, C] -> [B, N/q_stride, C]
             # The projection changes channel dimension
             x_residual = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
             # MMSeg Adaptation: Check if do_pool exists and works on [B,N,C]
             # If do_pool needs spatial layout, this needs rework, but MaskUnitAttention implies N is flat token list.
             # Let's assume do_pool works on [B, N, C] by max-pooling chunks of size q_stride.
        else:
             # If no dimension change, no projection needed.
             # If q_stride > 1, the input `x` needs pooling to match `attn_out` shape for residual.
             if self.attn.q_stride > 1:
                 x_residual = do_pool(x, stride=self.attn.q_stride)
             else:
                 x_residual = x

        x = x_residual + self.drop_path(attn_out)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Patch embed that supports any number of spatial dimensions (1d, 2d, 3d).
    (Code identical to original, relies on hiera_utils.conv_nd and do_masked_conv)
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # MMSeg Adaptation: Mask is removed for standard backbone usage
        # x = do_masked_conv(x, self.proj, mask) # Original
        x = self.proj(x) # Output: [B, C_out, D', H', W'] or [B, C_out, H', W']

        # Flatten spatial/temporal dims and transpose to [B, N, C]
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x

# --- Main Hiera Class Adaptation ---

@MODELS.register_module()
class HieraSAM2(BaseModule): # MMSeg Adaptation: Inherit from BaseModule
    """ Hiera Vision Transformer Backbone for MMSegmentation """
    # MMSeg Adaptation: Removed @has_config, PyTorchModelHubMixin

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 1024, # MMSeg Adaptation: Use img_size convention
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,
        q_stride: Tuple[int, ...] = (2, 2), # e.g., (2,2) for 2D, (1,2,2) for 3D
        mask_unit_size: Tuple[int, ...] = (8, 8), # e.g., (8,8) for 2D, (1,8,8) for 3D

        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_cfg: dict = dict(type='LN', eps=1e-6), # MMSeg Adaptation: Use norm_cfg
        act_layer: Callable = nn.GELU, # MMSeg Adaptation: Standardize act_layer if needed, here GELU
        sep_pos_embed: bool = False,
        interpolate_pos_encoding: bool = True, # MMSeg Adaptation: Flag for pos embed interpolation
        out_indices: Sequence[int] = (0, 1, 2, 3), # MMSeg Adaptation: Indices of stages to output
        frozen_stages: int = -1, # MMSeg Adaptation: Standard backbone freezing param
        init_cfg: Optional[dict] = None, # MMSeg Adaptation: Standard init config
        # MMSeg Adaptation: Removed class/head specific params: num_classes, head_dropout, head_init_scale
    ):
        super().__init__(init_cfg=init_cfg) # MMSeg Adaptation: Call BaseModule init

        self.img_size = _pair(img_size) # Ensure tuple like (224, 224)
        # MMSeg Adaptation: Determine spatial dims from patch_kernel length
        self.spatial_dims = len(patch_kernel)
        if self.spatial_dims not in [2, 3]:
             raise ValueError("Hiera backbone currently supports 2D or 3D input.")

        # MMSeg Adaptation: Handle norm_layer from norm_cfg
        self.norm_layer =  build_norm_layer(norm_cfg, embed_dim)[1]# Get the LayerNorm class

        depth = sum(stages)
        self.stages_depth = stages
        self.patch_stride_spatial = patch_stride[-self.spatial_dims:] # Spatial part of stride
        self.tokens_spatial_shape = [i // s for i, s in zip(self.img_size[-self.spatial_dims:], self.patch_stride_spatial)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages), f"q_pool ({q_pool}) must be less than num stages ({len(stages)})"
        self.q_pool, self.q_stride_spatial = q_pool, q_stride[-self.spatial_dims:]
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        # MMSeg Adaptation: Calculate mask spatial shape based on spatial dims
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size[-self.spatial_dims:])
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        # MMSeg Adaptation: Store out_indices and frozen_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.interpolate_pos_encoding = interpolate_pos_encoding

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
             if self.spatial_dims != 3:
                  raise ValueError("Separable pos embed currently only supported for 3D input")
             # Example assumes 3D: T, H, W
             self.pos_embed_spatial = nn.Parameter(
                  torch.zeros(1, self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2], embed_dim)
             )
             self.pos_embed_temporal = nn.Parameter(
                  torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
             )
        else:
             self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules (CRITICAL: Assumes hiera_utils are available)
        # MMSeg Adaptation: Ensure input_size, patch_stride, q_stride match expected format of Unroll/Reroll
        unroll_q_strides = [q_stride] * len(self.stage_ends[:-1]) # q_stride repeated
        self.unroll = Unroll(
            self.img_size, patch_stride, unroll_q_strides
        )
        self.reroll = Reroll(
            self.img_size,
            patch_stride,
            unroll_q_strides,
            self.stage_ends,
            q_pool,
        )

        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage_idx = 0
        cur_embed_dim = embed_dim
        cur_num_heads = num_heads
        cur_flat_mu_size = flat_mu_size # Track current flat mask unit size after pooling
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = cur_embed_dim
            is_stage_transition = (i - 1) in self.stage_ends
            is_q_pool_block = i in q_pool_blocks

            if is_stage_transition:
                dim_out = int(cur_embed_dim * dim_mul)
                cur_num_heads = int(cur_num_heads * head_mul)
                cur_stage_idx += 1

            if is_q_pool_block:
                 if cur_flat_mu_size % flat_q_stride != 0:
                      raise ValueError(f"Current flat_mu_size {cur_flat_mu_size} not divisible by flat_q_stride {flat_q_stride} at block {i}")
                 cur_flat_mu_size //= flat_q_stride


            # Mask unit or global attention. Lag by 1 block.
            use_mask_unit_attn = mask_unit_attn[cur_stage_idx]

            block = HieraBlock(
                dim=cur_embed_dim,
                dim_out=dim_out,
                heads=cur_num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=self.norm_layer, # Pass the callable norm_layer
                act_layer=act_layer,
                q_stride=(flat_q_stride if is_q_pool_block else 1),
                window_size=cur_flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            cur_embed_dim = dim_out
            self.blocks.append(block)

        # MMSeg Adaptation: Add norm layers for each output stage
        self.num_features = [] # Store output feature dimensions
        for i, stage_end_idx in enumerate(self.stage_ends):
             norm_layer = self.norm_layer(self.blocks[stage_end_idx].dim_out)
             layer_name = f'norm{i}'
             self.add_module(layer_name, norm_layer)
             self.num_features.append(self.blocks[stage_end_idx].dim_out)


        # MMSeg Adaptation: Removed classification head init
        # MMSeg Adaptation: Weight init handled by init_weights / init_cfg

        # MMSeg Adaptation: Freeze stages logic
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages param and norm in patch_embed."""
        if self.frozen_stages >= 0:
             self.patch_embed.eval()
             for param in self.patch_embed.parameters():
                 param.requires_grad = False
             if self.sep_pos_embed:
                   self.pos_embed_spatial.requires_grad = False
                   self.pos_embed_temporal.requires_grad = False
             else:
                   self.pos_embed.requires_grad = False


        for i in range(1, self.frozen_stages + 1):
             # Freeze blocks in the stage
             stage_start_idx = 0 if i == 1 else self.stage_ends[i-2] + 1
             stage_end_idx = self.stage_ends[i-1]
             for idx in range(stage_start_idx, stage_end_idx + 1):
                  m = self.blocks[idx]
                  m.eval()
                  for param in m.parameters():
                      param.requires_grad = False

             # Freeze the norm layer for the stage if it exists and is needed
             if i-1 in self.out_indices:
                 norm_layer = getattr(self, f'norm{i-1}')
                 norm_layer.eval()
                 for param in norm_layer.parameters():
                     param.requires_grad = False


    # MMSeg Adaptation: Added init_weights method
    def init_weights(self):
        """Initialize weights based on init_cfg."""
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
             logger.info('Initiating weights randomly.')
             # Default initialization
             if self.sep_pos_embed:
                 trunc_normal_(self.pos_embed_spatial, std=0.02)
                 trunc_normal_(self.pos_embed_temporal, std=0.02)
             else:
                 trunc_normal_(self.pos_embed, std=0.02)
             self.apply(self._init_weights) # Apply default linear/conv init
        elif isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained':
             pretrained_path = self.init_cfg['checkpoint']
             logger.info(f'Loading pretrained weights from {pretrained_path}')

             # Define key mappings (example, adjust as needed based on checkpoint keys)
             revise_keys = [
                 (r'^head\.projection', None), # Remove classification head
                 (r'^norm\.', None),        # Remove final norm if model had one for classification
                 (r'^patch_embed\.proj\.', 'patch_embed.proj.'),
                 # Add more specific mappings if needed, e.g. block names
                 # (r'^blocks\.(\d+)\.', r'blocks.\1.') # Simple identity map if names match
             ]
             # If pos embed keys differ:
             # revise_keys.extend([
             #      (r'^pos_embed', 'pos_embed'), # if names match
             # ])


             load_checkpoint(self, pretrained_path, map_location='cpu', strict=False, logger=logger, revise_keys=revise_keys)
        else:
             # Initialize using other methods specified in init_cfg (e.g., Kaiming, Xavier)
             # This relies on BaseModule's init_weights implementation
             super().init_weights()

    # Keep original _init_weights for default init
    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
             # Use mmcv trunc_normal_ for consistency if preferred, or keep original
             # nn.init.trunc_normal_(m.weight, std=0.02)
             trunc_normal_(m.weight, std=0.02)
             if m.bias is not None:
                 nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, (nn.LayerNorm)): # MMSeg Adaptation: Check specific norm type used
             nn.init.constant_(m.bias, init_bias)
             nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Which parameters to exclude from weight decay."""
        if self.sep_pos_embed:
             return {"pos_embed_spatial", "pos_embed_temporal"}
        else:
             return {"pos_embed"}

    def _get_pos_embed(self, N_spatial: int, N_temporal: int = 1) -> torch.Tensor:
         """ Get positional encoding, potentially interpolating it."""
         if self.sep_pos_embed:
             # Assuming T, H, W format for tokens_spatial_shape
             if N_temporal != self.tokens_spatial_shape[0]:
                  # Interpolate temporal embedding
                  pos_embed_temporal = F.interpolate(
                      self.pos_embed_temporal.transpose(1, 2).reshape(1, -1, self.tokens_spatial_shape[0]),
                      size=N_temporal, mode='linear', align_corners=False
                  ).reshape(1, -1, N_temporal).transpose(1, 2)
             else:
                  pos_embed_temporal = self.pos_embed_temporal

             target_spatial_size = N_spatial # Should be H*W
             if target_spatial_size != self.pos_embed_spatial.shape[1]:
                   # Interpolate spatial embedding (assuming 2D spatial for simplicity here)
                   num_patches_w = self.tokens_spatial_shape[-1]
                   num_patches_h = self.tokens_spatial_shape[-2]
                   # Reshape to 2D grid: [1, H*W, C] -> [1, H, W, C] -> [1, C, H, W]
                   pos_embed_spatial_grid = self.pos_embed_spatial.reshape(
                       1, num_patches_h, num_patches_w, -1).permute(0, 3, 1, 2)
                   # Calculate target H, W (need to know aspect ratio or assume square)
                   # This requires knowing the target image's H, W after patch embedding
                   # Let's assume target H/W can be inferred from N_spatial if roughly square
                   target_h = target_w = int(math.sqrt(N_spatial))
                   if target_h * target_w != N_spatial:
                        # Fallback or error for non-square - requires target H/W info
                        logger = MMLogger.get_current_instance()
                        logger.warning(f"Cannot interpolate non-square spatial pos embed from {self.pos_embed_spatial.shape[1]} to {N_spatial}. Using original.")
                        pos_embed_spatial_interp = self.pos_embed_spatial # No interpolation
                   else:
                        pos_embed_spatial_interp = F.interpolate(
                             pos_embed_spatial_grid, size=(target_h, target_w), mode='bicubic', align_corners=False
                        )
                        # Reshape back: [1, C, H_new, W_new] -> [1, H_new, W_new, C] -> [1, H_new*W_new, C]
                        pos_embed_spatial_interp = pos_embed_spatial_interp.permute(0, 2, 3, 1).reshape(1, target_spatial_size, -1)

             else:
                   pos_embed_spatial_interp = self.pos_embed_spatial

             # Combine: Repeat spatial embed for each temporal step, add temporal embed repeated for each spatial token
             pos_embed = pos_embed_spatial_interp.repeat(
                 1, N_temporal, 1 # Repeat H*W embed T times
             ) + torch.repeat_interleave(
                 pos_embed_temporal, # Shape [1, T, C]
                 target_spatial_size, # Repeat each T embed H*W times
                 dim=1,
             )

         else:
              # Standard non-separable case
              total_tokens_target = N_spatial # Assuming 2D input for simplicity if not sep_pos_embed
              if total_tokens_target != self.pos_embed.shape[1]:
                   if self.interpolate_pos_encoding:
                       # Interpolate 1D embedding or reshape to 2D/3D and interpolate
                       # Example for 2D interpolation:
                       num_patches = self.pos_embed.shape[1]
                       num_patches_w = self.tokens_spatial_shape[-1]
                       num_patches_h = self.tokens_spatial_shape[-2]
                       if num_patches != num_patches_h * num_patches_w:
                            raise ValueError("pos_embed shape mismatch with token shape")

                       target_h = target_w = int(math.sqrt(total_tokens_target))
                       if target_h * target_w != total_tokens_target:
                            # Use 1D linear interpolation as fallback
                            logger = MMLogger.get_current_instance()
                            logger.warning(f"Non-square target token number {total_tokens_target}. Using 1D linear interpolation for pos embed.")
                            pos_embed = F.interpolate(
                                self.pos_embed.transpose(1,2), size=total_tokens_target, mode='linear', align_corners=False
                            ).transpose(1,2)
                       else:
                            pos_embed_grid = self.pos_embed.reshape(1, num_patches_h, num_patches_w, -1).permute(0, 3, 1, 2)
                            pos_embed_interp = F.interpolate(
                                pos_embed_grid, size=(target_h, target_w), mode='bicubic', align_corners=False
                            )
                            pos_embed = pos_embed_interp.permute(0, 2, 3, 1).reshape(1, total_tokens_target, -1)
                   else:
                        pos_embed = self.pos_embed # No interpolation, assumes fixed size or handled by caller
              else:
                   pos_embed = self.pos_embed

         return pos_embed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function."""
        B, C, *spatial_shape = x.shape # H, W or T, H, W

        # MMSeg Adaptation: Check input size vs self.img_size if not interpolating
        if not self.interpolate_pos_encoding and tuple(spatial_shape[-self.spatial_dims:]) != tuple(self.img_size):
             logger = MMLogger.get_current_instance()
             logger.warning(f"Input size {tuple(spatial_shape)} differs from model's configured img_size {self.img_size}. Positional encoding might be incorrect.")


        x = self.patch_embed(x) # -> [B, N, C_embed]
        N = x.shape[1] # Number of tokens

        # MMSeg Adaptation: Calculate target spatial/temporal tokens for pos embed
        # This requires knowing how N relates back to spatial/temporal dims.
        # Assuming N is product of spatial dims (and temporal if sep_pos_embed)
        if self.sep_pos_embed:
             # Need to infer N_spatial, N_temporal from N and original aspect ratio
             # This is tricky without knowing the target spatial shape *after* patch_embed
             # Let's assume N = N_temporal * N_spatial and aspect ratios are preserved roughly
             orig_T = self.tokens_spatial_shape[0]
             orig_H = self.tokens_spatial_shape[1]
             orig_W = self.tokens_spatial_shape[2]
             orig_N_spatial = orig_H * orig_W
             # Infer target T, H, W from current input shape spatial_shape and patch stride
             current_T = spatial_shape[0] // self.patch_stride[0] if self.spatial_dims == 3 else 1
             current_H = spatial_shape[-2] // self.patch_stride[-2]
             current_W = spatial_shape[-1] // self.patch_stride[-1]
             current_N_spatial = current_H * current_W
             current_N = current_T * current_N_spatial
             if current_N != N:
                  logger = MMLogger.get_current_instance()
                  logger.warning(f"Calculated token number {current_N} mismatch with actual {N}. Positional encoding might be incorrect.")
                  # Fallback: use original shapes for pos embed? Or error?
                  # Using calculated N for pos embed size seems more robust if input size varies
                  N_temporal = current_T
                  N_spatial = current_N_spatial
             else:
                  N_temporal = current_T
                  N_spatial = current_N_spatial

        else: # Non-separable, assume 2D input or flattened 3D treated as 1D sequence
             N_spatial = N # Treat all tokens as spatial for interpolation purposes
             N_temporal = 1 # Not used


        x = x + self._get_pos_embed(N_spatial=N_spatial, N_temporal=N_temporal)
        # MMSeg Adaptation: Assuming Unroll takes [B, N, C] and returns potentially rearranged [B, N', C]
        x = self.unroll(x)

        # MMSeg Adaptation: Removed MAE mask logic

        outs = []
        cur_block_idx = 0
        for stage_idx, stage_depth in enumerate(self.stages_depth):
             for i in range(stage_depth):
                 x = self.blocks[cur_block_idx](x)
                 cur_block_idx += 1

             # Check if output is needed for this stage
             if stage_idx in self.out_indices:
                 # MMSeg Adaptation: Reroll expects the *end* block index of the stage
                 stage_end_block_idx = self.stage_ends[stage_idx]
                 # Reroll converts token list back to spatial: [B, N_stage, C_stage] -> [B, C_stage, H_stage, W_stage] (or T)
                 # Critical: Ensure Reroll implementation produces the correct spatial shape.
                 out_spatial = self.reroll(x, stage_end_block_idx) # mask=None for standard inference

                 # Apply the normalization layer for this stage output
                 norm_layer = getattr(self, f'norm{stage_idx}')
                 # Norm needs [B, N, C] format. Reshape, Norm, Reshape back.
                 B_s, C_s, *spatial_s = out_spatial.shape
                 out_spatial_flat = out_spatial.view(B_s, C_s, -1).transpose(1, 2) # B, N_s, C_s
                 out_norm = norm_layer(out_spatial_flat)
                 out_norm_spatial = out_norm.transpose(1, 2).view(B_s, C_s, *spatial_s) # B, C_s, H_s, W_s

                 outs.append(out_norm_spatial)

        # MMSeg Adaptation: Return tuple of features from specified stages
        return tuple(outs)