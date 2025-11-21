# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


for k in ckpt_list:
    if k.startswith('head'):
        continue
    if k.startswith('norm'):
        new_k = k.replace('norm.', 'ln1.')
    elif k.startswith('patch_embed'):
        if 'proj' in k:
            new_k = k.replace('proj', 'projection')
        else:
            new_k = k
    elif k.startswith('image_encoder'):
        if 'norm' in k:
            new_k = k.replace('norm', 'ln')
        elif 'mlp.fc1' in k:
            new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
        elif 'mlp.fc2' in k:
            new_k = k.replace('mlp.fc2', 'ffn.layers.1')
        elif 'attn.qkv' in k:
            new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
        elif 'attn.proj' in k:
            new_k = k.replace('attn.proj', 'attn.attn.out_proj')
        else:
            new_k = k
        new_k = new_k.replace('blocks.', 'layers.')
    else:
        new_k = k
 
checkpoint_folder = '/cluster/projects/nn10004k/packages_install/sam_checkpoints' 
sam_checkpoint = osp.join(checkpoint_folder, 'sam_vit_l_0b3195.pth')
sam2_checkpoint = osp.join(checkpoint_folder, 'sam2.1_hiera_base_plus.pt')

sam_mmseg_ckpt = '/cluster/home/snf52395/.cache/torch/hub/checkpoints/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth'

checkpoint_sam = CheckpointLoader.load_checkpoint(sam_checkpoint, map_location='cpu')
checkpoint_mmseg = CheckpointLoader.load_checkpoint(sam_mmseg_ckpt, map_location='cpu')
checkpoint_sam2 = CheckpointLoader.load_checkpoint(sam2_checkpoint, map_location='cpu')

sam_key_list = list(checkpoint_sam.keys())
sam2_key_list = list(checkpoint_sam2['model'].keys())

sam_encoder_list = [k for k in sam_key_list if k.startswith('image_encoder.')]
sam2_encoder_list = [k for k in sam2_key_list if k.startswith('image_encoder.')]


sam_mmseg_key_list = list(checkpoint_mmseg['state_dict'].keys())








# Filter and rename keys for the image encoder trunk
encoder_trunk_prefix = 'image_encoder.trunk.'
encoder_state_dict = {}
for k, v in state_dict.items():
    if k.startswith(encoder_trunk_prefix):
        # Remove the prefix to match this module's names
        new_key = k[len(encoder_trunk_prefix):]

        # --- Specific Key Renaming (Examples - Adjust as needed!) ---
        # Patch Embed: SAM2 'proj' -> self.patch_embed.proj
        if new_key.startswith('patch_embed.proj.'):
            pass # Already matches if PatchEmbed uses self.proj

        # Blocks: SAM2 'blocks.X.normY/attn/mlp...' -> self.stages[stage_idx][block_idx_in_stage].normY/attn/mlp...
        # This requires careful mapping from flat block index to stage/block index
        if new_key.startswith('blocks.'):
            parts = new_key.split('.')
            block_idx_flat = int(parts[1])
            # Determine stage and block index within stage
            stage_idx, block_idx_in_stage = self._get_stage_block_idx(block_idx_flat)
            if stage_idx is not None:
                    # Reconstruct key: stages.{stage_idx}.{block_idx_in_stage}.{rest_of_key}
                    rest_of_key = '.'.join(parts[2:])
                    # Handle MLP layer naming difference
                    if 'mlp.layers.0' in rest_of_key:
                        rest_of_key = rest_of_key.replace('mlp.layers.0', 'mlp.0') # Map to Sequential index 0
                    elif 'mlp.layers.1' in rest_of_key:
                        rest_of_key = rest_of_key.replace('mlp.layers.1', 'mlp.3') # Map to Sequential index 3 (Linear)

                    new_key = f"stages.{stage_idx}.{block_idx_in_stage}.{rest_of_key}"
            else:
                logger.warning(f"Could not map flat block index {block_idx_flat} for key: {k}")
                continue # Skip keys that don't map cleanly

        # Downsampling Proj: SAM2 'blocks.X.proj.' -> self.downsamplers[stage_idx].reduction
        if '.proj.' in new_key and new_key.startswith('blocks.'):
                parts = new_key.split('.')
                block_idx_flat = int(parts[1])
                # Downsampling usually happens *after* the last block of a stage
                stage_idx, block_idx_in_stage = self._get_stage_block_idx(block_idx_flat)
                config = self.ARCH_CONFIGS[self.arch]
                if stage_idx is not None and block_idx_in_stage == config['depths'][stage_idx] - 1 and stage_idx < self.num_stages - 1:
                    # This block likely contains the downsampling proj
                    weight_or_bias = parts[-1] # weight or bias
                    new_key = f"downsamplers.{stage_idx}.reduction.{weight_or_bias}"
                else:
                    # This proj might be something else, or mapping is wrong
                    logger.warning(f"Skipping potential proj key with unexpected mapping: {k}")
                    continue

        # Positional Embedding: Check shapes and interpolate if needed
        if new_key == 'pos_embed':
            if v.shape != self.pos_embed.shape:
                logger.warning(f"Interpolating pos_embed from {v.shape} to {self.pos_embed.shape}")
                # Assuming loaded shape is (1, H_old, W_old, C) or (1, N_old, C)
                if v.dim() == 3: # (1, N_old, C)
                    N_old = v.shape[1]
                    H_old = W_old = int(math.sqrt(N_old))
                    if H_old * W_old != N_old:
                        raise ValueError("Pretrained pos_embed N is not square")
                    v = v.view(1, H_old, W_old, -1)
                # Target shape is (1, H_new, W_new, C)
                H_new, W_new = self.pos_embed.shape[1:3]
                v = interpolate_pos_encoding(v, H_new, W_new)

        encoder_state_dict[new_key] = v

# Load the processed state dict
msg = self.load_state_dict(encoder_state_dict, strict=False)
logger.info(f"Pretrained weights loaded with status: {msg}")

except FileNotFoundError:
logger.error(f"Pretrained checkpoint not found at: {ckpt_path}")
except Exception as e:
logger.error(f"Error loading pretrained weights: {e}", exc_info=True)