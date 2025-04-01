#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_sam2_hiera_to_mmseg(ckpt):
    """Convert keys in SAM2 Hiera checkpoint to MMSegmentation style.
    
    Args:
        ckpt (dict): The loaded checkpoint dictionary.
        
    Returns:
        dict: The converted checkpoint dictionary.
    """
    new_ckpt = OrderedDict()
    
    # We only need the encoder part (image_encoder.trunk.*)
    for k, v in ckpt.items():
        if k.startswith('image_encoder.trunk.'):
            # Remove the 'image_encoder.trunk.' prefix
            new_key = k.replace('image_encoder.trunk.', '')
            
            # Map specific keys to match our HieraSAM2 backbone implementation
            # Position embeddings
            if 'pos_embed' in new_key:
                # Keep as is, our implementation expects these keys
                pass
            
            # Patch embedding
            if 'patch_embed' in new_key:
                # Keep as is, our implementation expects these keys
                pass
            
            # Transformer blocks
            if 'blocks' in new_key:
                new_key = new_key.replace('blocks', 'layers')
                
                # Attention
                if 'attn.qkv' in new_key:
                    new_key = new_key.replace('attn.qkv', 'attn.attn.in_proj')
                elif 'attn.proj' in new_key:
                    new_key = new_key.replace('attn.proj', 'attn.attn.out_proj')
                
                # Layer norms - keep as is
                
                # MLP/FFN - SAM2 Hiera already uses layers.0 and layers.1
                # No need to replace mlp.fc1 and mlp.fc2
                
                # Projection layers in some blocks
                # Keep as is, we'll handle them in the backbone implementation
            
            # Final norm
            if new_key == 'norm.weight' or new_key == 'norm.bias':
                # Keep as is
                pass
            
            new_ckpt[new_key] = v
    
    return new_ckpt


def analyze_checkpoint(state_dict):
    """Analyze the checkpoint structure to understand its architecture.
    
    Args:
        state_dict (dict): The loaded checkpoint dictionary.
        
    Returns:
        dict: Analysis results.
    """
    analysis = {}
    
    # Count number of blocks
    block_indices = set()
    for k in state_dict.keys():
        if 'image_encoder.trunk.blocks.' in k:
            parts = k.split('.')
            for i, part in enumerate(parts):
                if part == 'blocks':
                    block_idx = int(parts[i+1])
                    block_indices.add(block_idx)
    
    analysis['num_blocks'] = len(block_indices)
    analysis['max_block_idx'] = max(block_indices) if block_indices else -1
    
    # Check for special blocks with projection layers
    proj_blocks = []
    for k in state_dict.keys():
        if 'image_encoder.trunk.blocks.' in k and '.proj.' in k:
            parts = k.split('.')
            for i, part in enumerate(parts):
                if part == 'blocks':
                    block_idx = int(parts[i+1])
                    if block_idx not in proj_blocks:
                        proj_blocks.append(block_idx)
    
    analysis['proj_blocks'] = sorted(proj_blocks)
    
    # Check position embedding
    pos_embed_keys = [k for k in state_dict.keys() if 'image_encoder.trunk.pos_embed' in k]
    analysis['has_pos_embed'] = len(pos_embed_keys) > 0
    analysis['pos_embed_keys'] = pos_embed_keys
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Convert SAM2 Hiera checkpoint to MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    parser.add_argument('dst', help='save path')
    parser.add_argument('--print-keys', action='store_true', 
                        help='Print all keys in the checkpoint')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze the checkpoint structure')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    if args.analyze:
        analysis = analyze_checkpoint(state_dict)
        print("Checkpoint Analysis:")
        print(f"  Number of blocks: {analysis['num_blocks']}")
        print(f"  Max block index: {analysis['max_block_idx']}")
        print(f"  Blocks with projection layers: {analysis['proj_blocks']}")
        print(f"  Has position embedding: {analysis['has_pos_embed']}")
        print(f"  Position embedding keys: {analysis['pos_embed_keys']}")
    
    if args.print_keys:
        # Print all keys that start with 'image_encoder.trunk.'
        encoder_keys = [k for k in state_dict.keys() if k.startswith('image_encoder.trunk.')]
        print(f"Found {len(encoder_keys)} keys with 'image_encoder.trunk.' prefix:")
        for k in encoder_keys:
            print(f"  {k}")
    
    weight = convert_sam2_hiera_to_mmseg(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)
    
    print(f"Converted checkpoint saved to {args.dst}")
    print(f"Number of keys in the converted checkpoint: {len(weight)}")


if __name__ == '__main__':
    main()