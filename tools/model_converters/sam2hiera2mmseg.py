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
            # Patch embedding
            if 'stem' in new_key:
                new_key = new_key.replace('stem', 'patch_embed')
                if 'proj' in new_key:
                    new_key = new_key.replace('proj', 'projection')
            
            # Transformer blocks
            if 'blocks' in new_key:
                new_key = new_key.replace('blocks', 'layers')
                
                # Attention
                if 'attn.qkv' in new_key:
                    new_key = new_key.replace('attn.qkv', 'attn.attn.in_proj')
                elif 'attn.proj' in new_key:
                    new_key = new_key.replace('attn.proj', 'attn.attn.out_proj')
                
                # Layer norms
                if 'norm1' in new_key:
                    new_key = new_key.replace('norm1', 'norm1')
                elif 'norm2' in new_key:
                    new_key = new_key.replace('norm2', 'norm2')
                
                # MLP/FFN
                if 'mlp.fc1' in new_key:
                    new_key = new_key.replace('mlp.fc1', 'mlp.layers.0.0')
                elif 'mlp.fc2' in new_key:
                    new_key = new_key.replace('mlp.fc2', 'mlp.layers.1')
            
            # Final norm
            if new_key == 'norm.weight' or new_key == 'norm.bias':
                new_key = new_key.replace('norm', 'norm')
            
            new_ckpt[new_key] = v
    
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert SAM2 Hiera checkpoint to MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    parser.add_argument('dst', help='save path')
    parser.add_argument('--print-keys', action='store_true', 
                        help='Print all keys in the checkpoint')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
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