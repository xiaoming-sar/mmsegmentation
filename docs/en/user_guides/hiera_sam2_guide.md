# Using SAM2 Hiera Backbone in MMSegmentation

This guide explains how to convert and use the SAM2 Hiera backbone in MMSegmentation.

## Overview

SAM2 (Segment Anything Model 2) uses a Hiera (Hierarchical Vision Transformer) as its image encoder. This guide shows how to:

1. Convert the SAM2 Hiera checkpoint to be compatible with MMSegmentation
2. Use the converted checkpoint with the HieraSAM2 backbone

## SAM2 Hiera Architecture

The SAM2 Hiera architecture has several key features:
- 24 transformer layers
- Hierarchical structure with projection layers at specific blocks (2, 5, 21)
- Position embeddings (`pos_embed` and `pos_embed_window`)
- MLP structure with `layers.0` and `layers.1` instead of traditional `fc1` and `fc2`

## Converting the Checkpoint

To convert a SAM2 Hiera checkpoint to MMSegmentation format:

```bash
python tools/model_converters/sam2hiera2mmseg.py ${SAM2_CHECKPOINT_PATH} ${OUTPUT_PATH}
```

For example:

```bash
python tools/model_converters/sam2hiera2mmseg.py sam2_hiera_base.pth hiera_sam2_base_converted.pth
```

### Analyzing the Checkpoint

To analyze the structure of the checkpoint before conversion:

```bash
python tools/model_converters/sam2hiera2mmseg.py ${SAM2_CHECKPOINT_PATH} ${OUTPUT_PATH} --analyze
```

This will print information about the number of blocks, which blocks have projection layers, and position embedding details.

### Inspecting Keys

To inspect the keys in the original checkpoint:

```bash
python tools/model_converters/sam2hiera2mmseg.py ${SAM2_CHECKPOINT_PATH} ${OUTPUT_PATH} --print-keys
```

## Using the Converted Checkpoint

After converting the checkpoint, you can use it with the HieraSAM2 backbone in your configuration file:

```python
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='HieraSAM2',
        img_size=1024,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=24,  # SAM2 Hiera has 24 layers
        num_heads=12,
        mlp_ratio=4.,
        out_indices=[2, 5, 11, 23],  # Hierarchical feature maps at different stages
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN'),
        patch_norm=True,
        frozen_stages=-1,
        proj_layers=[2, 5, 21],  # Layers with projection in SAM2 Hiera
        use_pos_embed=True,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='path/to/converted/checkpoint.pth')),
    # ... rest of your model config
)
```

## Sample Configuration

A sample configuration is provided in `configs/_base_/models/hiera_sam2.py`. You can use it as a starting point for your own configuration.

## Key Parameters

- `num_layers`: Set to 24 for SAM2 Hiera
- `proj_layers`: Set to [2, 5, 21] to match the SAM2 Hiera architecture
- `use_pos_embed`: Set to True to use position embeddings
- `out_indices`: Choose which layers to extract features from (e.g., [2, 5, 11, 23])

## Notes

- The HieraSAM2 backbone implementation is based on the architecture described in the SAM2 paper.
- Only the encoder part of the SAM2 model is used (the part with keys starting with `image_encoder.trunk.*`).
- The implementation handles the special projection layers at blocks 2, 5, and 21.
- Position embeddings (`pos_embed` and `pos_embed_window`) are supported.
- The MLP structure matches SAM2 Hiera's implementation with `layers.0` and `layers.1`.