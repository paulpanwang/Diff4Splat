"""Analyze the checkpoint to understand model dimensions."""
import torch
import os
from collections import defaultdict

ckpt_path = "./resources/ckpts/wan_cc_60000/006000/pytorch_model/mp_rank_00_model_states.pt"

print("=" * 60)
print("Analyzing checkpoint dimensions")
print("=" * 60)

print(f"\nLoading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
module = ckpt['module']

print(f"\nTotal keys in module: {len(module)}")

# Group keys by prefix
prefixes = defaultdict(list)
for k in module.keys():
    parts = k.split('.')
    if parts[0] == 'diffusion' and parts[1] == 'model':
        prefix = parts[2] if len(parts) > 2 else 'root'
        prefixes[prefix].append(k)
    else:
        prefixes[parts[0]].append(k)

print(f"\nKey prefixes:")
for prefix, keys in sorted(prefixes.items()):
    print(f"  {prefix}: {len(keys)} keys")

# Analyze block structure
print("\n" + "=" * 60)
print("Analyzing block structure (blocks.*)")
print("=" * 60)

block_numbers = set()
for k in module.keys():
    if 'blocks.' in k:
        parts = k.split('.')
        for i, p in enumerate(parts):
            if p == 'blocks' and i + 1 < len(parts):
                if parts[i+1].isdigit():
                    block_numbers.add(int(parts[i+1]))

print(f"\nNumber of blocks: {len(block_numbers)}")
print(f"Block indices: {sorted(list(block_numbers))[:5]}...{sorted(list(block_numbers))[-5:]}")

# Check key layer shapes
print("\n" + "=" * 60)
print("Key layer shapes in checkpoint")
print("=" * 60)

important_keys = [
    'diffusion.model.patch_embedding.weight',
    'diffusion.model.text_embedding.0.weight',
    'diffusion.model.text_embedding.2.weight',
    'diffusion.model.time_embedding.0.weight',
    'diffusion.model.time_embedding.2.weight',
    'diffusion.model.time_projection.1.weight',
    'diffusion.model.blocks.0.self_attn.q.weight',
    'diffusion.model.blocks.0.self_attn.k.weight',
    'diffusion.model.blocks.0.self_attn.v.weight',
    'diffusion.model.blocks.0.self_attn.o.weight',
    'diffusion.model.blocks.0.cross_attn.q.weight',
    'diffusion.model.blocks.0.mlp.fc1.weight',
    'diffusion.model.blocks.0.mlp.fc2.weight',
    'plucker_embed.conv_in.weight',
    'plucker_embed.conv_out.weight',
    'negative_prompt_embed',
]

for k in important_keys:
    if k in module:
        shape = tuple(module[k].shape) if hasattr(module[k], 'shape') else 'N/A'
        print(f"  {k}: {shape}")
    else:
        # Try to find similar keys
        found = False
        for mk in module.keys():
            if k.replace('diffusion.model.', '') in mk:
                shape = tuple(module[mk].shape) if hasattr(module[mk], 'shape') else 'N/A'
                print(f"  {mk}: {shape}")
                found = True
                break
        if not found:
            print(f"  {k}: NOT FOUND")

# Calculate dimensions
print("\n" + "=" * 60)
print("Calculated model dimensions")
print("=" * 60)

# From patch_embedding
# shape: (dim, in_dim + extra_dim, patch_t, patch_h, patch_w)
patch_shape = tuple(module['diffusion.model.patch_embedding.weight'].shape)
print(f"\npatch_embedding.shape: {patch_shape}")
print(f"  Model dim (out): {patch_shape[0]}")
print(f"  Total in_channels: {patch_shape[1]}")
print(f"  Patch size: {patch_shape[2:]}")

# From attn q weight
# shape: (dim, dim) for q,k,v,o
attn_shape = tuple(module['diffusion.model.blocks.0.self_attn.q.weight'].shape)
print(f"\nself_attn.q.weight.shape: {attn_shape}")
print(f"  dim: {attn_shape[0]}")

# From mlp fc1
# shape: (ffn_dim, dim)
mlp_shape = tuple(module['diffusion.model.blocks.0.mlp.fc1.weight'].shape)
print(f"\nmlp.fc1.weight.shape: {mlp_shape}")
print(f"  ffn_dim: {mlp_shape[0]}")

# From text_embedding
# shape: (dim, text_dim)
text_emb_shape = tuple(module['diffusion.model.text_embedding.0.weight'].shape)
print(f"\ntext_embedding.0.weight.shape: {text_emb_shape}")
print(f"  text_dim: {text_emb_shape[1]}")

# From time_embedding
# shape: (dim, freq_dim)
time_emb_shape = tuple(module['diffusion.model.time_embedding.0.weight'].shape)
print(f"\ntime_embedding.0.weight.shape: {time_emb_shape}")
print(f"  freq_dim: {time_emb_shape[1]}")

# Summary
print("\n" + "=" * 60)
print("Config.json parameters for Wan2.2-TI2V-5B")
print("=" * 60)

print(f"""
{{
  "_class_name": "WanModel",
  "_diffusers_version": "0.30.0",
  "dim": {patch_shape[0]},
  "eps": 1e-06,
  "ffn_dim": {mlp_shape[0]},
  "freq_dim": {time_emb_shape[1]},
  "in_dim": 48,  # Standard for Wan2.2, will be modified for plucker
  "model_type": "t2v",
  "num_heads": {attn_shape[0] // 96},  // Assuming head_dim=96
  "num_layers": {len(block_numbers)},
  "out_dim": 48,  // Same as in_dim for Wan models
  "patch_size": [{patch_shape[2]}, {patch_shape[3]}, {patch_shape[4]}],
  "qk_norm": true,
  "cross_attn_norm": true,
  "text_dim": {text_emb_shape[1]},
  "text_len": 512,
  "window_size": [-1, -1]
}}
""")

print(f"\nNote: patch_embedding shows {patch_shape[1]} input channels.")
print(f"This includes 48 (base in_dim) + 48 (plucker_embed) = 96? But we have {patch_shape[1]}.")
print(f"Actually looking at the checkpoint, the base WanModel.from_pretrained")
print(f"will initialize with in_dim=48, and then the code in wan_wrapper.py")
print(f"will create a NEW patch_embedding with in_dim=48+extra_in_dim (48 for plucker).")

print("\n" + "=" * 60)
print("Analysis complete")
print("=" * 60)
