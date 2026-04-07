"""Analyze the checkpoint to understand model dimensions - part 2."""
import torch

ckpt_path = "./resources/ckpts/wan_cc_60000/006000/pytorch_model/mp_rank_00_model_states.pt"

print("=" * 60)
print("Analyzing checkpoint - Block structure")
print("=" * 60)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
module = ckpt['module']

# Look at keys in block 0
print("\nKeys in diffusion.model.blocks.0 (first 40):")
block0_keys = []
for k in module.keys():
    if k.startswith('diffusion.model.blocks.0.'):
        block0_keys.append(k.replace('diffusion.model.blocks.0.', ''))

for k in sorted(block0_keys)[:40]:
    print(f"  {k}")
    
# Look at mlp keys if any
print("\n\nLooking for mlp or ffn:")
for k in sorted(block0_keys):
    if 'mlp' in k.lower() or 'ffn' in k.lower():
        shape = tuple(module[f'diffusion.model.blocks.0.{k}'].shape)
        print(f"  {k}: {shape}")

# Look at time_projection more carefully
print("\n\nTime projection:")
for k in module.keys():
    if 'time_projection' in k:
        shape = tuple(module[k].shape)
        print(f"  {k}: {shape}")

# Note: time_projection.1.shape is (9216, 1536)
# 9216 / 1536 = 6, which is typical for swiglu (3 * 2)
# So ffn_dim = 9216 / 3 = 3072? Wait, let me think...
# Actually for SwiGLU: x -> gate * sigmoid(gate) * up, where:
#   gate and up are each (ffn_dim, dim), so total (2*ffn_dim, dim)
# But we have 9216 which is 6 * 1536 = 6*dim

print("\n" + "=" * 60)
print("Derived config.json for Wan2.2-TI2V-5B")
print("=" * 60)

# From the checkpoint:
dim = 1536
# time_projection.1.weight: (9216, 1536)
# 9216 = 6 * 1536 = 6 * dim
# For swiglu, this is typically 3 * ffn_dim (where ffn_dim = 2*dim or similar)
# 9216 / 3 = 3072 = 2 * dim
ffn_dim = 3072  # 9216 / 3
freq_dim = 256
text_dim = 4096
num_layers = 30
num_heads = 16  # Common for dim=1536, head_dim=96
patch_size = [1, 2, 2]

print(f"""
{{
  "_class_name": "WanModel",
  "_diffusers_version": "0.30.0",
  "dim": {dim},
  "eps": 1e-06,
  "ffn_dim": {ffn_dim},
  "freq_dim": {freq_dim},
  "in_dim": 16,  // Wait... need to check
  "model_type": "t2v",
  "num_heads": {num_heads},
  "num_layers": {num_layers},
  "out_dim": 16,
  "patch_size": {patch_size},
  "qk_norm": true,
  "cross_attn_norm": true,
  "text_dim": {text_dim},
  "text_len": 512,
  "window_size": [-1, -1]
}}
""")

# Wait, there's a confusion about in_dim. Let me think:
# - patch_embedding in checkpoint has shape (1536, 36, 1, 2, 2)
# - This is the modified version after plucker_embed was added
# - The original in_dim before modification was less than 36
#
# But what was the original?
# 
# Looking at Wan2.1:
# - latent_dim = 16
# - patch_embedding in_dim = 36 (16 + 16 + 4 for masks or something?)
#
# For Wan2.2:
# - latent_dim = 48 (from options.py)
# - But checkpoint shows 36...
#
# Actually wait - let me re-read wan_wrapper.py more carefully:
#
# Line 280-287:
#   new_conv = nn.Conv3d(extra_in_dim + self.model.in_dim, ...)
#   new_conv.weight.data[:, :self.model.in_dim, ...] = self.model.patch_embedding.weight.data
#   new_conv.weight.data[:, self.model.in_dim:, ...] = 0.
#
# So the FIRST self.model.in_dim channels are the original.
# The extra_in_dim channels come after.
#
# We have 36 total. If extra_in_dim=48, then:
#   self.model.in_dim + 48 = 36 -> impossible, can't be negative!
#
# So maybe extra_in_dim is NOT 48 when this checkpoint was trained?
# Let me check the options again...

print("\nRe-analyzing patch_embedding logic...")
print(f"  checkpoint patch_embedding shape: (1536, 36, 1, 2, 2)")
print(f"  This means in_dim + extra_in_dim = 36")
print(f"  But opt.plucker_embed_dim = 48 (from options.py)")
print(f"  This doesn't add up...")
print(f"\n  Maybe for this checkpoint, input_plucker=False?")
print(f"  Or maybe plucker_embed_dim was different?")
print(f"\n  Alternatively:")
print(f"    For Wan2.1, latent_dim=16, in_dim=36 (16*2 + 4 for masks)")
print(f"    For Wan2.2-TI2V-5B, maybe latent_dim=16 too?")
print(f"    Then in_dim=36 matches exactly, and maybe plucker wasn't added yet?")

# Actually wait - looking at the checkpoint more carefully:
# The key prefixes include 'plucker_embed: 2 keys', so plucker WAS trained!
# 
# Let me look at what's in img_emb, head, etc.
print("\n\nWhat's in img_emb:")
for k in module.keys():
    if 'img_emb' in k:
        print(f"  {k}: {tuple(module[k].shape)}")

print("\nWhat's in head:")
for k in module.keys():
    if 'head.' in k:
        print(f"  {k}: {tuple(module[k].shape)}")

# Head should have shape (out_dim, dim) or similar
# If out_dim == latent_dim == 16, that makes sense

print("\n" + "=" * 60)
print("Most likely config")
print("=" * 60)

print("""
Based on the analysis:
- patch_embedding.weight: (1536, 36, 1, 2, 2)
- This suggests in_dim = 36
- Which matches Wan2.1's setup (not Wan2.2's 48 latent dim)

Wait but options.py says:
  vae_path expects Wan2.2_VAE.pth
  latent_dim = 48

Maybe the issue is:
1. The diffusion model (WanModel) has in_dim=16 (latent space)
2. But the patch_embedding was modified to accept extra channels
3. For this checkpoint (camera control), maybe plucker_embed_dim is small?

Actually let me check Wan2.2_TI2V configs more carefully.
For Wan2.2-TI2V-5B:
- Text to Image to Video
- latent_dim = 16 (same as Wan2.1)
- But the VAE outputs 48 channels?

Wait, no - looking at options.py:
  compression_ratio = (4, 16, 16) for Wan2.2-TI2V-5B
  latent_dim = 48
  
And Wan2_2_VAEWrapper has:
  z_dim=48

So Wan2.2 VAE has 48 latent channels.
But the checkpoint shows patch_embedding with 36 in_channels...

Hypothesis:
- The base WanModel.from_pretrained() for Wan2.2-TI2V-5B expects in_dim=16?
- No wait, that doesn't match Wan2.1 either...

Actually, let me look at the actual wan_wrapper.py to see what happens:

1. self.model = WanModel.from_pretrained(pretrained_dir)  # loads base model
2. Then: new_conv = nn.Conv3d(extra_in_dim + self.model.in_dim, ...)
3. Then: new_conv.weight.data[:, :self.model.in_dim, ...] = self.model.patch_embedding.weight.data

So the self.model.in_dim is from config.json.
The checkpoint has the MODIFIED patch_embedding.

So the real question is: what is the ORIGINAL in_dim before modification?
If checkpoint has 36 total, and extra_in_dim = 48, that can't be.

Let me check what happens if extra_in_dim was actually 20? No that doesn't make sense.

Wait - maybe the checkpoint WAS NOT trained with input_plucker=True for the diffusion head?
No... the checkpoint has 'plucker_embed: 2 keys'

Let me think differently:
- WanTI2VDiffusionWrapper is created with extra_in_dim = plucker_embed_dim = 48
- This creates patch_embedding with in_channels = 48 + original_in_dim
- But what if the WanModel has original_in_dim = 36?
- That would be 48 + 36 = 84 channels, but checkpoint has only 36.

I think the issue is:
- The config for Wan2.2-TI2V-5B might have in_dim=16 (like Wan2.1)
- Then 16 + 20 = 36? But why 20?

Actually let me just check WanModel's default values:
  from model.py line 311-322:
    model_type='t2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,  <-- DEFAULT
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=16,
    num_layers=32,

So the default WanModel has in_dim=16, dim=2048.
But our checkpoint has dim=1536.

So for Wan2.2-TI2V-5B specifically:
- dim = 1536 (not 2048)
- num_layers = 30 (not 32)
- ffn_dim = let's derive from time_projection

From time_projection.1.weight: (9216, 1536)
In the model code, time_projection.1 is after the 2nd layer of the MLP.
Looking at model.py more carefully would help.

Actually let me just create a config with:
- dim=1536
- num_layers=30
- in_dim=16 (default, and the patch_embedding will be modified in wrapper)
- ffn_dim=3072 (since 9216/3 = 3072 for swiglu where 3*ffn_dim is stored)
- num_heads=16 (1536/16=96, reasonable head_dim)
""")
