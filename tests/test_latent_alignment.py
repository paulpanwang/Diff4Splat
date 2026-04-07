"""
Test script to verify the latent alignment pipeline:
images -> WanVAE (fixed) -> wan_latent -> [mapping?] -> TinyVAE (trainable) -> recon_images
                                                -> TinyVAE.decode directly -> recon_images2

This tests:
1. Wan2_1_VAEWrapper loading and encoding
2. TinyVAE (TAEHV) structure
3. Tensor shape conversions between the two
4. Optional: MoVieS integration

Usage:
    python tests/test_latent_alignment.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============== Checkpoint Paths ==============
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
WAN21_VAE_PATH = os.path.join(BASE_DIR, "resources", "ckpts", "Wan2.1-I2V-14B-480P", "vae_step_411000.pth")
WAN22_VAE_PATH = os.path.join(BASE_DIR, "resources", "ckpts", "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
MOVIES_CKPT_PATH = os.path.join(BASE_DIR, "resources", "ckpts", "movies_ckpt", "002000", "model.safetensors")


def check_file_exists(path, name):
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status} {name}: {path}")
    return exists


def print_dimension_info():
    """Print key dimension information about WanVAE and TinyVAE."""
    print("\n" + "=" * 60)
    print("Key Dimension Information")
    print("=" * 60)
    print("""
Wan2.1 VAE:
  - Input shape:    (B, 3, T, H, W)   range: [-1, 1]
  - Latent shape:   (B, 16, f, h, w)   f = 1 + (T-1)//4,  h = H//8,  w = W//8
  - Has built-in mean/std normalization when encoding

TinyVAE (TAEHV):
  - Input shape:    (N, T, 3, H, W)    range: [0, 1] for encode_video
  - Latent shape:   (N, f, 16, h, w)   f = T // 4 (approximately), h = H//8, w = W//8
  - Note: axis order is (N, T, C, H, W) vs Wan's (B, C, T, H, W)!

Key Issue for Alignment:
  1. Different axis orders: need to permute
  2. Different time compression handling (Wan2.1 keeps first frame)
  3. Different normalization: Wan uses fixed mean/std, TinyVAE outputs approx Gaussian
  4. Input range difference: [-1,1] vs [0,1]
""")


def test_wan_vae(device):
    """Test Wan2_1_VAEWrapper loading and encoding."""
    if not os.path.exists(WAN21_VAE_PATH):
        print(f"\nSkipping Wan2.1 VAE test (checkpoint missing)")
        return None

    print("\nLoading Wan2.1 VAE Wrapper...")
    try:
        from src.models.networks.wan_wrapper import Wan2_1_VAEWrapper

        wan_vae = Wan2_1_VAEWrapper(WAN21_VAE_PATH)
        wan_vae = wan_vae.to(device).eval()

        print(f"  ✓ Wan2_1_VAEWrapper loaded successfully")
        print(f"    - Mean shape: {wan_vae.mean.shape}")
        print(f"    - Std shape: {wan_vae.std.shape}")

        # Test with dummy input
        B, T, H, W = 1, 5, 256, 512
        dummy_images_wan = torch.rand(B, 3, T, H, W).to(device) * 2 - 1  # [-1, 1]

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                wan_latent = wan_vae.encode(dummy_images_wan)

        print(f"\n  Encoding test:")
        print(f"    Input:  {dummy_images_wan.shape}")
        print(f"    Output: {wan_latent.shape}")

        f_expected = 1 + (T - 1) // 4
        h_expected = H // 8
        w_expected = W // 8
        print(f"    Expected f={f_expected}, h={h_expected}, w={w_expected}")

        return wan_vae

    except Exception as e:
        print(f"  ✗ Failed to load Wan2.1 VAE: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tiny_vae(device):
    """Test TinyVAE (TAEHV) loading and encoding/decoding."""
    print("\nLoading TinyVAE (TAEHV)...")
    try:
        from src.models.tiny_vae import TAEHV

        tiny_vae = TAEHV(checkpoint_path=None)
        tiny_vae = tiny_vae.to(device)

        print(f"  ✓ TAEHV created successfully")
        print(f"    - latent_channels: {tiny_vae.latent_channels}")
        print(f"    - frames_to_trim (decode): {tiny_vae.frames_to_trim}")

        # Test encoding
        N, T, H, W = 1, 9, 256, 512
        dummy_images_tiny = torch.rand(N, T, 3, H, W).to(device)

        print(f"\n  Encoding test:")
        print(f"    Input:  {dummy_images_tiny.shape} (NTCHW format expected by encode_video)")

        tiny_latent = tiny_vae.encode_video(dummy_images_tiny, parallel=True)
        print(f"    Output: {tiny_latent.shape}")

        t_expected = T // 4
        h_expected = H // 8
        w_expected = W // 8
        print(f"    Expected approx: (N={N}, T~{t_expected}, C=16, H={h_expected}, W={w_expected})")

        # Test decoding
        print(f"\n  Decoding test:")
        print(f"    Input latent: {tiny_latent.shape}")
        recon = tiny_vae.decode_video(tiny_latent, parallel=True)
        print(f"    Output: {recon.shape}")
        print(f"    (frames_to_trim={tiny_vae.frames_to_trim} was trimmed from start)")

        return tiny_vae

    except Exception as e:
        print(f"  ✗ Failed with TinyVAE: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_movies_model(device):
    """Test MoVieS (SplatRecon) model loading."""
    if not os.path.exists(MOVIES_CKPT_PATH):
        return

    print("\n\n" + "=" * 60)
    print("Testing MoVieS (SplatRecon) Loading")
    print("=" * 60)

    try:
        from src.options import opt_dict
        from src.models import SplatRecon
        from safetensors.torch import load_file

        opt = opt_dict["movies_static"]
        print(f"Using config: movies_static")
        print(f"  input_res: {opt.input_res}")
        print(f"  num_input_frames: {opt.num_input_frames}")

        print("\nCreating SplatRecon model...")
        model = SplatRecon(opt, load_lpips=False)
        model = model.to(device)

        print(f"Loading MoVieS checkpoint from: {MOVIES_CKPT_PATH}")
        state_dict = load_file(MOVIES_CKPT_PATH)

        print(f"Checkpoint keys (first 10):")
        for i, k in enumerate(list(state_dict.keys())[:10]):
            print(f"  {k}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"\nMissing keys (first 10): {missing[:10] if len(missing) > 10 else missing}")
        if unexpected:
            print(f"Unexpected keys (first 10): {unexpected[:10] if len(unexpected) > 10 else unexpected}")
        print(f"  ✓ Loaded with strict=False")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()


class LatentAxisMapper(nn.Module):
    """
    Maps between WanVAE latent format and TinyVAE latent format.

    Wan2.1 format: (B, C, T, H, W)
    TinyVAE format: (N, T, C, H, W)  for encode_video output
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

    def wan_to_tiny(self, wan_latent):
        """wan_latent: (B, C, T, H, W) -> returns: (N, T, C, H, W)"""
        return wan_latent.permute(0, 2, 1, 3, 4)

    def tiny_to_wan(self, tiny_latent):
        """tiny_latent: (N, T, C, H, W) -> returns: (B, C, T, H, W)"""
        return tiny_latent.permute(0, 2, 1, 3, 4)


class LatentAlignmentMLP(nn.Module):
    """A small network to align distributions."""
    def __init__(self, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv3d(latent_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv3d(hidden_dim, latent_dim, kernel_size=1)
        self.gn = nn.GroupNorm(8, hidden_dim)

    def forward(self, x):
        identity = x
        h = F.gelu(self.gn(self.conv1(x)))
        out = self.conv2(h) + identity
        return out


def test_alignment_modules():
    """Test the alignment modules with dummy data."""
    print("\n\n" + "=" * 60)
    print("Testing Alignment Modules")
    print("=" * 60)

    mapper = LatentAxisMapper()
    aligner = LatentAlignmentMLP(latent_dim=16)

    wan_latent_dummy = torch.randn(1, 16, 2, 32, 64)
    print(f"\n  Wan latent:    {wan_latent_dummy.shape}")

    tiny_format = mapper.wan_to_tiny(wan_latent_dummy)
    print(f"  -> Tiny format: {tiny_format.shape}")

    aligned = aligner(wan_latent_dummy)
    print(f"  Aligned:       {aligned.shape}")


def main():
    print("=" * 60)
    print("Latent Alignment Pipeline Test")
    print("=" * 60)

    # Check required checkpoints
    print("\nChecking Required Checkpoints")
    print("=" * 60)
    all_exist = True
    all_exist &= check_file_exists(WAN21_VAE_PATH, "Wan2.1 VAE")
    all_exist &= check_file_exists(WAN22_VAE_PATH, "Wan2.2 VAE")
    all_exist &= check_file_exists(MOVIES_CKPT_PATH, "MoVieS Checkpoint")

    if not all_exist:
        print("\nWARNING: Some checkpoints are missing!")
        print("Please download:")
        print("  1. Wan2.1 VAE from: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
        print("  2. MoVieS from: https://huggingface.co/chenguolin/MoVieS")

    print_dimension_info()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    test_wan_vae(device)
    test_tiny_vae(device)
    test_movies_model(device)
    test_alignment_modules()

    print("\n\n" + "=" * 60)
    print("Test script completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
