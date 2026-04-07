"""
Test script to verify Wan model loading and check dimension mismatches.

Usage:
    python tests/test_wan_model.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_wan_model_simple(wan_dir):
    """Simple test: just load WanModel from config.json."""
    print("\n" + "=" * 60)
    print("Test 1: Simple WanModel Loading")
    print("=" * 60)

    try:
        from src.models.networks.wan_modules.model import WanModel

        print(f"\nLoading model from: {wan_dir}")
        print(f"Config exists: {os.path.exists(os.path.join(wan_dir, 'config.json'))}")

        model = WanModel.from_pretrained(wan_dir)
        print(f"✓ Model loaded successfully!")

        # Print model configuration
        print(f"\nModel config:")
        print(f"  dim: {model.config.dim}")
        print(f"  num_layers: {model.config.num_layers}")
        print(f"  num_heads: {model.config.num_heads}")
        print(f"  in_dim: {model.config.in_dim}")
        print(f"  out_dim: {model.config.out_dim}")
        print(f"  patch_size: {model.config.patch_size}")

        # Print some key layer shapes
        print(f"\nKey layer shapes:")
        print(f"  patch_embedding.weight: {model.patch_embedding.weight.shape}")
        print(f"  text_embedding.0.weight: {model.text_embedding[0].weight.shape}")
        print(f"  time_embedding.0.weight: {model.time_embedding[0].weight.shape}")
        if hasattr(model, 'time_projection'):
            print(f"  time_projection.1.weight: {model.time_projection[1].weight.shape}")

        # Count blocks
        num_blocks = len(model.blocks)
        print(f"\nNumber of blocks: {num_blocks}")

        # Check first block attention layer
        if num_blocks > 0:
            print(f"  blocks.0.self_attn.q.weight: {model.blocks[0].self_attn.q.weight.shape}")
            print(f"  blocks.0.cross_attn.q.weight: {model.blocks[0].cross_attn.q.weight.shape}")

        return model

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_checkpoint_comparison(model, ckpt_path):
    """Compare model with fine-tuned checkpoint."""
    print("\n" + "=" * 60)
    print("Test 2: Comparing with Fine-tuned Checkpoint")
    print("=" * 60)

    if model is None:
        print("Skipping: model not loaded")
        return

    print(f"\nLoading checkpoint from: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"✗ Checkpoint not found: {ckpt_path}")
        return

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        module = ckpt['module']

        print(f"\nCheckpoint layer shapes (diffusion.model.*):")

        # Map model weights to checkpoint key names
        mismatches = []
        matches = []

        for name, param in model.named_parameters():
            ckpt_name = f"diffusion.model.{name}"

            if ckpt_name in module:
                ckpt_shape = module[ckpt_name].shape
                model_shape = param.shape

                if ckpt_shape == model_shape:
                    matches.append(name)
                else:
                    mismatches.append(f"  {name}: model={model_shape}, checkpoint={ckpt_shape}")
            elif 'plucker_embed' not in name:  # Skip plucker_embed since it's added separately
                print(f"  {name}: NOT FOUND in checkpoint")

        # Also check for extra keys in checkpoint
        extra_keys = []
        for k in module.keys():
            if k.startswith('diffusion.model.'):
                short_name = k.replace('diffusion.model.', '')
                # Check if this is in the model
                found = False
                for name, param in model.named_parameters():
                    if name == short_name:
                        found = True
                        break
                if not found:
                    extra_keys.append(short_name)

        if mismatches:
            print(f"\n✗ Shape mismatches ({len(mismatches)}):")
            for m in mismatches:
                print(m)
        else:
            print(f"\n✓ All common layer shapes match!")

        if extra_keys:
            print(f"\nExtra keys in checkpoint ({len(extra_keys)}):")
            for k in extra_keys[:20]:  # Show first 20
                print(f"  {k}")
            if len(extra_keys) > 20:
                print(f"  ... and {len(extra_keys) - 20} more")

        if matches:
            print(f"\nMatching keys: {len(matches)}")

        print(f"\nNote: patch_embedding will likely mismatch because")
        print(f"      fine-tuning added plucker_embed channels.")

    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("Wan Model Test Suite")
    print("=" * 60)

    # Paths
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    wan_dir = os.path.join(base_dir, "resources", "ckpts", "Wan2.2-TI2V-5B")
    ckpt_path = os.path.join(base_dir, "resources", "ckpts", "wan_cc_60000", "006000",
                               "pytorch_model", "mp_rank_00_model_states.pt")

    # Run tests
    model = test_wan_model_simple(wan_dir)
    test_checkpoint_comparison(model, ckpt_path)

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
