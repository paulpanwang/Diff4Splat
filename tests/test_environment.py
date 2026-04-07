"""
Test script to verify the Diff4Splat environment and check for missing components.

Usage:
    python tests/test_environment.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_python_version():
    """Test Python version compatibility."""
    print(f"[1/6] Python version: {sys.version}")
    return sys.version_info >= (3, 8)

def test_basic_imports():
    """Test basic package imports."""
    print("\n[2/6] Testing basic imports...")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

def test_key_packages():
    """Test key package imports."""
    print("\n[3/6] Testing key package imports...")
    packages = [
        "transformers", "diffusers", "accelerate", "safetensors",
        "omegaconf", "einops", "cv2", "decord", "lpips"
    ]
    all_ok = True
    for pkg in packages:
        try:
            if pkg == "cv2":
                import cv2
                print(f"  ✓ opencv-python: {cv2.__version__}")
            else:
                module = __import__(pkg)
                version = getattr(module, "__version__", "OK")
                print(f"  ✓ {pkg}: {version}")
        except ImportError as e:
            print(f"  ✗ {pkg}: MISSING - {e}")
            all_ok = False
    return all_ok

def test_project_imports():
    """Test project module imports."""
    print("\n[4/6] Testing project imports...")
    try:
        from src.options import opt_dict
        print(f"  ✓ src.options loaded")
        print(f"  ✓ Available option types: {list(opt_dict.keys())}")
    except ImportError as e:
        print(f"  ✗ src.options import failed: {e}")
        return False

    try:
        from src.models import Wan, SplatRecon
        print(f"  ✓ src.models.Wan imported")
        print(f"  ✓ src.models.SplatRecon imported")
        return True
    except ImportError as e:
        print(f"  ✗ src.models import failed: {e}")
        return False

def test_checkpoints():
    """Check downloaded checkpoints."""
    print("\n[5/6] Checking downloaded checkpoints...")
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'ckpts')
    if os.path.exists(ckpt_dir):
        print(f"  ✓ Checkpoint directory exists: {ckpt_dir}")
        for f in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, f)
            size = os.path.getsize(fpath) / 1024 / 1024 / 1024
            print(f"    - {f}: {size:.2f} GB")
    else:
        print(f"  ✗ Checkpoint directory not found: {ckpt_dir}")
    return os.path.exists(ckpt_dir)

def print_summary():
    """Print summary of required components."""
    print("\n[6/6] Summary of required components...")
    print("""
Required components for full functionality:
1. ✓ Wan diffusion model weights - In downloaded checkpoint
2. ? Wan VAE weights - Need separate file (Wan2.2_VAE.pth)
3. ? Wan base model config - For WanModel.from_pretrained()
4. ✓ MoVieS checkpoint - movies_ckpt.safetensors

Current paths expected:
- wan_dir: ./resources/ckpts/Wan2.2-TI2V-5B
- vae_path: ./resources/ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
- MoVieS: ./resources/movies_ckpt.safetensors
""")

def main():
    print("=" * 60)
    print("Diff4Splat Environment Test")
    print("=" * 60)

    all_passed = True
    all_passed &= test_python_version()
    all_passed &= test_basic_imports()
    all_passed &= test_key_packages()
    all_passed &= test_project_imports()
    all_passed &= test_checkpoints()
    print_summary()

    print("\n" + "=" * 60)
    if all_passed:
        print("Environment test complete! All checks passed.")
    else:
        print("Environment test complete! Some checks failed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
