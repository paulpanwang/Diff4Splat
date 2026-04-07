"""
Preprocessing script to create NPZ files for 3DGS reconstruction.

The model expects inputs in a specific format (similar to VGGT):
- images: (F, 3, H, W) float32 in [0, 1]
- C2W: (F, 4, 4) float32 camera-to-world matrices (OpenCV/COLMAP convention)
- fxfycxcy: (F, 4) float32 normalized intrinsics [fx, fy, cx, cy]
  where fx = fx_px / W, fy = fy_px / H, cx = cx_px / W, cy = cy_px / H

Camera Normalization (applied automatically in dataset):
1. canonical: Transform all cameras so that first camera is at origin
2. norm_xyz: Scale scene so that average point distance from camera is ~1

Usage:
    # From preprocessed poses (e.g., COLMAP output)
    python src/preprocess_npz.py \
        --image_dir ./data/my_video/frames \
        --poses_file ./data/my_video/poses.npz \
        --output_file ./data/my_video/preprocessed.npz
    
    # With synthetic orbit camera (for single image)
    python src/preprocess_npz.py \
        --image_path ./data/single_image.jpg \
        --synthetic_orbit \
        --num_frames 13 \
        --output_file ./data/single_image/orbit.npz
"""

import os
import sys
import argparse
from typing import *
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio
import torchvision.transforms as tvT


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for 3DGS reconstruction")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing input images (sorted)"
    )
    input_group.add_argument(
        "--image_path",
        type=str,
        help="Path to single input image (for synthetic camera)"
    )
    
    # Pose options
    parser.add_argument(
        "--poses_file",
        type=str,
        default=None,
        help="NPZ file containing 'C2W' (F,4,4) and 'fxfycxcy' (F,4) or 'intrinsics' (F,3,3)"
    )
    parser.add_argument(
        "--synthetic_orbit",
        action="store_true",
        help="Generate synthetic orbit camera trajectory"
    )
    
    # Synthetic camera params
    parser.add_argument(
        "--num_frames",
        type=int,
        default=13,
        help="Number of frames in synthetic trajectory"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Field of view in degrees for synthetic camera"
    )
    parser.add_argument(
        "--orbit_radius",
        type=float,
        default=1.5,
        help="Radius of camera orbit (assuming scene is normalized)"
    )
    parser.add_argument(
        "--orbit_height",
        type=float,
        default=0.3,
        help="Height variation in orbit (relative to radius)"
    )
    
    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output NPZ file path"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to (H, W). If not set, keeps original size."
    )
    
    return parser.parse_args()


def load_images_from_dir(
    image_dir: str,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Load images from directory (sorted) and convert to [0,1] float32 tensors."""
    image_paths = sorted([p for p in Path(image_dir).iterdir() 
                          if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    
    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    images = []
    to_tensor = tvT.ToTensor()
    
    for p in image_paths:
        img = iio.imread(str(p))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        
        img_tensor = to_tensor(img)  # (3, H, W) in [0,1]
        
        if resize is not None:
            img_tensor = tvT.Resize(resize, tvT.InterpolationMode.BICUBIC)(img_tensor)
        
        images.append(img_tensor)
    
    images = torch.stack(images)  # (F, 3, H, W)
    return images, [str(p) for p in image_paths]


def load_single_image(
    image_path: str,
    resize: Optional[Tuple[int, int]] = None,
    num_frames: int = 1,
) -> torch.Tensor:
    """Load single image and optionally duplicate to multiple frames."""
    img = iio.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    
    to_tensor = tvT.ToTensor()
    img_tensor = to_tensor(img)
    
    if resize is not None:
        img_tensor = tvT.Resize(resize, tvT.InterpolationMode.BICUBIC)(img_tensor)
    
    # Duplicate if needed
    images = img_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # (F, 3, H, W)
    return images


def load_poses(
    poses_file: str,
    num_frames: Optional[int] = None,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera poses from NPZ file.
    
    Expected format (either):
    - 'C2W': (F, 4, 4) camera-to-world matrices
    - 'fxfycxcy': (F, 4) normalized intrinsics, OR
    - 'intrinsics': (F, 3, 3) camera intrinsics matrices
    
    Returns:
        C2W: (F, 4, 4) float32
        fxfycxcy: (F, 4) float32 normalized
    """
    data = np.load(poses_file)
    
    if "C2W" not in data:
        # Try common alternatives
        if "pose" in data:
            C2W = data["pose"].astype(np.float32)
        elif "extrinsics" in data:
            C2W = data["extrinsics"].astype(np.float32)
        else:
            raise ValueError(f"Poses file must contain 'C2W' or 'pose'. Keys: {list(data.keys())}")
    else:
        C2W = data["C2W"].astype(np.float32)
    
    # Handle intrinsics
    if "fxfycxcy" in data:
        fxfycxcy = data["fxfycxcy"].astype(np.float32)
    elif "intrinsics" in data:
        intrinsics = data["intrinsics"].astype(np.float32)  # (F, 3, 3)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None]  # single frame
        
        if image_height is None or image_width is None:
            raise ValueError("Need image height/width when using 'intrinsics' matrix")
        
        # Convert from pixels to normalized
        fx = intrinsics[:, 0, 0] / image_width
        fy = intrinsics[:, 1, 1] / image_height
        cx = intrinsics[:, 0, 2] / image_width
        cy = intrinsics[:, 1, 2] / image_height
        fxfycxcy = np.stack([fx, fy, cx, cy], axis=-1).astype(np.float32)
    else:
        raise ValueError(
            f"Poses file must contain 'fxfycxcy' or 'intrinsics'. Keys: {list(data.keys())}"
        )
    
    # Trim or expand if needed
    if num_frames is not None:
        if len(C2W) > num_frames:
            C2W = C2W[:num_frames]
            fxfycxcy = fxfycxcy[:num_frames]
        elif len(C2W) < num_frames:
            # Repeat last frame
            C2W = np.concatenate([C2W, np.repeat(C2W[-1:], num_frames - len(C2W), axis=0)], axis=0)
            fxfycxcy = np.concatenate([fxfycxcy, np.repeat(fxfycxcy[-1:], num_frames - len(fxfycxcy), axis=0)], axis=0)
    
    return C2W, fxfycxcy


def generate_synthetic_orbit(
    num_frames: int,
    fov_deg: float,
    image_height: int,
    image_width: int,
    radius: float = 1.5,
    height_var: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic orbit camera trajectory.
    
    Creates a circular orbit around the origin with some height variation.
    Camera looks at origin.
    
    Returns:
        C2W: (F, 4, 4) camera-to-world matrices
        fxfycxcy: (F, 4) normalized intrinsics
    """
    # Focal length from FOV
    fov_rad = np.deg2rad(fov_deg)
    # Assuming vertical FOV
    fy_px = image_height / (2 * np.tan(fov_rad / 2))
    fx_px = fy_px  # square pixels
    fx = fx_px / image_width
    fy = fy_px / image_height
    cx = 0.5
    cy = 0.5
    
    fxfycxcy = np.array([[fx, fy, cx, cy]], dtype=np.float32).repeat(num_frames, axis=0)
    
    # Generate orbit angles
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    
    C2W = []
    for i, angle in enumerate(angles):
        # Circular motion in XZ plane with height variation
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height_var * np.sin(angle * 2)  # height varies twice per orbit
        
        # Camera position
        cam_pos = np.array([x, y, z])
        
        # Look at origin
        forward = -cam_pos  # point to origin
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Up vector (world up)
        world_up = np.array([0, 1, 0])
        
        # Compute camera basis
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Build rotation matrix (camera to world)
        # Columns are right, up, -forward
        R = np.column_stack([right, up, -forward])
        
        # Build 4x4 matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = cam_pos
        
        C2W.append(pose)
    
    C2W = np.stack(C2W).astype(np.float32)
    
    return C2W, fxfycxcy


def create_npz_output(
    images: torch.Tensor,
    C2W: np.ndarray,
    fxfycxcy: np.ndarray,
    output_file: str,
):
    """Create output NPZ file in expected format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Convert images to numpy
    images_np = images.numpy().astype(np.float32)
    
    # Timesteps: normalized [0, 1] for input frames
    num_frames = len(images)
    timesteps = np.linspace(0, 1, num_frames, dtype=np.float32)
    
    # Save
    np.savez(
        output_file,
        images=images_np,
        C2W=C2W.astype(np.float32),
        fxfycxcy=fxfycxcy.astype(np.float32),
        timestep=timesteps,
    )
    
    print(f"Saved to: {output_file}")
    print(f"  - images: {images_np.shape} (float32, [0,1])")
    print(f"  - C2W: {C2W.shape} (float32)")
    print(f"  - fxfycxcy: {fxfycxcy.shape} (float32, normalized)")
    print(f"  - timestep: {timesteps.shape} (float32, [0,1])")


def main():
    args = parse_args()
    
    # Load images
    if args.image_dir is not None:
        images, _ = load_images_from_dir(args.image_dir, args.resize)
        num_frames = len(images)
    else:
        images = load_single_image(args.image_path, args.resize, args.num_frames)
        num_frames = args.num_frames
    
    _, H, W = images.shape[1], images.shape[2], images.shape[3]
    print(f"Images: {tuple(images.shape)}")
    
    # Load or generate poses
    if args.synthetic_orbit:
        print(f"Generating synthetic orbit with {num_frames} frames...")
        C2W, fxfycxcy = generate_synthetic_orbit(
            num_frames=num_frames,
            fov_deg=args.fov,
            image_height=H,
            image_width=W,
            radius=args.orbit_radius,
            height_var=args.orbit_height,
        )
    elif args.poses_file is not None:
        print(f"Loading poses from: {args.poses_file}")
        C2W, fxfycxcy = load_poses(
            args.poses_file,
            num_frames=num_frames,
            image_height=H,
            image_width=W,
        )
    else:
        raise ValueError("Must provide either --poses_file or --synthetic_orbit")
    
    # Verify shapes
    assert len(C2W) == num_frames, f"Poses frame count mismatch: {len(C2W)} vs {num_frames}"
    assert len(fxfycxcy) == num_frames, f"Intrinsics frame count mismatch"
    
    print(f"Poses: C2W={C2W.shape}, fxfycxcy={fxfycxcy.shape}")
    
    # Create output
    create_npz_output(images, C2W, fxfycxcy, args.output_file)
    
    print()
    print("=" * 60)
    print("Next steps:")
    print("=" * 60)
    print(f"1. Run inference:")
    print(f"   python src/infer_nvs.py \\\n"
          f"       --name {os.path.splitext(os.path.basename(args.output_file))[0]} \\\n"
          f"       --data_dir {os.path.dirname(args.output_file)} \\\n"
          f"       --ckpt_path ./resources/ckpts/lrdm_ckpt.safetensors")
    print()


if __name__ == "__main__":
    main()
