"""
Novel View Synthesis (NVS) Inference Script for 3DGS Reconstruction.

Given posed input images and camera trajectory, this script:
1. Uses VGGSplaT backbone to predict 3DGS attributes (depth, color, scale, opacity, rotation)
2. (For dynamic scenes) Uses motion head to predict per-pixel motion vectors
3. Renders novel views using GaussianRenderer
4. Saves outputs as MP4 videos

Usage:
    python src/infer_nvs.py --name sample_name --data_dir ./data --ckpt_path ./ckpt/model.safetensors

Input format (NPZ):
    - images: (F, 3, H, W) float32 in [0, 1]
    - C2W: (F, 4, 4) float32 camera-to-world matrices
    - fxfycxcy: (F, 4) float32 normalized intrinsics [fx, fy, cx, cy]
"""

import os
import argparse
import numpy as np
import imageio.v2 as iio
from typing import *
from torch import Tensor

import torch
from safetensors.torch import load_file
from einops import rearrange

from src.options import opt_dict
from src.models import SplatRecon
from src.utils import (
    tensor_to_video,
    colorize_depth,
    normalize_among_last_dims,
)


def concat_videos(video_paths: List[str], output_path: str) -> None:
    """Concatenate multiple videos horizontally for comparison."""
    videos = [iio.mimread(video_path) for video_path in video_paths]
    videos = np.concatenate(videos, axis=0)
    iio.mimwrite(output_path, videos, macro_block_size=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3DGS NVS Inference")
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Sample name (used for input/output directories)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="resources/DAVIS",
        help="Directory containing input NPZ files"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="resources/ckpts/lrdm_ckpt.safetensors",
        help="Path to model checkpoint (safetensors format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Output directory for rendered videos"
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=None,
        help="Number of input frames (default: use all frames in NPZ)"
    )
    parser.add_argument(
        "--num_output_frames",
        type=int,
        default=None,
        help="Number of output frames (default: same as input)"
    )
    parser.add_argument(
        "--opt_type",
        type=str,
        default="lrdm",
        choices=["lrdm", "lrdm_static", "lrdm_dynamic", "drecon", "srecon", "movies", "movies_static"],
        help="Model config type: lrdm (dynamic, default), lrdm_static (static)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    return parser.parse_args()


@torch.inference_mode()
def main(args: argparse.Namespace):
    # Create output directory
    sample_output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(sample_output_dir, exist_ok=True)

    # Load config and model
    print(f"Loading model config: {args.opt_type}")
    opt = opt_dict[args.opt_type]
    
    print(f"Loading model from: {args.ckpt_path}")
    model = SplatRecon(opt, load_lpips=False)  # LPIPS not needed for inference
    
    # Load checkpoint
    if not os.path.exists(args.ckpt_path):
        # Try alternative path
        alt_path = os.path.join(os.path.dirname(__file__), "..", args.ckpt_path)
        if os.path.exists(alt_path):
            args.ckpt_path = alt_path
    
    state_dict = load_file(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"Warning: Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    
    model.eval()
    model = model.to(args.device)

    # Load input data
    npz_path = os.path.join(args.data_dir, f"{args.name}.npz")
    print(f"Loading input data from: {npz_path}")
    
    npz = np.load(npz_path)
    
    # Input images
    input_images = torch.from_numpy(npz["images"]).float().unsqueeze(0)  # (1, F_in, 3, H, W)
    F_total = input_images.shape[1]
    
    # Camera poses
    input_C2W = torch.from_numpy(npz["C2W"]).float().unsqueeze(0)  # (1, F_in, 4, 4)
    
    # Camera intrinsics (normalized)
    if "fxfycxcy" in npz:
        input_fxfycxcy = torch.from_numpy(npz["fxfycxcy"]).float().unsqueeze(0)  # (1, F_in, 4)
    else:
        # Try to reconstruct from intrinsics matrix if available
        if "intrinsics" in npz:
            intrinsics = torch.from_numpy(npz["intrinsics"]).float()  # (F, 3, 3) or (F, 4, 4)
            H, W = input_images.shape[-2:]
            # Extract normalized [fx, fy, cx, cy]
            fx = intrinsics[:, 0, 0] / W
            fy = intrinsics[:, 1, 1] / H  
            cx = intrinsics[:, 0, 2] / W
            cy = intrinsics[:, 1, 2] / H
            input_fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).unsqueeze(0)
        else:
            raise ValueError("Input NPZ must contain either 'fxfycxcy' or 'intrinsics'")
    
    # Determine input/output frame counts
    if args.num_input_frames is None:
        args.num_input_frames = F_total
    if args.num_output_frames is None:
        args.num_output_frames = args.num_input_frames
    
    F_in = args.num_input_frames
    F_out = args.num_output_frames
    
    # Trim or pad if needed
    input_images = input_images[:, :F_in, ...]
    input_C2W = input_C2W[:, :F_in, ...]
    input_fxfycxcy = input_fxfycxcy[:, :F_in, ...]
    
    # Timesteps for input/output (normalized [0, 1])
    # These are used for time-aware motion prediction in dynamic scenes
    input_timesteps = torch.linspace(0, 1, steps=F_in).unsqueeze(0)  # (1, F_in)
    output_timesteps = torch.linspace(0, 1, steps=F_out).unsqueeze(0)  # (1, F_out)
    
    # Output intrinsics (use first input view's intrinsics for all outputs)
    output_fxfycxcy = input_fxfycxcy[:, 0:1, :].repeat(1, F_out, 1)

    # Save input video for reference
    input_video_path = os.path.join(sample_output_dir, "input_video.mp4")
    iio.mimwrite(input_video_path, tensor_to_video(input_images), macro_block_size=1)
    print(f"Saved input video: {input_video_path}")

    # Move tensors to device
    input_images = input_images.to(device=args.device, dtype=torch.bfloat16)
    input_C2W = input_C2W.to(device=args.device, dtype=torch.bfloat16)
    input_fxfycxcy = input_fxfycxcy.to(device=args.device, dtype=torch.bfloat16)
    input_timesteps = input_timesteps.to(device=args.device, dtype=torch.bfloat16)
    output_timesteps = output_timesteps.to(device=args.device, dtype=torch.bfloat16)
    output_fxfycxcy = output_fxfycxcy.to(device=args.device, dtype=torch.bfloat16)

    # Forward pass through backbone
    print(f"Running model forward pass...")
    print(f"  Input:  {tuple(input_images.shape)} images + poses")
    print(f"  Output: {F_out} novel views")
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # backbone_outputs: Dict with static 3DGS attributes:
        #   - "depth": (B, F_in, 1, H, W)
        #   - "color": (B, F_in, 3, H, W) or SH coefficients
        #   - "scale": (B, F_in, 3, H, W)
        #   - "rotation": (B, F_in, 4, H, W) - quaternion
        #   - "opacity": (B, F_in, 1, H, W)
        #
        # pred_motions: (B, F_out, F_in, 4, H, W) - 3D offsets + confidence
        # pred_motion_gs: List[F_out] of Dict - time-conditioned 3DGS attributes
        
        backbone_outputs, pred_motions, pred_motion_gs = \
            model.backbone(
                input_images,
                input_C2W,
                input_fxfycxcy,
                input_timesteps,
                output_timesteps,
                frames_chunk_size=16,  # Process in chunks for memory efficiency
            )

    # Save depth visualization
    if "depth" in backbone_outputs:
        depth_video_path = os.path.join(sample_output_dir, "output_depth.mp4")
        # Invert depth for better visualization (farther = darker)
        iio.mimwrite(
            depth_video_path,
            tensor_to_video(colorize_depth(1./backbone_outputs["depth"], batch_mode=True)),
            macro_block_size=1
        )
        print(f"Saved depth map: {depth_video_path}")

    # ============== Rendering Mode 1: Fixed camera, moving timesteps ==============
    # This shows dynamics at a fixed camera location
    print("Rendering: Fixed camera, moving timesteps...")
    
    render_outputs: Dict[str, Tensor] = {}
    render_outputs_list: List[Dict[str, Tensor]] = []
    
    for i in range(F_out):
        # Set motion for this timestep
        if pred_motions is not None:
            # pred_motions[:, i, :, :3] = xyz offsets at output timestep i
            backbone_outputs["offset"] = pred_motions[:, i, :, :3, ...]
        
        # Set time-conditioned 3DGS attributes if available
        if pred_motion_gs is not None:
            backbone_outputs.update(pred_motion_gs[i])
        
        # Render
        # - Use input cameras for 3DGS centers
        # - But render from FIRST camera's view
        _render_outputs = model.gs_renderer.render(
            backbone_outputs,
            input_C2W,  # Cameras where 3DGS are located
            input_fxfycxcy,
            input_C2W[:, 0:1, ...],  # Target camera: first input camera
            output_fxfycxcy[:, 0:1, ...],
        )
        render_outputs_list.append(_render_outputs)
    
    # Concatenate outputs
    for k in render_outputs_list[0].keys():
        if k not in ["gaussian_usage", "voxel_ratio"]:
            render_outputs[k] = torch.cat(
                [render_outputs_list[i][k] for i in range(F_out)], 
                dim=1
            )
    
    render_images_cam0 = render_outputs["image"]  # (B, F_out, 3, H, W)
    
    cam0_video_path = os.path.join(sample_output_dir, "output_render_camera0.mp4")
    iio.mimwrite(cam0_video_path, tensor_to_video(render_images_cam0), macro_block_size=1)
    print(f"  Saved: {cam0_video_path}")
    
    # Motion visualization
    if pred_motions is not None:
        # Motion at camera 0, varying time
        motion_cam0 = rearrange(
            normalize_among_last_dims(
                rearrange(pred_motions[:, 0, :, :3, ...], "b f c h w -> b c f h w"),
                num_dims=3
            ),
            "b c f h w -> b f c h w"
        )
        motion_cam0_path = os.path.join(sample_output_dir, "output_motion_camera0.mp4")
        iio.mimwrite(motion_cam0_path, tensor_to_video(motion_cam0), macro_block_size=1)
        
        # Motion at time 0, varying camera
        motion_time0 = rearrange(
            normalize_among_last_dims(
                rearrange(pred_motions[:, :, 0, :3, ...], "b f c h w -> b c f h w"),
                num_dims=3
            ),
            "b c f h w -> b f c h w"
        )
        motion_time0_path = os.path.join(sample_output_dir, "output_motion_time0.mp4")
        iio.mimwrite(motion_time0_path, tensor_to_video(motion_time0), macro_block_size=1)
        
        print(f"  Saved motion visualizations")

    # ============== Rendering Mode 2: Fixed timestep, moving cameras ==============
    # This shows the same moment from different camera views
    print("Rendering: Fixed timestep, moving cameras...")
    
    render_outputs_list = []
    
    for i in range(F_out):
        # Fix motion to LAST timestep
        if pred_motions is not None:
            backbone_outputs["offset"] = pred_motions[:, -1, :, :3, ...]
        
        if pred_motion_gs is not None:
            backbone_outputs.update(pred_motion_gs[-1])
        
        # Render from i-th camera
        _render_outputs = model.gs_renderer.render(
            backbone_outputs,
            input_C2W,
            input_fxfycxcy,
            input_C2W[:, i:i+1, ...],  # Target: i-th camera view
            output_fxfycxcy[:, i:i+1, ...],
        )
        render_outputs_list.append(_render_outputs)
    
    # Concatenate
    render_outputs = {}
    for k in render_outputs_list[0].keys():
        if k not in ["gaussian_usage", "voxel_ratio"]:
            render_outputs[k] = torch.cat(
                [render_outputs_list[i][k] for i in range(F_out)],
                dim=1
            )
    
    render_images_time_last = render_outputs["image"]
    
    time_last_video_path = os.path.join(sample_output_dir, "output_render_time-1.mp4")
    iio.mimwrite(time_last_video_path, tensor_to_video(render_images_time_last), macro_block_size=1)
    print(f"  Saved: {time_last_video_path}")

    # Create combined comparison video
    combined_path = os.path.join(sample_output_dir, "output_render_comparison.mp4")
    concat_videos([cam0_video_path, time_last_video_path], combined_path)
    print(f"Saved comparison: {combined_path}")

    print()
    print("=" * 60)
    print("Inference complete!")
    print("=" * 60)
    print(f"Outputs saved in: {sample_output_dir}")
    print("  - input_video.mp4: Original input frames")
    print("  - output_depth.mp4: Predicted depth maps")
    print("  - output_render_camera0.mp4: Moving time at camera 0")
    print("  - output_render_time-1.mp4: Moving cameras at final time")
    print("  - output_render_comparison.mp4: Both side-by-side")
    print("  - output_motion_*.mp4: Motion field visualizations (if dynamic)")


if __name__ == "__main__":
    main(parse_args())
