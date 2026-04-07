"""
Latent Alignment Models for aligning TinyVAE with WanVAE.

Design Goals:
1. Train TinyVAE decoder to reconstruct from WanVAE latents
2. Optionally train TinyVAE encoder to produce WanVAE-compatible latents
3. Support integration with LRDM (Latent Reconstruction Dynamic Model) pipeline

Data Format Summary:
- WanVAE (Wan2.1):
  - Input:  (B, 3, T, H, W) in [-1, 1]
  - Output: (B, 16, f, h, w) where f = 1 + (T-1)//4, h = H//8, w = W//8
  - Has built-in mean/std normalization

- TinyVAE (TAEHV):
  - Input for encode_video: (N, T, 3, H, W) in [0, 1]
  - Output latent: (N, f, 16, h, w) with time compression ~4x, space 8x
  - Uses NTCHW axis order internally for 2D conv
"""

from typing import *
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tiny_vae import TAEHV, apply_model_with_memblocks


class LatentAxisMapper:
    """
    Handles axis permutation between WanVAE and TinyVAE formats.
    
    Wan format:   (B, C, T, H, W)
    Tiny format:  (N, T, C, H, W)  for encode_video output
    """
    
    @staticmethod
    def wan_to_tiny(wan_latent: Tensor) -> Tensor:
        """(B, C, T, H, W) -> (B, T, C, H, W)"""
        return wan_latent.permute(0, 2, 1, 3, 4)
    
    @staticmethod
    def tiny_to_wan(tiny_latent: Tensor) -> Tensor:
        """(N, T, C, H, W) -> (N, C, T, H, W)"""
        return tiny_latent.permute(0, 2, 1, 3, 4)


class WanLatentDenormalizer(nn.Module):
    """
    Undoes the normalization that Wan2_1_VAEWrapper applies.
    
    Wan2_1_VAEWrapper.encode does:
      normalized = (raw - mean) * (1/std)
    
    To get raw latent back:
      raw = normalized * std + mean
    
    This is important if we want to feed Wan latents into a decoder
    that expects "native" VAE latent distribution (not standardized).
    """
    
    def __init__(self):
        super().__init__()
        # Wan2.1 statistics from wan_wrapper.py
        self.register_buffer("mean", torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]))
        self.register_buffer("std", torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]))
    
    def forward(self, normalized_latent: Tensor) -> Tensor:
        """
        Args:
            normalized_latent: (B, 16, T, H, W) from Wan2_1_VAEWrapper.encode
        Returns:
            raw_latent: (B, 16, T, H, W) in "native" VAE space
        """
        mean = self.mean.view(1, 16, 1, 1, 1)
        std = self.std.view(1, 16, 1, 1, 1)
        return normalized_latent * std + mean


class LatentAlignmentNet(nn.Module):
    """
    A lightweight alignment network to map between latent spaces.
    
    Since Wan2.1 and TinyVAE both have latent_dim=16, we use a
    residual 1x1 conv to learn any distribution mismatch.
    
    Operates on Wan-format (B, C, T, H, W) using 3D convs.
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.conv_in = nn.Conv3d(latent_dim, hidden_dim, kernel_size=1)
        self.conv_out = nn.Conv3d(hidden_dim, latent_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, hidden_dim)
        
        # Zero-init output for residual learning
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            aligned: (B, C, T, H, W)
        """
        identity = x
        h = F.gelu(self.norm(self.conv_in(x)))
        out = self.conv_out(h) + identity
        return out


class TAEHVDecoderOnly(nn.Module):
    """
    TinyVAE decoder-only version for training to reconstruct from Wan latents.
    
    The original TAEHV has:
    - Encoder: compresses RGB to latent
    - Decoder: reconstructs RGB from latent
    
    This class wraps just the decoder with proper handling.
    
    Note: The full TAEHV decoder has clamping at the input:
      Clamp() -> conv(latent->256) -> ReLU -> ...
    
    And it expects NTCHW format with specific axis order for MemBlocks.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        decoder_time_upscale: Tuple[bool, bool] = (True, True),
        decoder_space_upscale: Tuple[bool, bool, bool] = (True, True, True),
    ):
        super().__init__()
        
        # Create full TAEHV first (we'll extract decoder)
        full_taehv = TAEHV(
            checkpoint_path=None,  # Don't load yet
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
        )
        
        # Get decoder components
        # TAEHV.decoder is a Sequential starting with Clamp
        self.decoder = full_taehv.decoder
        self.frames_to_trim = full_taehv.frames_to_trim
        self.latent_channels = full_taehv.latent_channels
        
        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # Filter to decoder-only keys
            decoder_dict = {
                k.replace("decoder.", ""): v 
                for k, v in state_dict.items() 
                if k.startswith("decoder.")
            }
            if decoder_dict:
                self.decoder.load_state_dict(decoder_dict)
                print(f"[TAEHVDecoderOnly] Loaded decoder weights from {checkpoint_path}")
            else:
                # Try loading directly (maybe checkpoint already has decoder keys)
                self.decoder.load_state_dict(state_dict, strict=False)
    
    def decode_video(self, x: Tensor, parallel: bool = True, show_progress_bar: bool = False) -> Tensor:
        """
        Args:
            x: (N, T, C, H, W) latent tensor (TinyVAE format, NTCHW for 5D)
               Note: Original encode_video outputs this format.
               For Wan latent, use mapper.wan_to_tiny first.
        Returns:
            RGB: (N, T', 3, H', W') where T' = T - frames_to_trim
        """
        # apply_model_with_memblocks expects NTCHW 5D
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        # Trim initial frames (causal artifact)
        return x[:, self.frames_to_trim:]


class TAEHVEncoderOnly(nn.Module):
    """
    TinyVAE encoder-only for training to map RGB -> Wan-compatible latent.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        
        full_taehv = TAEHV(checkpoint_path=None)
        self.encoder = full_taehv.encoder
        self.latent_channels = full_taehv.latent_channels
        
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            encoder_dict = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            if encoder_dict:
                self.encoder.load_state_dict(encoder_dict)
            else:
                self.encoder.load_state_dict(state_dict, strict=False)
    
    def encode_video(self, x: Tensor, parallel: bool = True, show_progress_bar: bool = True) -> Tensor:
        """
        Args:
            x: (N, T, 3, H, W) RGB in [0, 1] (TinyVAE NTCHW 5D)
        Returns:
            latent: (N, T', 16, H', W')
        """
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)


# ============== Full Alignment Model ==============

class TinyVAEWanAligner(nn.Module):
    """
    Full model for aligning TinyVAE with WanVAE.
    
    Supports two training paths:
    
    Path A (Decoder training):
      images -> WanVAE.encode -> wan_latent -> [denorm] -> [mapper] 
                 -> TinyDecoder -> recon_images -> Loss(MSE + LPIPS)
    
    Path B (Encoder training):
      images -> TinyEncoder -> tiny_latent -> [mapper] -> [aligner] -> pred_latent
                                            |
                                            v compare with
      images -> WanVAE.encode -> wan_latent (target)
    
    And optionally:
      pred_latent -> TinyDecoder -> recon_cycle -> cycle consistency loss
    """
    
    def __init__(
        self,
        tiny_vae_ckpt: Optional[str] = None,
        use_denorm: bool = True,  # Undo Wan normalization before Tiny decoder
        use_alignment_net: bool = True,  # Use 1x1 conv residual alignment
        decoder_time_upscale: Tuple[bool, bool] = (True, True),
        decoder_space_upscale: Tuple[bool, bool, bool] = (True, True, True),
    ):
        super().__init__()
        
        self.use_denorm = use_denorm
        self.use_alignment_net = use_alignment_net
        
        # Components
        if use_denorm:
            self.denormalizer = WanLatentDenormalizer()
        else:
            self.denormalizer = None
        
        if use_alignment_net:
            self.alignment_net = LatentAlignmentNet(latent_dim=16)
        else:
            self.alignment_net = None
        
        # Axis mapper (stateless, no params)
        self.mapper = LatentAxisMapper()
        
        # TinyVAE components
        # Note: We create encoder and decoder separately to allow 
        # freezing one while training the other
        self.encoder = TAEHVEncoderOnly(checkpoint_path=tiny_vae_ckpt)
        self.decoder = TAEHVDecoderOnly(
            checkpoint_path=tiny_vae_ckpt,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
        )
    
    # ========== Path A: Wan -> Decode ==========
    
    def decode_from_wan_latent(
        self,
        wan_latent: Tensor,  # (B, 16, f, h, w) NORMALIZED from Wan wrapper
        parallel: bool = True,
    ) -> Tensor:
        """
        Reconstruct images from WanVAE latents using trained TinyVAE decoder.
        
        Pipeline:
        wan_latent (normalized) 
          -> [optional: denormalize to raw latent]
          -> [optional: alignment net]
          -> axis permute to Tiny format
          -> Tiny decoder
          -> RGB
        
        Args:
            wan_latent: (B, C, T, H, W) from Wan2_1_VAEWrapper.encode
        Returns:
            images: (N, T', 3, H', W') in [0, 1] approx (or as trained)
                    Note: T' is reduced by frames_to_trim
        """
        x = wan_latent
        
        # Optional: undo Wan's normalization
        if self.denormalizer is not None:
            x = self.denormalizer(x)
        
        # Optional: alignment network (on Wan-format tensor)
        if self.alignment_net is not None:
            x = self.alignment_net(x)
        
        # Map to Tiny format: (B, C, T, H, W) -> (B, T, C, H, W)
        x = self.mapper.wan_to_tiny(x)
        
        # Decode
        recon = self.decoder.decode_video(x, parallel=parallel)
        return recon
    
    # ========== Path B: RGB -> Tiny Encoder -> Align ==========
    
    def encode_to_wan_like(
        self,
        images: Tensor,  # (N, T, 3, H, W) in [0, 1] for Tiny
        parallel: bool = True,
    ) -> Tensor:
        """
        Encode RGB images to Wan-like latent using Tiny encoder + alignment.
        
        Returns Wan-format tensor for direct comparison or decoding.
        
        Note: There's a time dimension mismatch to be careful about:
          - Wan2.1: first frame handled specially, so 5 input frames -> 2 latent steps
          - Tiny:  causal compression, so 9 input frames -> ~2 latent steps (if 4x)
        
        The training script should handle matching input time lengths.
        """
        # Tiny encode
        tiny_latent = self.encoder.encode_video(images, parallel=parallel)
        
        # Tiny format -> Wan format
        wan_like = self.mapper.tiny_to_wan(tiny_latent)
        
        # Optional alignment
        if self.alignment_net is not None:
            wan_like = self.alignment_net(wan_like)
        
        return wan_like


# ============== Helper: Data Range Conversion ==============

def images_wan_to_tiny(images_wan: Tensor) -> Tensor:
    """
    Convert images from WanVAE format to TinyVAE format.
    
    WanVAE input:   (B, 3, T, H, W) in [-1, 1]
    TinyVAE input:  (N, T, 3, H, W) in [0, 1]
    
    Also handles range: [-1,1] -> [0,1]
    """
    # Range
    images_01 = (images_wan + 1) / 2.0
    
    # Axis: (B, 3, T, H, W) -> (B, T, 3, H, W)
    return images_01.permute(0, 2, 1, 3, 4)


def images_tiny_to_wan(images_tiny: Tensor) -> Tensor:
    """
    Convert images from TinyVAE format to WanVAE format.
    
    TinyVAE input:  (N, T, 3, H, W) in [0, 1]  
    WanVAE input:   (B, 3, T, H, W) in [-1, 1]
    """
    # Axis: (N, T, 3, H, W) -> (N, 3, T, H, W)
    images_axis = images_tiny.permute(0, 2, 1, 3, 4)
    
    # Range: [0,1] -> [-1,1]
    return images_axis * 2.0 - 1.0


# ============== Time Alignment Helpers ==============

def match_latent_time(
    wan_latent: Tensor, 
    tiny_latent: Tensor,
    wan_input_frames: int,
    tiny_input_frames: int,
) -> Tuple[Tensor, Tensor]:
    """
    (Experimental) Try to align latent time dimensions.
    
    Wan2.1 pattern for T input frames:
      latent_T = 1 + (T - 1) // 4
      E.g., T=1 -> 1, T=2->1, T=3->1, T=4->1, T=5->2, T=6->2, ...
    
    TinyVAE: Uses MemBlock + 2x TPool twice = causal 4x compression.
      The relation may be: latent_T = (input_T + offset) // 4
    
    For training, it's easier to use input lengths that result in same
    latent time dimension, or only supervise on overlapping indices.
    """
    wan_t = wan_latent.shape[2]
    tiny_t = tiny_latent.shape[2]  # after tiny_to_wan
    
    if wan_t == tiny_t:
        return wan_latent, tiny_latent
    
    # Simple: crop/pad to match
    min_t = min(wan_t, tiny_t)
    return wan_latent[:, :, :min_t], tiny_latent[:, :, :min_t]


import os  # for the path checks above
