# Modified from https://github.com/guandeh17/Self-Forcing/blob/main/utils/wan_wrapper.py

from typing import *
from torch import Tensor

import os
import torch
from torch import nn

from .wan_modules.t5 import umt5_xxl
from .wan_modules.tokenizers import HuggingfaceTokenizer
from .wan_modules.vae2_1 import _video_vae as _video_vae2_1
from .wan_modules.vae2_2 import _video_vae as _video_vae2_2
from .wan_modules.model import WanModel
from .scheduler import FlowMatchScheduler


class WanTextEncoderWrapper(nn.Module):
    def __init__(self, pretrained_dir: str):
        super().__init__()

        self.model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu",
        ).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(
            os.path.join(pretrained_dir, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", weights_only=False))

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(pretrained_dir, "google/umt5-xxl"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]):
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.model(ids, mask)

        for u, v in zip(prompt_embeds, seq_lens):
            u[v:] = 0.  # set padding to 0.
        return prompt_embeds  # (B, N=512, D)


class Wan2_1_VAEWrapper(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()

        self.register_buffer("mean", torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]))
        self.register_buffer("std", torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]))

        self.model = _video_vae2_1(
            pretrained_path=pretrained_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode(self, videos: Tensor):
        return torch.stack([
            self.model.encode(u.unsqueeze(0), [self.mean, 1./self.std]).float().squeeze(0)
            for u in videos
        ], dim=0)  # (B, D, f, h, w)

    def decode(self, latents: Tensor):
        return torch.stack([
            self.model.decode(latent.unsqueeze(0), [self.mean, 1./self.std]).float().clamp(-1., 1.).squeeze(0)
            for latent in latents
        ], dim=0)  # (B, 3, F, H, W)


class Wan2_2_VAEWrapper(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()

        self.register_buffer("mean", torch.tensor([
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
            0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ]))
        self.register_buffer("std", torch.tensor([
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ]))

        self.model = _video_vae2_2(
            pretrained_path=pretrained_path,
            z_dim=48,
            dim=160,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
        ).eval().requires_grad_(False)

    def encode(self, videos: Tensor):
        return torch.stack([
            self.model.encode(u.unsqueeze(0), [self.mean, 1./self.std]).float().squeeze(0)
            for u in videos
        ], dim=0)  # (B, D, f, h, w)

    def decode(self, latents: Tensor):
        return torch.stack([
            self.model.decode(latent.unsqueeze(0), [self.mean, 1./self.std]).float().clamp(-1., 1.).squeeze(0)
            for latent in latents
        ], dim=0)  # (B, 3, F, H, W)


# class WanI2VDiffusionWrapper(nn.Module):
#     def __init__(self,
#         pretrained_dir: str,
#         extra_in_dim: int,
#         num_train_timesteps: int = 1000,
#         num_inference_steps: int = 50,
#         boundary: float = 0.9,
#         shift: float = 5.,
#         sigma_min: float = 0.,
#         extra_one_step: bool = True,
#         use_gradient_checkpointing: bool = True,
#         use_gradient_checkpointing_offload: bool = False,
#     ):
#         super().__init__()

#         self.low_noise_model = WanModel.from_pretrained(pretrained_dir, subfolder="low_noise_model")
#         self.low_noise_model.use_gradient_checkpointing = use_gradient_checkpointing
#         self.low_noise_model.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

#         self.boundary = boundary

#         self.high_noise_model = WanModel.from_pretrained(pretrained_dir, subfolder="high_noise_model")
#         self.high_noise_model.use_gradient_checkpointing = use_gradient_checkpointing
#         self.high_noise_model.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

#         # Handle extra inputs
#         if extra_in_dim > 0:
#             new_conv = nn.Conv3d(
#                 extra_in_dim + self.low_noise_model.in_dim,
#                 self.low_noise_model.dim,
#                 kernel_size=self.low_noise_model.patch_size,
#                 stride=self.low_noise_model.patch_size,
#             )
#             new_conv.weight.data[:, :-self.low_noise_model.in_dim, ...] = 0.
#             new_conv.weight.data[:, -self.low_noise_model.in_dim:, ...] = self.low_noise_model.patch_embedding.weight.data
#             if self.low_noise_model.patch_embedding.bias.data is not None:
#                 new_conv.bias.data = self.low_noise_model.patch_embedding.bias.data
#             self.low_noise_model.patch_embedding = new_conv

#         self.scheduler = FlowMatchScheduler(
#             num_train_timesteps=num_train_timesteps,
#             num_inference_steps=num_inference_steps,
#             shift=shift,
#             sigma_min=sigma_min,
#             extra_one_step=extra_one_step,
#         )
#         self.scheduler.set_timesteps(num_train_timesteps, training=True)

#         # self.max_seq_len = 75600  # 81 x 720 x 1280 -> 21 x 45 x 80
#         # self.max_seq_len = 32760  # 81 x 480 x 832 -> 21 x 30 x 52
#         self.max_seq_len = 11440  # 49 x 352 x 640 -> 13 x 22 x 40

#     def forward(self,
#         noisy_latents: Tensor,  # (B, D, f, h, w)
#         timesteps: Tensor,  # (1,)
#         prompt_embeds: Tensor,  # (B, N, D')
#         cond_latents: Optional[Tensor] = None,  # (B, D, f, h, w)
#         plucker_embeds: Optional[Tensor] = None,  # (B, D, f, h, w)
#     ):
#         (B, _, f, h, w), device = noisy_latents.shape, noisy_latents.device
#         F = 1 + (f - 1) * 4  # `4`: hard-coded for Wan2.2-I2V-A14B

#         # Image condition
#         masks = torch.ones(B, 1, F, h, w, device=device)
#         masks[:, :, 1:] = 0.
#         masks = torch.cat([torch.repeat_interleave(masks[:, :, 0:1, ...], repeats=4, dim=2), masks[:, :, 1:, ...]], dim=2)
#         masks = masks.view(B, 1, masks.shape[2]//4, 4, h, w)
#         masks = masks.transpose(2, 3)[:, 0, ...]  # (B, 4, f, h, w)
#         cond_latents = torch.cat([masks, cond_latents], dim=1)  # (B, D+4, f, h, w)

#         # (Optional) Concatenate plucker embeds
#         if plucker_embeds is not None:
#             noisy_latents = torch.cat([plucker_embeds, noisy_latents], dim=1)

#         # Choice models based on timesteps
#         if timesteps.item() >= self.boundary * self.scheduler.num_train_timesteps:
#             model = self.high_noise_model
#         else:
#             model = self.low_noise_model

#         model_outputs = torch.stack(model(
#             [noisy_latent for noisy_latent in noisy_latents],
#             timesteps,
#             [prompt_embed for prompt_embed in prompt_embeds],
#             self.max_seq_len,
#         ), dim=0)  # (B, D, f, h, w)
#         return model_outputs

#     def _convert_flow_pred_to_x0(self, flow_pred: Tensor, xt: Tensor, timestep: Tensor) -> Tensor:
#         """
#         Convert flow matching's prediction to x0 prediction.
#         flow_pred: the prediction with shape [B, C, H, W]
#         xt: the input noisy data with shape [B, C, H, W]
#         timestep: the timestep with shape [B]

#         pred = noise - x0
#         x_t = (1-sigma_t) * x0 + sigma_t * noise
#         we have x0 = x_t - sigma_t * pred
#         see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
#         """
#         # use higher precision for calculations
#         original_dtype = flow_pred.dtype
#         flow_pred, xt, sigmas, timesteps = map(
#             lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
#                                                         self.scheduler.sigmas,
#                                                         self.scheduler.timesteps]
#         )

#         timestep_id = torch.argmin(
#             (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
#         sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
#         x0_pred = xt - sigma_t * flow_pred
#         return x0_pred.to(original_dtype)

#     def _convert_x0_to_flow_pred(self, x0_pred: Tensor, xt: Tensor, timestep: Tensor) -> Tensor:
#         """
#         Convert x0 prediction to flow matching's prediction.
#         x0_pred: the x0 prediction with shape [B, C, H, W]
#         xt: the input noisy data with shape [B, C, H, W]
#         timestep: the timestep with shape [B]

#         pred = (x_t - x_0) / sigma_t
#         """
#         # use higher precision for calculations
#         original_dtype = x0_pred.dtype
#         x0_pred, xt, sigmas, timesteps = map(
#             lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
#                                                       self.scheduler.sigmas,
#                                                       self.scheduler.timesteps]
#         )
#         timestep_id = torch.argmin(
#             (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
#         sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
#         flow_pred = (xt - x0_pred) / sigma_t
#         return flow_pred.to(original_dtype)


class WanTI2VDiffusionWrapper(nn.Module):
    def __init__(self,
        pretrained_dir: str,
        extra_in_dim: int,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        shift: float = 5.,
        sigma_min: float = 0.,
        extra_one_step: bool = True,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        super().__init__()

        self.model = WanModel.from_pretrained(pretrained_dir)
        self.model.use_gradient_checkpointing = use_gradient_checkpointing
        self.model.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # Handle extra inputs
        if extra_in_dim > 0:
            new_conv = nn.Conv3d(
                extra_in_dim + self.model.in_dim,
                self.model.dim,
                kernel_size=self.model.patch_size,
                stride=self.model.patch_size,
            )
            new_conv.weight.data[:, :self.model.in_dim, ...] = self.model.patch_embedding.weight.data
            new_conv.weight.data[:, self.model.in_dim:, ...] = 0.
            if self.model.patch_embedding.bias.data is not None:
                new_conv.bias.data = self.model.patch_embedding.bias.data
            self.model.patch_embedding = new_conv

        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
        )
        self.scheduler.set_timesteps(num_train_timesteps, training=True)

        # self.max_seq_len = 27280  # 121 x 704 x 1280 -> 31 x 22 x 40
        self.max_seq_len = 2860  # 49 x 352 x 640 -> 13 x 11 x 20

    def forward(self,
        noisy_latents: Tensor,  # (B, D, f, h, w)
        timesteps: Tensor,  # (1,)
        prompt_embeds: Tensor,  # (B, N, D')
        cond_latents: Optional[Tensor] = None,  # (B, D, 1, h, w)
        plucker_embeds: Optional[Tensor] = None,  # (B, D, f, h, w)
    ):
        cond_masks = torch.ones_like(noisy_latents)

        # Use clean latents for the first frame
        if cond_latents is not None:
            cond_masks[:, :, 0] = 0.
            noisy_latents = (1. - cond_masks) * cond_latents + cond_masks * noisy_latents  # (B, D, f, h, w)

        # (Optional) Concatenate plucker embeds
        if plucker_embeds is not None:
            noisy_latents = torch.cat([noisy_latents, plucker_embeds], dim=1)

        # Set timesteps as zero (clean) for the first frame
        temp_ts = (cond_masks[:, 0, :, ::2, ::2] * timesteps[:, None, None, None]).flatten(1)  # (B, f, hh, ww) -> (B, f * hh * ww)
        temp_ts = torch.cat([
            temp_ts,
            temp_ts.new_ones((temp_ts.shape[0], self.max_seq_len - temp_ts.shape[1])) * timesteps[:, None],
        ], dim=1)  # (B, `self.max_seq_len`)
        timesteps = temp_ts

        model_outputs = torch.stack(self.model(
            [noisy_latent for noisy_latent in noisy_latents],
            timesteps,
            [prompt_embed for prompt_embed in prompt_embeds],
            self.max_seq_len,
        ), dim=0)  # (B, D, f, h, w)
        return model_outputs

    def _convert_flow_pred_to_x0(self, flow_pred: Tensor, xt: Tensor, timestep: Tensor) -> Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _convert_x0_to_flow_pred(self, x0_pred: Tensor, xt: Tensor, timestep: Tensor) -> Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      self.scheduler.sigmas,
                                                      self.scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)
