from typing import *
from torch import Tensor
from src.utils import StepTracker
import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as tF
from peft import LoraConfig, inject_adapter_in_model
from einops import rearrange
from pytorch_msssim import ssim as SSIM
from lpips import LPIPS

from src.options import Options
from src.models.networks import (
    FeatureEmbed,
    WanTextEncoderWrapper,
    Wan2_2_VAEWrapper,
    WanTI2VDiffusionWrapper,
)
from src.utils import convert_to_buffer, plucker_ray


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Wan(nn.Module):
    def __init__(self, opt: Options, step_tracker: Optional[StepTracker] = None):
        super().__init__()

        self.opt = opt
        self.step_tracker = step_tracker

        # VAE
        if opt.load_vae_in_wan:
            self.vae = Wan2_2_VAEWrapper(opt.vae_path)
            convert_to_buffer(self.vae, persistent=False)  # no gradient & not save to checkpoint
        else:
            self.vae = None

        # Text encoder
        if opt.load_text_encoder:
            self.text_encoder = WanTextEncoderWrapper(opt.wan_dir)
            convert_to_buffer(self.text_encoder, persistent=False)  # no gradient & not save to checkpoint
        else:
            self.text_encoder = None

        # Diffusion model
        self.diffusion = WanTI2VDiffusionWrapper(
            opt.wan_dir,
            opt.plucker_embed_dim if opt.input_plucker else 0,  # extra_in_dim
            opt.num_train_timesteps,
            opt.num_inference_steps,
            opt.shift,
            opt.sigma_min,
            opt.extra_one_step,
        )

        # (Optional) Plucker embeddings
        if opt.input_plucker:
            self.plucker_embed = FeatureEmbed(
                "causal3d",
                input_channels=6,
                out_channels=opt.plucker_embed_dim,
                t_ratio=opt.compression_ratio[0],
                s_ratio=opt.compression_ratio[1],
            )
            self.plucker_ln = nn.GroupNorm(
                num_groups=1,
                num_channels=opt.plucker_embed_dim,
            )

        # Add LoRA in the diffusion model, will freeze all parameters except LoRA layers
        if opt.use_lora_in_diffusion_model:
            self._add_lora_to_diffusion_model(
                target_modules=opt.lora_target_modules.split(","),
                lora_rank=opt.lora_rank,
            )

        # Set other trainable parameters except LoRA layers in the diffusion model
        if opt.more_trainable_diffusion_params is not None:
            trainble_names = opt.more_trainable_diffusion_params.split(",")
            if opt.use_lora_in_diffusion_model:
                trainble_names.append("lora")
            if opt.input_plucker:
                trainble_names.append("plucker")
            for name, param in self.diffusion.model.named_parameters():
                _flag = False
                for trainble_name in trainble_names:
                    if trainble_name in name:
                        param.requires_grad_(True)
                        _flag = True
                        break
                if not _flag:
                    param.requires_grad_(False)

        # LPIPS for evaluation
        if self.opt.use_lpips:
            self.lpips_loss = LPIPS(net="vgg")
            convert_to_buffer(self.lpips_loss, persistent=False)  # no gradient & not save to checkpoint
        else:
            self.lpips_loss = None

        # Pre-compute negative prompt embedding
        self.register_buffer(
            "negative_prompt_embed",
            torch.from_numpy(np.load(self.opt.negative_prompt_embed_path)),
        )  # (1, N, D')

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Any], dtype: torch.dtype = torch.float32):
        assert self.vae is not None, "VAE is not loaded in Wan"
        outputs = {}

        images = data["image"].to(dtype)  # (B, F, 3, H, W)
        B, device = images.shape[0], images.device

        # Text encoder
        if self.text_encoder is not None:
            prompts = data["prompt"]  # a list of strings
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                self.text_encoder.eval()
                prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
                negative_prompt_embeds = self.text_encoder([self.opt.negative_prompt]).repeat(B, 1, 1)  # (B, N=512, D')
        else:
            prompt_embeds = data["prompt_embed"].to(dtype)  # (B, N=512, D')
            negative_prompt_embeds = self.negative_prompt_embed.to(device=device, dtype=dtype).repeat(B, 1, 1)  # (B, N=512, D')

        # VAE
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            self.vae.eval()
            latents = self.encode(images * 2. - 1.)  # (B, D, f, h, w)
            cond_latents = latents[:, :, 0:1, :, :].clone()  # (B, D, 1, h, w)

        # (Optional) Plucker embeddings
        if self.opt.input_plucker:
            C2W = data["metric_C2W"].to(dtype)  # (B, F, 4, 4)
            fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, F, 4)
            plucker, _ = plucker_ray(images.shape[-2], images.shape[-1], C2W, fxfycxcy)  # (B, F, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w")
            plucker_embeds = self.plucker_ln(self.plucker_embed(plucker))  # (B, D, f, h, w)
        else:
            plucker_embeds = None

        # Classifier-free guidance dropout
        if self.training:
            masks = (torch.rand(B, device=device) < self.opt.cfg_dropout).to(dtype)
            prompt_embeds = prompt_embeds * masks[:, None, None] + negative_prompt_embeds * (1 - masks)[:, None, None]
            plucker_embeds = plucker_embeds * masks[:, None, None, None, None]

        # Diffusion
        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)
        noises = torch.randn_like(latents)
        min_t, max_t = int(self.opt.min_timestep_boundary * self.opt.num_train_timesteps), \
            int(self.opt.max_timestep_boundary * self.opt.num_train_timesteps)
        timesteps_id = torch.randint(min_t, max_t, (1,))  # batch share the same timestep for simpler scheduler api
        timesteps = self.diffusion.scheduler.timesteps[timesteps_id].to(dtype=dtype, device=device)
        noisy_latents = self.diffusion.scheduler.add_noise(latents, noises, timesteps)
        targets = self.diffusion.scheduler.training_target(latents, noises, timesteps)

        model_outputs = self.diffusion(
            noisy_latents,
            timesteps,
            prompt_embeds,
            cond_latents,
            plucker_embeds,
        )
        loss = tF.mse_loss(model_outputs.float(), targets.float())
        outputs["loss"] = self.diffusion.scheduler.training_weight(timesteps) * loss

        # For visualizaiton
        pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_latents, timesteps)
        outputs["pred_x0"] = pred_x0  # decode outside this function

        return outputs

    ################################ Helper functions ################################

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def evaluate(self, data: Dict[str, Any], verbose: bool = True):
        outputs = {}

        images = data["image"]  # (B, F, 3, H, W)
        B, F, _, H, W = images.shape
        f = 1 + (F - 1) // self.opt.compression_ratio[0]
        h = H // self.opt.compression_ratio[1]
        w = W // self.opt.compression_ratio[2]

        # Text encoder
        if self.text_encoder is not None:
            prompts = data["prompt"]  # a list of strings
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
            negative_prompt_embeds = self.text_encoder([self.opt.negative_prompt]).repeat(B, 1, 1)  # (B, N=512, D')
        else:
            prompt_embeds = data["prompt_embed"]  # (B, N=512, D')
            negative_prompt_embeds = self.negative_prompt_embed.repeat(B, 1, 1)  # (B, N=512, D')
        if self.opt.cfg_scale > 1.:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # VAE
        self.vae.eval()
        cond_latents = self.vae.encode(rearrange(images[:, 0:1, ...] * 2. - 1., "b f c h w -> b c f h w"))  # (B, D, 1, h, w)

        # Diffusion
        self.diffusion.eval()
        self.diffusion.scheduler.set_timesteps(self.opt.num_inference_steps, training=False)
        latents = torch.randn(B, self.opt.latent_dim, f, h, w, device=images.device, dtype=images.dtype)

        # (Optional) Plucker embeddings
        if self.opt.input_plucker:
            C2W = data["metric_C2W"]  # (B, F, 4, 4)
            fxfycxcy = data["fxfycxcy"]  # (B, F, 4)
            plucker, _ = plucker_ray(images.shape[-2], images.shape[-1], C2W, fxfycxcy)  # (B, F, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w")
            plucker_embeds = self.plucker_ln(self.plucker_embed(plucker))  # (B, D, f, h, w)
            if self.opt.cfg_scale > 1.:
                plucker_embeds = torch.cat([torch.zeros_like(plucker_embeds), plucker_embeds], dim=0)
        else:
            plucker_embeds = None

        for i, timestep in tqdm(enumerate(self.diffusion.scheduler.timesteps),
            total=len(self.diffusion.scheduler.timesteps), ncols=125, disable=not verbose):
            timesteps = timestep.unsqueeze(0).repeat(B).to(dtype=images.dtype, device=images.device)
            if self.opt.cfg_scale > 1.:
                timesteps = torch.cat([timesteps, timesteps], dim=0)
                cond_latents = torch.cat([cond_latents, cond_latents], dim=0)
                latents = torch.cat([latents, latents], dim=0)

            model_outputs = self.diffusion(
                latents,
                timesteps,
                prompt_embeds,
                cond_latents,
                plucker_embeds,
            )
            if self.opt.cfg_scale > 1.:
                model_outputs_uncond, model_outputs_cond = model_outputs.chunk(2)
                model_outputs = model_outputs_uncond + self.opt.cfg_scale * (model_outputs_cond - model_outputs_uncond)
                latents, cond_latents = latents.chunk(2)[1], cond_latents.chunk(2)[1]

            latents = self.diffusion.scheduler.step(model_outputs, self.diffusion.scheduler.timesteps[i], latents)

        # Decode
        latents[:, :, 0:1, ...] = cond_latents
        pred_images = (self.decode_latent(latents).clamp(-1., 1.) + 1.) / 2.  # (B, D, f, h, w) -> (B, F, 3, H, W)

        # For visualization
        outputs["images_gt"] = images
        outputs["images_pred"] = pred_images

        # Evaluation metrics: PSNR, SSIM, LPIPS
        outputs["psnr"] = -10. * torch.log10(torch.mean((images - pred_images) ** 2, dim=(1, 2, 3, 4)))  # (B,)
        outputs["ssim"] = SSIM(
            rearrange(pred_images, "b f c h w -> (b f) c h w"),
            rearrange(images, "b f c h w -> (b f) c h w"),
            data_range=1., size_average=False,
        )  # (B*F,)
        outputs["ssim"] = rearrange(outputs["ssim"], "(b f) -> b f", b=B).mean(dim=1)  # (B,)
        if self.lpips_loss is not None:
            outputs["lpips"] = self.lpips_loss(
                rearrange(pred_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                rearrange(images, "b f c h w -> (b f) c h w") * 2. - 1.,
            )  # (B*F, 1, 1, 1)
            outputs["lpips"] = rearrange(outputs["lpips"], "(b f) c h w -> b f c h w", b=B).mean(dim=(1, 2, 3, 4))  # (B,)

        return outputs

    @torch.no_grad()
    def encode(self, images: Tensor):
        """ Image to VAE latent.

        Inputs:
            - `images`: (B, F, 3, H, W)

        Outputs:
            - `latents`: (B, D, f, h, w)
        """
        assert self.vae is not None, "VAE is not loaded in VGGT-VAE"
        self.vae.eval()
        images = rearrange(images, "b f c h w -> b c f h w")  # (B, F, 3, H, W) -> (B, 3, F, H, W)
        return self.vae.encode(images).to(images.dtype)  # (B, D, f, h, w)

    @torch.no_grad()
    def decode_latent(self, latents: Tensor):
        """ VAE latent to Image.

        Inputs:
            - `latents`: (B, D, f, h, w)

        Outputs:
            - `images`: (B, F, 3, H, W)
        """
        assert self.vae is not None, "VAE is not loaded in VGGT-VAE"
        self.vae.eval()
        images = self.vae.decode(latents)  # (B, D, f, h, w) -> (B, 3, F, H, W)
        return rearrange(images, "b c f h w -> b f c h w").to(latents.dtype)  # (B, F, 3, H, W)

    def _add_lora_to_diffusion_model(self, target_modules: List[str], lora_rank: int, lora_alpha: Optional[int] = None):
        if lora_alpha is None:
            lora_alpha = lora_rank

        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        self.diffusion.model = inject_adapter_in_model(lora_config, self.diffusion.model)
