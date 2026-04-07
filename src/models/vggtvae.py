from typing import *
from torch import Tensor
from src.utils import StepTracker

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as tF
from einops import rearrange, repeat

from src.options import Options
from src.models.networks.conv import FeatureEmbed
from src.models.networks.scheduler import FlowMatchScheduler
from src.models.networks.wan_wrapper import Wan2_2_VAEWrapper
from src.models.networks.attention import Attention, MemEffAttention

from src.models.networks.gs_aggregator import GSAggregator
from src.models.networks.gs_block import MyBlock, TimestepEmbedder, modulate
from src.utils import to_tuple, convert_to_buffer, colorize_depth, plucker_ray, zero_init_module, inverse_c2w, fxfycxcy_to_intrinsics

import sys; sys.path.append("extensions/vggt")
from extensions.vggt.vggt.layers import PatchEmbed
from extensions.vggt.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from extensions.vggt.vggt.models.vggt import VGGT
from extensions.vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding


class VGGTVAE(nn.Module):
    def __init__(self, opt: Options, step_tracker: Optional[StepTracker] = None):
        super().__init__()

        self.opt = opt
        self.step_tracker = step_tracker

        vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        self.register_buffer("_resnet_mean", vggt.aggregator._resnet_mean.squeeze(1), persistent=False)
        self.register_buffer("_resnet_std", vggt.aggregator._resnet_std.squeeze(1), persistent=False)

        # VGGT DINOv2
        if opt.load_vggt_dino:
            self.dino = vggt.aggregator.patch_embed
        else:
            self.dino = None

        # VAE
        if opt.load_vae_in_vggtvae:
            self.vae = Wan2_2_VAEWrapper(opt.vae_path)
        else:
            self.vae = None

        # Latent embedding
        res = to_tuple(opt.input_res)
        self.latent_patch_embed = PatchEmbed(
            img_size=(res[0]//self.opt.compression_ratio[0], res[1]//self.opt.compression_ratio[1]),
            patch_size=opt.latent_patch_size,
            in_chans=48,  # hard-coded for Wan2.2-TI2V-5B
            embed_dim=vggt.aggregator.patch_embed.embed_dim,
        )
        self.latent_rope = RotaryPositionEmbedding2D(frequency=100)  # TODO: make it configurable
        self.latent_position_getter = PositionGetter()
        self.latent_blocks = nn.ModuleList(
            [
                MyBlock(
                    dim=vggt.aggregator.patch_embed.embed_dim,
                    num_heads=16,  # TODO: make it configurable
                    rope=self.latent_rope,
                    attn_class=MemEffAttention if opt.memory_efficient_attention else Attention,
                    is_dit_block=opt.is_dit_block,
                )
                for _ in range(opt.num_latent_blocks)
            ]
        )
        self.latent_ln = nn.LayerNorm(vggt.aggregator.patch_embed.embed_dim)
        if opt.is_dit_block:
            self.t_embedder = TimestepEmbedder(vggt.aggregator.patch_embed.embed_dim)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(vggt.aggregator.patch_embed.embed_dim, 2 * vggt.aggregator.patch_embed.embed_dim, bias=True),
            )
            torch.nn.init.zeros_(self.adaLN_modulation[-1].weight)
            torch.nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # (Optional) Camera embedding
        if opt.input_camera_info:
            self.plucker_embed = FeatureEmbed(
                "causal3d",
                6,  # hard-coded for plucker embedding
                vggt.aggregator.patch_embed.embed_dim,
                t_ratio=opt.compression_ratio[0],
                s_ratio=opt.compression_ratio[1] * opt.latent_patch_size,
            )
            self.plucker_ln = nn.GroupNorm(
                num_groups=1,
                num_channels=vggt.aggregator.patch_embed.embed_dim,
            )
            zero_init_module(self.plucker_embed)
            zero_init_module(self.plucker_ln)  # zero init norm weights

        # Backbone
        self.backbone = GSAggregator(
            patch_embed="dummy",  # seperate DINO from `GSAggregator`
            extra_dim=9 if opt.input_camera_info else 0,  # `9`: (3: R, 4: t, 2:fxfy)
            memory_efficient_attention=opt.memory_efficient_attention,
            block_fn=MyBlock,
            is_dit_block=opt.is_dit_block,
            refine_inputs=opt.refine_inputs,
            refine_freq=opt.refine_freq,
            refine_crossattn=opt.refine_crossattn,
        )
        self.backbone.load_state_dict(vggt.aggregator.state_dict(), strict=False)

        # Pointmap, depth and camera heads
        self.point_head = vggt.point_head
        self.depth_head = vggt.depth_head
        self.camera_head = vggt.camera_head

        # Noise scheduler
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=opt.num_train_timesteps,
            num_inference_steps=opt.num_inference_steps,
            shift=opt.shift,
            sigma_min=opt.sigma_min,
            extra_one_step=opt.extra_one_step,
        )
        self.scheduler.set_timesteps(opt.num_train_timesteps, training=True)

        # VGGT as teacher
        if not opt.load_vggt:
            del vggt
            self.vggt = None
        else:
            self.vggt = vggt
            self.vggt.point_head = None
            self.vggt.depth_head = None
            self.vggt.camera_head = None
            self.vggt.track_head = None

        # Handle not used parameters: no gradient & not save to checkpoint
        if self.vae is not None:
            convert_to_buffer(self.vae, persistent=False)
        if self.dino is not None:
            convert_to_buffer(self.dino, persistent=False)
        if self.vggt is not None:
            convert_to_buffer(self.vggt, persistent=False)
        if opt.buffer_vggt_backbone:
            convert_to_buffer(self.backbone, persistent=False)
        convert_to_buffer(self.point_head, persistent=False)
        convert_to_buffer(self.depth_head, persistent=False)
        convert_to_buffer(self.camera_head, persistent=False)

        if opt.freeze_vggt_backbone and not opt.buffer_vggt_backbone:
            self.backbone.requires_grad_(False)
            for name, param in self.backbone.named_parameters():
                if "refine_blocks" in name or "adaLN_modulation" in name:
                    param.requires_grad_(True)

    def forward(self, *args, func_name="compute_loss_dinodistill", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss_denoise(self, data: Dict[str, Any], dtype: torch.dtype = torch.float32):
        assert self.vae is not None, "VAE is not loaded in VGGT-VAE"
        assert self.vggt is not None, "VGGT is not loaded in VGGT-VAE"
        outputs = {}

        images = data["image"].to(dtype)  # (B, F, 3, H, W)
        depths = data["depth"].to(dtype)  # (B, F, H, W)
        metric_C2W = data["metric_C2W"].to(dtype)  # (B, F, 4, 4)
        fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, F, 4)

        F_in, (H, W), device = images.shape[1], images.shape[-2:], images.device
        _downsample = self.opt.compression_ratio[1] * self.opt.latent_patch_size
        H_14, W_14 = H // _downsample * 14, W // _downsample * 14  # `14`: hard-coded for DINOv2

        # VGGT inputs
        F_recon = 1 + (F_in - 1) // self.opt.compression_ratio[0]  # not reconstruct (upsample) on the frame dimension
        out_idxs = torch.arange(0, F_in, math.ceil(F_in/F_recon), dtype=torch.long, device=images.device)  # (F_recon,)
        assert len(out_idxs) == F_recon
            ## Actual images for VGGT
        recon_images = images[:, out_idxs, ...]  # (B, F_recon, 3, H, W)
        recon_images_14 = tF.interpolate(
            rearrange(recon_images, "b f c h w -> (b f) c h w"),
            size=(H_14, W_14),
            mode="bilinear",
        )  # (B*F_recon, 3, H_14, W_14)
        recon_images_14 = rearrange(recon_images_14, "(b f) c h w -> b f c h w", f=F_recon)  # (B, F_recon, 3, H_14, W_14)
            ## Actual depths for VGGT
        recon_depths = depths[:, out_idxs, ...]  # (B, F_recon, H, W)
        recon_depths_14 = tF.interpolate(
            rearrange(recon_depths, "b f h w -> (b f) h w").unsqueeze(1),
            size=(H_14, W_14),
            mode="nearest-exact",
        ).squeeze(1)  # (B*F_recon, H_14, W_14)
        recon_depths_14 = rearrange(recon_depths_14, "(b f) h w -> b f h w", f=F_recon)  # (B, F_recon, H_14, W_14)

        # (Optional) VGGT on condition images
        if self.opt.refine_crossattn:
            with torch.no_grad():
                cond_vggt_aggregated_tokens_list, cond_vggt_patch_start_idx = self.vggt.aggregator(recon_images_14[:, 0:1, ...])
                cond_vggt_patch_tokens = cond_vggt_aggregated_tokens_list[-1][:, 0, cond_vggt_patch_start_idx:]  # (B, N, D)
        else:
            cond_vggt_patch_tokens = None

        # (Optional) Plucker embedding
        if self.opt.input_camera_info:
            plucker, _ = plucker_ray(H, W, metric_C2W, fxfycxcy)  # (B, F, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w")
            plucker_embeds = self.plucker_ln(self.plucker_embed(plucker))  # (B, D, F, h, w)
            plucker_embeds = rearrange(plucker_embeds, "b d f h w -> (b f) (h w) d")  # (B*F, h*w, D)
        else:
            plucker_embeds = 0.

        # (Optional) Prepare pose information
        if self.opt.input_camera_info:
            metric_W2C = inverse_c2w(metric_C2W[:, out_idxs, ...])  # (B, F_recon, 4, 4)
            intrinsics = fxfycxcy_to_intrinsics(fxfycxcy[:, out_idxs, ...])  # (B, F_recon, 3, 3)
            extra_info = extri_intri_to_pose_encoding(metric_W2C, intrinsics, (1, 1)).to(metric_C2W.dtype)  # (B, F_recon, 9)

        # VAE
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            self.vae.eval()
            latents = self.encode(images * 2. - 1.)  # (B, D, f, h, w)
            cond_latents = latents[:, :, 0:1, :, :].clone()  # (B, D, 1, h, w)

        # Diffusion
        self.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)
        noises = torch.randn_like(latents)
        min_t, max_t = int(self.opt.min_timestep_boundary * self.opt.num_train_timesteps), \
            int(self.opt.max_timestep_boundary * self.opt.num_train_timesteps)
        timesteps_id = torch.randint(min_t, max_t, (1,))  # batch share the same timestep for simpler scheduler api
        timesteps = self.scheduler.timesteps[timesteps_id].to(dtype=dtype, device=device)
        noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)
        noisy_latents[:, :, 0:1, :, :] = cond_latents  # replace the first frame with the condition frame

        latent_embeds, t_embeds = self.latent_embed(noisy_latents, timesteps.repeat(noisy_latents.shape[0]))
        latent_embeds = latent_embeds + plucker_embeds

        model_outputs = self.decode_embed(
            latent_embeds,
            recon_images_14,
            extra_info=extra_info,
            cross_patch_tokens=cond_vggt_patch_tokens,
            t_embeds=t_embeds,
            only_depth=True,
            frames_chunk_size=self.opt.frames_chunk_size,
        )
        aggregated_tokens_list = model_outputs["aggregated_tokens_list"]  # a list of (B, f, N=hh*ww, 2D)
        patch_start_idx = model_outputs["patch_start_idx"]
        pred_depths = model_outputs["depth"]  # (B, F_recon, H_14, W_14)

        # For visualization
        outputs["images"] = recon_images_14
        outputs["images_noisy"] = self.scheduler.add_noise(recon_images_14, torch.randn_like(recon_images_14), timesteps)  # approximate noisy level
        outputs["images_depth_gt"] = colorize_depth(1. / recon_depths_14, batch_mode=True)  # disparity for visualization
        outputs["images_depth"] = colorize_depth(1. / pred_depths, batch_mode=True)  # disparity for visualization

        ################################ Losses and metrics ################################

        with torch.no_grad():
            vggt_aggregated_tokens_list, vggt_patch_start_idx = self.vggt.aggregator(recon_images_14)

        loss = 0.
        # for i in range(len(aggregated_tokens_list)):
        #     loss += tF.mse_loss(
        #         aggregated_tokens_list[i][:, :, patch_start_idx:, ...],
        #         vggt_aggregated_tokens_list[i][:, :, vggt_patch_start_idx:, ...],
        #         reduction="none"
        #     ).mean(dim=(1, 2, 3))  # (B,)
        # outputs["loss"] = loss = self.scheduler.training_weight(timesteps) * loss / len(aggregated_tokens_list)
        loss += tF.mse_loss(
            aggregated_tokens_list[-1][:, :, patch_start_idx:, ...],
            vggt_aggregated_tokens_list[-1][:, :, vggt_patch_start_idx:, ...],
            reduction="none"
        ).mean(dim=(1, 2, 3))  # (B,)
        outputs["loss"] = loss = self.scheduler.training_weight(timesteps) * loss

        return outputs

    def compute_loss_dinodistill(self, data: Dict[str, Any], dtype: torch.dtype = torch.float32):
        assert self.dino is not None, "DINO is not loaded in VGGT-VAE"
        outputs = {}

        images = data["image"].to(dtype)  # (B, F, 3, H, W)
        depths = data["depth"].to(dtype)  # (B, F, H, W)

        F_in, (H, W) = images.shape[1], images.shape[-2:]
        _downsample = self.opt.compression_ratio[1] * self.opt.latent_patch_size
        H_14, W_14 = H // _downsample * 14, W // _downsample * 14  # `14`: hard-coded for DINOv2

        # VGGT DINOv2 inputs
        F_recon = 1 + (F_in - 1) // self.opt.compression_ratio[0]  # not reconstruct (upsample) on the frame dimension
        out_idxs = torch.arange(0, F_in, math.ceil(F_in/F_recon), dtype=torch.long, device=images.device)  # (F_recon,)
        assert len(out_idxs) == F_recon
            ## Actual images for VGGT
        recon_images = images[:, out_idxs, ...]  # (B, F_recon, 3, H, W)
        recon_images_14 = tF.interpolate(
            rearrange(recon_images, "b f c h w -> (b f) c h w"),
            size=(H_14, W_14),
            mode="bilinear",
        )  # (B*F_recon, 3, H_14, W_14)
        recon_images_14 = rearrange(recon_images_14, "(b f) c h w -> b f c h w", f=F_recon)  # (B, F_recon, 3, H_14, W_14)
            ## Actual depths for VGGT
        recon_depths = depths[:, out_idxs, ...]  # (B, F_recon, H, W)
        recon_depths_14 = tF.interpolate(
            rearrange(recon_depths, "b f h w -> (b f) h w").unsqueeze(1),
            size=(H_14, W_14),
            mode="nearest-exact",
        ).squeeze(1)  # (B*F_recon, H_14, W_14)
        recon_depths_14 = rearrange(recon_depths_14, "(b f) h w -> b f h w", f=F_recon)  # (B, F_recon, H_14, W_14)

        # VGGT-VAE forward as a student model
        latents = self.encode(images)  # (B, D, F_recon, h, w)
        latent_embeds, _ = self.latent_embed(latents)  # (B*F_recon, N=hh*ww, D')

        # VGGT DINOv2 forward as a teacher model
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            self.dino.eval()
            dino_features = self.dino(
                (rearrange(recon_images_14, "b f c h w -> (b f) c h w") - self._resnet_mean) / self._resnet_std,
            )
            if isinstance(dino_features, dict):
                dino_features = dino_features["x_norm_patchtokens"]  # (B*F_recon, N=hh*ww, D')

        # For visualization
        outputs["images"] = recon_images_14
        outputs["images_depth_gt"] = colorize_depth(1. / recon_depths_14, batch_mode=True)  # disparity for visualization
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            geometry_outputs = self.decode_embed(latent_embeds, recon_images_14, only_depth=True, frames_chunk_size=self.opt.frames_chunk_size)
            pred_depths = geometry_outputs["depth"]  # (B, F_recon, 1, H, W)
        outputs["images_depth"] = colorize_depth(1. / pred_depths, batch_mode=True)  # disparity for visualization

        ################################ Losses and metrics ################################
        loss = 0.

        mse_loss = tF.mse_loss(latent_embeds, dino_features, reduction="none").mean(dim=(1, 2))  # (B*f,)
        outputs["mse_loss"] = mse_loss = rearrange(mse_loss, "(b f) -> b f", f=F_recon).mean(dim=-1)  # (B,)
        loss += mse_loss

        # Cf. https://github.com/hustvl/LightningDiT/blob/main/vavae/ldm/modules/losses/contperceptual.py#L123
        latent_embeds_norm, dino_features_norm = tF.normalize(latent_embeds, dim=-1), tF.normalize(dino_features, dim=-1)
        dino_cos_sim = torch.einsum("bic,bjc->bij", dino_features_norm, dino_features_norm)
        latent_cos_sim = torch.einsum("bic,bjc->bij", latent_embeds_norm, latent_embeds_norm)
        diff = torch.abs(dino_cos_sim - latent_cos_sim)  # (B*f, N, N)
        cos_sim = tF.cosine_similarity(latent_embeds, dino_features, dim=-1).mean(dim=-1)  # (B*f,)

        distmat_loss = tF.relu(diff - self.opt.distmat_margin).mean(dim=(1, 2))  # (B*f,)
        outputs["distmat_loss"] = distmat_loss = rearrange(distmat_loss, "(b f) -> b f", f=F_recon).mean(dim=-1)  # (B,)
        loss += self.opt.distmat_weight * distmat_loss

        cos_loss = tF.relu(1. - self.opt.cos_margin - cos_sim)  # (B*f,)
        outputs["cos_loss"] = cos_loss = rearrange(cos_loss, "(b f) -> b f", f=F_recon).mean(dim=-1)  # (B,)
        loss += self.opt.cos_weight * cos_loss

        outputs["loss"] = loss  # (B,)
        return outputs


    ################################ Helper functions ################################


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

    def latent_embed(self, latents: Tensor, timesteps: Optional[Tensor] = None):
        """ VAE latent to VGGT DINOv2 features with optional timestep condition.

        Inputs:
            - `latents`: (B, D, f, h, w)
            - `timesteps`: (B,)

        Outputs:
            - `latent_embeds`: (B*f, N=hh*ww, D')
            - `t_embeds`: (B*f, D')
        """
        B, _, f, h, w = latents.shape

        latent_embeds = self.latent_patch_embed(rearrange(latents, "b d f h w -> (b f) d h w"))  # (B*f, N=hh*ww, D)

        if self.opt.is_dit_block:
            # timesteps = repeat(timesteps, "b -> (b f)", f=f)
            timesteps = timesteps.unsqueeze(1).repeat(1, f)
            timesteps[:, 0] = 0  # set the first frame's timestep to 0; TODO: make it configurable
            timesteps = timesteps.flatten()  # (B*f,)
            t_embeds = self.t_embedder(timesteps)  # (B*f,) -> (B*f, D)
        else:
            t_embeds = None

        pos = self.latent_position_getter(B * f, h//self.opt.latent_patch_size, w//self.opt.latent_patch_size, device=latents.device)
        for block in self.latent_blocks:
            if self.training:
                latent_embeds = checkpoint(block, latent_embeds, None, pos, t_embeds, use_reentrant=False)
            else:
                latent_embeds = block(latent_embeds, None, pos, t_embeds)

        latent_embeds = self.latent_ln(latent_embeds)

        if self.opt.is_dit_block:
            shift, scale = self.adaLN_modulation(t_embeds).chunk(2, dim=1)
            latent_embeds = modulate(latent_embeds, shift, scale)

        return latent_embeds, t_embeds

    def decode_embed(self,
        latent_embeds: Tensor,
        images: Tensor,  # for shape information only
        extra_info: Optional[Tensor] = None,
        cross_patch_tokens: Optional[Tensor] = None,
        t_embeds: Optional[Tensor] = None,
        use_heads: bool = True,
        only_depth: bool = False,
        frames_chunk_size: int = 8,
    ):
        """ VGGT DINOv2 features to geometry features and attributes.

        Inputs:
            - `latent_embeds`: (B*f, N=hh*ww, D')
            - `images`: (B, F, 3, H, W)
            - `extra_info`: (B, f, 9)
            - `cross_patch_tokens`: (B*f, N', D')
            - `t_embeds`: (B*f, D')
            - `use_heads`: whether to use heads to decode geometry features to attributes
            - `only_depth`: whether to only decode depth
            - `frames_chunk_size`: chunk size on frame dimension for saving memory

        Outputs:
            - `return_dict`: a dictionary of geometry features and attributes
                - `aggregated_tokens_list`: a list of (B, f, N=hh*ww, 2*D')
                - `patch_start_idx`: a list of int
                - `point`: (B, F, 3, H, W)
                - `point_conf`: (B, F, H, W)
                - `depth`: (B, F, 1, H, W)
                - `depth_conf`: (B, F, H, W)
                - `pose_enc_list`: a list of (B, F, 9): absT_quaR_FoV
        """
        return_dict = {}

        # Backbone: `patch_tokens` are provided, so `images` are not really used
        aggregated_tokens_list, patch_start_idx = self.backbone(
            images,  # (B, F, 3, H, W)
            extra_info=extra_info,  # (B, f, 9)
            patch_tokens=latent_embeds,  # (B*F, N=h*w, D)
            cross_patch_tokens=cross_patch_tokens,  # (B*F, N', D)
            temb=t_embeds,  # (B*F, D)
        )
        return_dict["aggregated_tokens_list"] = aggregated_tokens_list
        return_dict["patch_start_idx"] = patch_start_idx

        if use_heads:
            # Point head
            if not only_depth:
                pred_points, point_confs = self.point_head(
                    aggregated_tokens_list,
                    images,
                    patch_start_idx,
                    frames_chunk_size=frames_chunk_size,
                )
                pred_points = rearrange(pred_points, "b f h w c -> b f c h w")
                return_dict["point"] = pred_points  # (B, F, 3, H, W)
                return_dict["point_conf"] = point_confs  # (B, F, H, W)

            # Depth head
            pred_depths, depth_confs = self.depth_head(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_chunk_size=frames_chunk_size,
            )
            pred_depths = rearrange(pred_depths, "b f h w c -> b f c h w")
            return_dict["depth"] = pred_depths  # (B, F, 1, H, W)
            return_dict["depth_conf"] = depth_confs  # (B, F, H, W)

            # Camera head
            if not only_depth:
                pred_pose_enc_list = self.camera_head(aggregated_tokens_list)  # a list of (B, F, 9): absT_quaR_FoV
                return_dict["pose_enc_list"] = pred_pose_enc_list

        return return_dict

    def decode(self,
        latents: Tensor,
        images: Tensor,  # for shape information only
        cross_patch_tokens: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        use_heads: bool = True,
        only_depth: bool = False,
        frames_chunk_size: int = 8,
    ):
        """ VAE latent to geometry features and attributes.

        Inputs:
            - `latents`: (B, D, f, h, w)
            - `images`: (B, F, 3, H, W)
            - `cross_patch_tokens`: (B*f, N', D')
            - `timesteps`: (B,)
            - `use_heads`: whether to use heads to decode geometry features to attributes
            - `only_depth`: whether to only decode depth
            - `frames_chunk_size`: chunk size on frame dimension for saving memory

        Outputs:
            - `return_dict`: a dictionary of geometry features and attributes
                - `aggregated_tokens_list`: a list of (B, f, N=hh*ww, 2*D')
                - `patch_start_idx`: a list of int
                - `point`: (B, F, 3, H, W)
                - `point_conf`: (B, F, H, W)
                - `depth`: (B, F, 1, H, W)
                - `depth_conf`: (B, F, H, W)
                - `pose_enc_list`: a list of (B, F, 9): absT_quaR_FoV
        """
        latent_embeds, t_embeds = self.latent_embed(latents, timesteps)
        return self.decode_embed(latent_embeds, images, cross_patch_tokens, t_embeds, use_heads, only_depth, frames_chunk_size)

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
