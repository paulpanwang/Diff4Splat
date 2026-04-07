from typing import *
from src.utils import StepTracker

import os
import h5py
import numpy as np
import imageio.v2 as iio
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.utils import unproject_depth
from src.data.base_dataset import BaseDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Dl3dvDataset(BaseDataset):
    def __init__(self,
        opt: Options,
        training: bool = True,
        step_tracker: Optional[StepTracker] = None,
    ):
        super().__init__(opt, "dl3dv", training, step_tracker)

        self._valid_idxs = list(range(len(self.sample_dirs)))

    def __len__(self):
        return len(self._valid_idxs)

    def _try_getitem(self, index: int) -> Dict[str, Any]:
        sample_dir = self.sample_dirs[index]
        sample_dir = os.path.join(self.root, sample_dir)
        uid = os.path.basename(sample_dir)

        # Use scenes with MVS depths
        if not os.path.exists(os.path.join(self.root + "_mvs", f"{uid}.h5")):
            if index in self._valid_idxs:
                self._valid_idxs.remove(index)
            if len(self._valid_idxs) == 0:
                raise ValueError(f"No valid samples left in DL3DV MVS")
            return self._try_getitem(np.random.choice(self._valid_idxs))

        # Image & camera are preprocessed in the preprocess script
        file_paths = os.listdir(sample_dir)
        image_paths = sorted(set([os.path.join(sample_dir, file_path) for file_path in file_paths if file_path.endswith(".jpg")]))
        num_frames = len(image_paths)

        # Sample frames
        input_frame_idxs, output_frame_idxs = self._frame_sample(num_frames)
        timesteps = torch.tensor([idx for idx in input_frame_idxs + output_frame_idxs]).float()  # (F,)
        timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())  # (F,)
            ## Randomly fixed timesteps for output frames
        if self.is_static and np.random.rand() < 0.5 and self.training:
            F_in = self.opt.num_input_frames
            timesteps[F_in:] = torch.ones_like(timesteps[F_in:]) * torch.rand(1)
        selected_frame_idxs = set(input_frame_idxs + output_frame_idxs)  # to avoid duplicate file access

        # Load images (.jpg)
        images = {
            idx: tvT.ToTensor()(iio.imread(image_paths[idx]))
            for idx in selected_frame_idxs
        }
        images = torch.stack([images[idx] for idx in (input_frame_idxs + output_frame_idxs)]).float()  # (F, 3, H, W)
        F, (H, W) = images.shape[0], images.shape[-2:]

        # # Load cameras (.h5): COLMAP + MegaSaM-scaled
        # with h5py.File(os.path.join(self.root + "_colmap", f"{uid}.h5"), "r") as f:
        #     assert len(image_paths) == len(f["camera_pose"])
        #     scale, shift = f["scale"][()], f["shift"][()]
        #     cameras = {}
        #     for idx in selected_frame_idxs:
        #         _camera_pose = f["camera_pose"][idx]
        #         _camera_pose[:3, 3] = _camera_pose[:3, 3] * scale + shift
        #         cameras[idx] = {"camera_pose": _camera_pose, "fxfycxcy": f["fxfycxcy"][idx]}
        # Load cameras (.npz): COLMAP
        camera_paths = sorted(set([os.path.join(sample_dir, file_path) for file_path in file_paths if file_path.endswith(".npz")]))
        assert len(camera_paths) == len(image_paths)
        cameras = {
            idx: np.load(camera_paths[idx])
            for idx in selected_frame_idxs
        }
        C2W = torch.from_numpy(np.stack([cameras[idx]["camera_pose"] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, 4, 4)
        fxfycxcy = torch.from_numpy(np.stack([cameras[idx]["fxfycxcy"] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, 4)

        # # Load depths (.h5): Video-Depth-Anything + DepthPro-scaled
        # with h5py.File(os.path.join(self.root + "_vda", f"{uid}.h5"), "r") as f:
        #     assert len(image_paths) == len(f["disparity"])
        #     scale, shift = f["scale"][()], f["shift"][()]  # metric scale and shift scalars for disparity
        #     disparities = {
        #         idx: f["disparity"][idx] * scale + shift
        #         for idx in selected_frame_idxs
        #     }
        # disparities = torch.from_numpy(np.stack([disparities[idx] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, H, W)
        #     ## Depth masks
        # depth_masks = (1./self.opt.zfar <= disparities) & (disparities <= 1./self.opt.znear)  # (F, H, W)
        # disparities = disparities.clamp(1./self.opt.zfar, 1./self.opt.znear)
        # depths = 1. / disparities
        # Load depths (.h5): MVS depths
        with h5py.File(os.path.join(self.root + "_mvs", f"{uid}.h5"), "r") as f:
            if len(f["depth"]) != num_frames:  # Benchmark: 80/140; 960P: 6176/10013
                if index in self._valid_idxs:
                    self._valid_idxs.remove(index)
                if len(self._valid_idxs) == 0:
                    raise ValueError(f"No valid samples left in DL3DV MVS")
                return self._try_getitem(np.random.choice(self._valid_idxs))

            assert len(image_paths) == len(f["depth"])
            depths = {
                idx: f["depth"][idx]
                for idx in selected_frame_idxs
            }
        depths = torch.from_numpy(np.stack([depths[idx] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, H, W)
            ## Depth masks
        depth_masks = (depths > 0) & (~torch.isnan(depths)) & (torch.isfinite(depths)) # & (depths < self.opt.zfar)  # (F, H, W)
        depths[~depth_masks] = self.opt.zfar  # set invalid depths to far depth; not necessary, but for better visualization
        depths.nan_to_num_(nan=self.opt.zfar, posinf=self.opt.zfar, neginf=self.opt.zfar)
        # depths = depths.clamp(self.opt.znear, self.opt.zfar)  # NOTE: don't clamp MVS depths, available only for metric depth

        # Dummy 3D point tracking
        tracks_world = torch.zeros(F, 1, 3).float()  # (F, N=1, 3)
        tracks_xy = torch.zeros(F, 1, 2).long()  # (F, N=1, 2)
        visibilities = torch.zeros(F, 1).bool()  # (F, N=1)

        # Data augmentation
        images, depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities = \
            self._data_augment(images, depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities)
            ## (Optional) Mask by quantile after downsampling for efficiency
        if self.min_depth_quantile is not None:
            depth_masks = depth_masks & (depths > np.quantile(depths.numpy(), self.min_depth_quantile))
        if self.max_depth_quantile is not None:
            depth_masks = depth_masks & (depths < np.quantile(depths.numpy(), self.max_depth_quantile))

        # Camera normalization
            ## 1. Transform 3D tracks if needed
            ## 2. Scale 3D tracks and depths if needed
        C2W, depths, tracks_world = self._camera_normalize(C2W, depths, tracks_world)

        # (Optional) Scaling depth and camera pose and 3D points according to the XYZ normalization; cf. VGGT
        if self.opt.norm_xyz:
            F_in = self.opt.num_input_frames
            _xyz = unproject_depth(depths[None, :F_in, ...], C2W[None, :F_in, ...], fxfycxcy[None, :F_in, ...]).squeeze(0)
            _xyz_norm = (_xyz.norm(dim=1) * depth_masks[:F_in, ...]).sum() / (depth_masks[:F_in, ...].sum() + 1e-6)
            scaling_factor = 1. / (_xyz_norm + 1e-6) * self.opt.camera_norm_unit
            depths = depths * scaling_factor
            C2W[:, :3, 3] = C2W[:, :3, 3] * scaling_factor
            tracks_world = tracks_world * scaling_factor

        return {
            "name": self.name,                         # str
            "uid": uid,                                # str
            "timestep": timesteps,                     # Tensor: (F,)
            "image": images,                           # Tensor: (F, 3, H, W)
            "C2W": C2W,                                # Tensor: (F, 4, 4)
            "fxfycxcy": fxfycxcy,                      # Tensor: (F, 4)
            "depth": depths,                           # Tensor: (F, H, W)
            "depth_mask": depth_masks.bool(),          # BoolTensor: (F, H, W)
            "track_world": tracks_world,               # Tensor: (F, N, 3)
            "track_xy": tracks_xy.long(),              # LongTensor: (F, N, 2)
            "visibility": visibilities.bool(),         # BoolTensor: (F, N)
            "depth_weight": torch.tensor(0.9),         # Tensor: (1,); MVS estimated depth
            "motion_weight": torch.tensor(1.),         # Tensor: (1,)
        }
