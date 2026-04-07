from typing import *
from src.utils import StepTracker

import os
from copy import deepcopy
import numpy as np
import cv2
import torch

from src.options import Options
from src.utils import inverse_c2w, homogenize_points
from src.data.base_dataset import BaseDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Tapvid3dDataset(BaseDataset):
    def __init__(self,
        opt: Options,
        training: bool = True,
        step_tracker: Optional[StepTracker] = None,
        subdataset: Literal["adt", "drivetrack", "pstudio"] = "adt",
    ):
        super().__init__(opt, "tapvid3d", training, step_tracker)

        self.root = os.path.join(self.root, subdataset)
        self.sample_dirs = os.listdir(self.root)

    def _try_getitem(self, index: int) -> Dict[str, Any]:
        sample_path = self.sample_dirs[index]
        sample_path = os.path.join(self.root, sample_path)
        uid = os.path.basename(sample_path)

        in_npz = np.load(sample_path)
        images_jpeg_bytes = in_npz["images_jpeg_bytes"]
        tracks_xyz = in_npz["tracks_XYZ"]  # (F, N, 3)
        visibilities = in_npz["visibility"]  # (F, N)
        intrinsics = in_npz["fx_fy_cx_cy"]  # (4,)

        if "extrinsics_w2c" in in_npz.files:
            extrinsics_w2c = in_npz["extrinsics_w2c"]
        else:
            extrinsics_w2c = np.eye(4, 4)[None, ...].repeat(len(tracks_xyz), axis=0)

        video = []
        for frame_bytes in images_jpeg_bytes:
            arr = np.frombuffer(frame_bytes, np.uint8)
            image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            video.append(image_rgb)
        video = np.stack(video, axis=0)

        idxs = np.linspace(0, min(len(video), self.max_bounded_gap)-1, self.opt.num_input_frames, dtype=int)
        video = video[idxs]
        tracks_xyz = tracks_xyz[idxs]
        visibilities = visibilities[idxs]
        extrinsics_w2c = extrinsics_w2c[idxs]

        def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
            """Project 3d points to 2d image plane."""
            u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
            v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

            f_u, f_v, c_u, c_v = camera_intrinsics

            u_d = u_d * f_u + c_u
            v_d = v_d * f_v + c_v

            # Mask of points that are in front of the camera and within image boundary
            masks = (camera_pov_points3d[..., 2] >= self.opt.znear)
            masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
            return np.stack([u_d, v_d], axis=-1), masks
        tracks_xy, infront_cameras = project_points_to_video_frame(tracks_xyz, intrinsics, video.shape[1], video.shape[2])  # (F, N, 2), (F, N)

        # Load images
        images = torch.from_numpy(video / 255.).float().permute(0, 3, 1, 2)  # (F, 3, H, W)
        F, (H, W) = images.shape[0], images.shape[-2:]
        timesteps = torch.linspace(0., 1., F).float()

        # Load cameras
        C2W = np.linalg.inv(extrinsics_w2c)  # (F, 4, 4)
        fxfycxcy = deepcopy(intrinsics)  # (4,)
        fxfycxcy[0] /= W
        fxfycxcy[1] /= H
        fxfycxcy[2] /= W
        fxfycxcy[3] /= H
        C2W = torch.from_numpy(C2W).float()  # (F, 4, 4)
        fxfycxcy = torch.from_numpy(fxfycxcy).float()  # (4,)
        fxfycxcy = fxfycxcy.unsqueeze(0).repeat(F, 1)  # (F, 4)

        # Load 3D tracks and depths
        tracks_xyz = torch.from_numpy(tracks_xyz).float()  # (F, N, 3)
        visibilities = torch.from_numpy(visibilities).bool()  # (F, N)
        tracks_xy = torch.from_numpy(tracks_xy).long()  # (F, N, 2)
        infront_cameras = torch.from_numpy(infront_cameras).bool()  # (F, N)
        visibilities = visibilities & infront_cameras  # (F, N)

        # Data augmentation
        images, _, _, C2W, fxfycxcy, tracks_xyz, tracks_xy, visibilities = \
            self._data_augment(images, None, None, C2W, fxfycxcy, tracks_xyz, tracks_xy, visibilities)

        # Camera normalization
        C2W = inverse_c2w(C2W[0:1]) @ C2W  # (F, 4, 4)

        # (Optional) Scaling depth and camera pose and 3D points according to the XYZ normalization; cf. VGGT
        if self.opt.norm_xyz:
            tracks_xyz_homo = homogenize_points(tracks_xyz)  # (F, N, 4)
            tracks_xyz_world = []
            for i in range(F):
                tracks_xyz_world.append(tracks_xyz_homo[i] @ C2W[i].T)
            tracks_xyz_world = torch.stack(tracks_xyz_world, dim=0)  # (F, N, 4)
            tracks_xyz_world = tracks_xyz_world[:, :, :3] / tracks_xyz_world[:, :, 3:]  # (F, N, 3)
            _xyz_norm = tracks_xyz_world.norm(dim=-1).mean()
            scaling_factor = 1. / (_xyz_norm + 1e-6) * self.opt.camera_norm_unit
            tracks_xyz = tracks_xyz * scaling_factor
            C2W[:, :3, 3] = C2W[:, :3, 3] * scaling_factor

        return {
            "name": self.name,                         # str
            "uid": uid,                                # str
            "timestep": timesteps,                     # Tensor: (F,)
            "image": images,                           # Tensor: (F, 3, H, W)
            "C2W": C2W,                                # Tensor: (F, 4, 4)
            "fxfycxcy": fxfycxcy,                      # Tensor: (F, 4)
            "track_xyz": tracks_xyz,                   # Tensor: (F, N, 3)
            "track_xy": tracks_xy.long(),              # LongTensor: (F, N, 2)
            "visibility": visibilities.bool(),         # BoolTensor: (F, N)
        }
