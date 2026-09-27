from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from motion import recover_from_ric


MODEL_PATH = (
    Path(os.environ["MOTION_SMPLH_MODEL"])
    if os.environ.get("MOTION_SMPLH_MODEL")
    else None
)


@lru_cache(maxsize=64)
def vertical_root_positions(motion_path: str) -> np.ndarray:
    """Match the browser's root-centred skeleton while preserving vertical motion."""
    features = np.load(motion_path, mmap_mode="r")
    joints = recover_from_ric(features)
    floor = float(np.min(joints[:, :, 1]))
    return np.asarray(joints[:, 0, 1] - floor, dtype=np.float32)


def rotation_6d_to_matrix(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64).reshape(-1, 6)
    first = vectors[:, :3]
    second = vectors[:, 3:]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


class SMPLHRenderer:
    def __init__(self, model_path: Optional[Path] = MODEL_PATH):
        if model_path is None or not model_path.is_file():
            raise FileNotFoundError(str(model_path))

        import torch
        from smplx import SMPL, SMPLH
        from smplx.utils import Struct

        archive = np.load(str(model_path), encoding="latin1")
        data = Struct(**{key: archive[key] for key in archive.files})
        data.hands_componentsl = np.zeros((0,))
        data.hands_componentsr = np.zeros((0,))
        data.hands_meanl = np.zeros((45,))
        data.hands_meanr = np.zeros((45,))
        vertices, dimensions, beta_count = data.shapedirs.shape
        if beta_count < SMPL.SHAPE_SPACE_DIM:
            data.shapedirs = np.concatenate(
                (
                    data.shapedirs,
                    np.zeros(
                        (vertices, dimensions, SMPL.SHAPE_SPACE_DIM - beta_count),
                        dtype=data.shapedirs.dtype,
                    ),
                ),
                axis=-1,
            )

        self.torch = torch
        self.model = SMPLH(
            str(model_path),
            model_type="smplh",
            data_struct=data,
            num_betas=16,
            batch_size=1,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.model.eval()
        self.faces = np.ascontiguousarray(self.model.faces, dtype="<u4")
        self.lock = threading.Lock()

    def vertices_for_frame(self, motion_path: Path, frame: int) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        features = np.load(str(motion_path), mmap_mode="r")
        if features.ndim != 2 or features.shape[1] != 263:
            raise ValueError("SMPL-H conversion expects (T, 263) motion")
        frame = max(0, min(int(frame), features.shape[0] - 1))

        body_matrices = rotation_6d_to_matrix(features[frame, 67:193])
        body_axis_angle = Rotation.from_matrix(body_matrices).as_rotvec().astype(
            np.float32
        )

        root_half_angles = np.zeros(features.shape[0], dtype=np.float32)
        root_half_angles[1:] = features[:-1, 0]
        root_half_angles = np.cumsum(root_half_angles)
        root_orient = np.array(
            [[0.0, -2.0 * root_half_angles[frame], 0.0]], dtype=np.float32
        )

        torch = self.torch
        with self.lock, torch.no_grad():
            output = self.model(
                global_orient=torch.from_numpy(root_orient),
                body_pose=torch.from_numpy(body_axis_angle.reshape(1, -1)),
                left_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
                right_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
                betas=torch.zeros((1, 16), dtype=torch.float32),
                transl=torch.zeros((1, 3), dtype=torch.float32),
                return_full_pose=False,
            )
        vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
        pelvis = output.joints[0, 0].detach().cpu().numpy()
        vertices -= pelvis
        vertices[:, 1] += vertical_root_positions(str(motion_path))[frame]
        if not np.isfinite(vertices).all():
            raise ValueError("SMPL-H conversion produced non-finite vertices")
        return np.ascontiguousarray(vertices, dtype="<f4")


_renderer = None
_renderer_lock = threading.Lock()


def get_renderer() -> SMPLHRenderer:
    global _renderer
    if _renderer is None:
        with _renderer_lock:
            if _renderer is None:
                _renderer = SMPLHRenderer()
    return _renderer
