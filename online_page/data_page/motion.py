from __future__ import annotations

import numpy as np


JOINT_COUNT = 22
FEATURE_COUNT = 263
KINEMATIC_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by quaternions stored as (w, x, y, z)."""
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def recover_from_ric(features: np.ndarray) -> np.ndarray:
    """Recover HumanML3D-style 22-joint positions from a 263-D RIC sequence."""
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != FEATURE_COUNT:
        raise ValueError(f"expected (T, {FEATURE_COUNT}), got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("motion contains NaN or infinite values")

    rot_vel = data[:, 0]
    rot_ang = np.zeros_like(rot_vel)
    rot_ang[1:] = rot_vel[:-1]
    rot_ang = np.cumsum(rot_ang, axis=0)

    root_quat = np.zeros((data.shape[0], 4), dtype=np.float32)
    root_quat[:, 0] = np.cos(rot_ang)
    root_quat[:, 2] = np.sin(rot_ang)
    inv_root_quat = root_quat.copy()
    inv_root_quat[:, 1:] *= -1.0

    root_pos = np.zeros((data.shape[0], 3), dtype=np.float32)
    root_pos[1:, (0, 2)] = data[:-1, 1:3]
    root_pos = _qrot(inv_root_quat, root_pos)
    root_pos = np.cumsum(root_pos, axis=0)
    root_pos[:, 1] = data[:, 3]

    local = data[:, 4 : 4 + (JOINT_COUNT - 1) * 3]
    local = local.reshape(data.shape[0], JOINT_COUNT - 1, 3)
    rotations = np.broadcast_to(
        inv_root_quat[:, None, :], (data.shape[0], JOINT_COUNT - 1, 4)
    )
    positions = _qrot(rotations, local)
    positions[:, :, 0] += root_pos[:, None, 0]
    positions[:, :, 2] += root_pos[:, None, 2]
    recovered = np.concatenate((root_pos[:, None, :], positions), axis=1)
    if not np.isfinite(recovered).all():
        raise ValueError("recovered joints contain NaN or infinite values")
    return recovered


def pack_motion(path: str) -> tuple[np.ndarray, int]:
    features = np.load(path, mmap_mode="r")
    joints = recover_from_ric(features)
    return np.ascontiguousarray(joints, dtype="<f4"), int(features.shape[0])
