"""
観測モデル g(x)。センサー座標系への変換など。

y_pred(t) = T( g( f(θ, q(t + Δt)) ) )
               ^^^
            ここを担当
"""

import numpy as np
from .base import ObservationModel


def pose_to_position(T: np.ndarray) -> np.ndarray:
    """SE(3) 行列からエンドエフェクタ位置を取り出す。"""
    return T[:3, 3]


def pose_to_axis_angle(T: np.ndarray) -> np.ndarray:
    """
    SE(3) 行列から位置 + 軸角度表現 (px, py, pz, ax, ay, az) を取り出す。

    軸角度: ω = θ * n（‖ω‖ = θ が回転角）
    """
    pos = T[:3, 3]
    R = T[:3, :3]
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if abs(theta) < 1e-9:
        axis = np.zeros(3)
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2 * np.sin(theta))
    return np.concatenate([pos, axis * theta])


def vec6_to_se3(v: np.ndarray) -> np.ndarray:
    """
    6DoF ベクトル [tx, ty, tz, rx, ry, rz] → SE(3) 行列。

    rx,ry,rz は軸角度表現（小角度近似なし、Rodrigues の式）。
    """
    tx, ty, tz, rx, ry, rz = v
    theta = np.sqrt(rx**2 + ry**2 + rz**2)
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    if theta > 1e-9:
        k = np.array([rx, ry, rz]) / theta
        K = np.array([
            [0,    -k[2],  k[1]],
            [k[2],  0,    -k[0]],
            [-k[1], k[0],  0   ],
        ])
        T[:3, :3] = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return T


class PoseObservation(ObservationModel):
    """
    手先の 3D 位置をそのまま観測値として返す。

    最も基本的な観測モデル。レーザートラッカー・CMM など位置計測に対応。
    戻り値: (3,) [x, y, z] [m]
    """

    def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
        return x[:3, 3]

    def predict_batch(self, poses: np.ndarray, params: dict) -> np.ndarray:
        """poses: (N, 4, 4) → (N*3,)"""
        return poses[:, :3, 3].flatten()


class DistanceObservation(ObservationModel):
    """
    固定点 origin から TCP までの距離スカラーを観測値として返す。

    用途: 球形エラーメータ（レーザートラッカーのレトロリフレクタ固定点基準）など。
    戻り値: (1,) [r] [m]
    """

    def __init__(self, origin: np.ndarray = None):
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)

    def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
        return np.array([np.linalg.norm(x[:3, 3] - self.origin)])

    def predict_batch(self, poses: np.ndarray, params: dict) -> np.ndarray:
        """poses: (N, 4, 4) → (N,)"""
        return np.linalg.norm(poses[:, :3, 3] - self.origin, axis=1)
