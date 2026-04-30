"""
観測モデル g(x)。センサー座標系への変換など。

y_pred(t) = T( g( f(θ, q(t + Δt)) ) )
               ^^^
            ここを担当
"""

import numpy as np
from .kinematics import RobotKinematics, DHParams
from .parameters import ParameterSet


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


class PositionObservationModel:
    """
    観測モデル：エンドエフェクタ位置を直接観測する場合。

    g(T) = T[:3, 3]
    """

    def predict(
        self,
        kin: RobotKinematics,
        q: np.ndarray,
        params: ParameterSet,
        param_lookup: dict,
    ) -> np.ndarray:
        """
        順運動学を解いてエンドエフェクタ位置を返す。

        param_lookup: パラメータ名 → params.params インデックスのマップ
        """
        T_tool  = _build_tool_transform(params, param_lookup)
        T_local = _build_local_transform(params, param_lookup)
        T = kin.forward(q, T_tool, T_local)
        return pose_to_position(T)


def _build_tool_transform(params: ParameterSet, param_lookup: dict) -> np.ndarray:
    v = np.array([
        params.params[param_lookup[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ])
    return vec6_to_se3(v)


def _build_local_transform(params: ParameterSet, param_lookup: dict) -> np.ndarray:
    v = np.array([
        params.params[param_lookup[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ])
    return vec6_to_se3(v)
