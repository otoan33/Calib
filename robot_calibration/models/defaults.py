"""
よく使うモデル実装のデフォルト集。

レシピでは差分だけ書けばよいよう、ここに汎用実装を用意する。
  DHKinematics      : Modified DH パラメータによる FK（6軸ロボット標準）
  PoseObservation   : 3D 位置をそのまま観測値として返す
  DistanceObservation: 固定点からの距離スカラーを観測値として返す
"""

import numpy as np
from .base import KinematicModel, ObservationModel
from .kinematics import RobotKinematics, DHParams
from .observation import vec6_to_se3


class DHKinematics(KinematicModel):
    """
    Modified DH パラメータによる順運動学。

    params キー（すべて Optional、省略時は 0.0）:
      d_alpha_i, d_a_i, d_d_i, d_theta_offset_i  — DH 誤差（i=0..n-1）
      tool_tx/ty/tz, tool_rx/ry/rz               — ツール変換誤差 [m, rad]
      local_tx/ty/tz, local_rx/ry/rz             — ローカル座標系誤差
      time_offset                                 — 時刻ずれ（pipeline 側で補間）

    レシピ側でサブクラス化して forward() をオーバーライドすれば
    任意の追加モデル（重力補償項・フレキシビリティなど）を追加できる。
    """

    def __init__(self, dh_nominal: list[dict]):
        self.dh_nominal = dh_nominal
        self.n_joints = len(dh_nominal)

    def _build_kin(self, params: dict) -> tuple[RobotKinematics, np.ndarray, np.ndarray]:
        dh_list = [
            DHParams(
                alpha=dh["alpha"] + params.get(f"d_alpha_{i}", 0.0),
                a=dh["a"]         + params.get(f"d_a_{i}",     0.0),
                d=dh["d"]         + params.get(f"d_d_{i}",     0.0),
                theta_offset=dh["theta_offset"] + params.get(f"d_theta_offset_{i}", 0.0),
            )
            for i, dh in enumerate(self.dh_nominal)
        ]
        T_tool  = vec6_to_se3(np.array([params.get(k, 0.0)
                               for k in ["tool_tx","tool_ty","tool_tz",
                                         "tool_rx","tool_ry","tool_rz"]]))
        T_local = vec6_to_se3(np.array([params.get(k, 0.0)
                               for k in ["local_tx","local_ty","local_tz",
                                         "local_rx","local_ry","local_rz"]]))
        return RobotKinematics(dh_list), T_tool, T_local

    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        kin, T_tool, T_local = self._build_kin(params)
        return kin.forward(q, T_tool, T_local)   # (4, 4) SE(3)

    def jacobian(self, q: np.ndarray, params: dict) -> np.ndarray:
        """関節角度に対する位置ヤコビアン ∂p/∂q ∈ R^{3 × n_joints}。"""
        kin, T_tool, T_local = self._build_kin(params)
        return kin.jacobian_position(q, T_tool, T_local)


class PoseObservation(ObservationModel):
    """
    手先の 3D 位置をそのまま観測値として返す。

    最も基本的な観測モデル。レーザートラッカー・CMM など位置計測に対応。
    戻り値: (3,) [x, y, z] [m]
    """

    def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
        return x[:3, 3]


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
