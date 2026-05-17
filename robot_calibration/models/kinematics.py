"""
DHパラメータによる順運動学と解析的ヤコビアン。DHKinematics を含む。

DH規約（Modified DH / Craig規約）:
  T_i = Rot_x(alpha_{i-1}) * Trans_x(a_{i-1}) * Rot_z(theta_i) * Trans_z(d_i)

各リンク変換:
  | cos(θ)        -sin(θ)         0          a    |
  | sin(θ)cos(α)   cos(θ)cos(α)  -sin(α)  -d sin(α) |
  | sin(θ)sin(α)   cos(θ)sin(α)   cos(α)   d cos(α) |
  |    0               0             0        1    |
"""

import numpy as np
from dataclasses import dataclass, field
from .base import KinematicModel
from .observation import vec6_to_se3


@dataclass
class DHParams:
    """1リンク分のDHパラメータ。"""
    alpha: float   # x軸回り回転 [rad]
    a: float       # x方向リンク長 [m]
    d: float       # z方向オフセット [m]
    theta_offset: float  # 関節角度オフセット [rad]（名目値からのずれ）


def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """
    Modified DH変換行列 T_i ∈ SE(3) を計算する。

    θ_total = theta（関節角度） は呼び出し元で theta + theta_offset として渡す。
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,    -st,     0,    a   ],
        [st*ca,  ct*ca, -sa,  -d*sa],
        [st*sa,  ct*sa,  ca,   d*ca],
        [0,      0,      0,    1   ],
    ])


def dh_transform_dtheta(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """
    dT/dθ：DHパラメータ θ に関する変換行列の偏微分。

    d/dθ [cos θ] = -sin θ,  d/dθ [sin θ] = cos θ
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [-st,   -ct,    0,    0   ],
        [ ct*ca, -st*ca, 0,   0   ],
        [ ct*sa, -st*sa, 0,   0   ],
        [ 0,     0,      0,   0   ],
    ])


def dh_transform_dalpha(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """dT/dα：alpha に関する偏微分。"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [0,      0,      0,     0      ],
        [-st*sa,  -ct*sa, -ca,  -d*ca  ],
        [ st*ca,   ct*ca, -sa,  -d*sa  ],
        [0,       0,      0,     0     ],
    ])


def dh_transform_da(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """dT/da：a に関する偏微分。"""
    return np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=float)


def dh_transform_dd(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """dT/dd：d に関する偏微分。"""
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [0, 0, 0,  0   ],
        [0, 0, 0, -sa  ],
        [0, 0, 0,  ca  ],
        [0, 0, 0,  0   ],
    ])


class RobotKinematics:
    """
    DHパラメータリストで定義される6軸ロボットの順運動学。

    ツール変換（T_tool）とローカル変換（T_local）を含む全変換：
        T_total = T_local @ T_0 @ T_1 @ ... @ T_n @ T_tool
    """

    def __init__(self, dh_params: list[DHParams]):
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    def forward(
        self,
        q: np.ndarray,
        T_tool: np.ndarray = None,
        T_local: np.ndarray = None,
    ) -> np.ndarray:
        """
        順運動学。関節角度 q [n_joints] からエンドエフェクタ姿勢 T ∈ SE(3) を返す。

        q[i] は i 番目関節の角度。theta_offset を加えた値でリンク行列を計算する。
        """
        if T_tool is None:
            T_tool = np.eye(4)
        if T_local is None:
            T_local = np.eye(4)

        T = np.eye(4)
        for i, p in enumerate(self.dh_params):
            theta = q[i] + p.theta_offset
            T = T @ dh_transform(p.alpha, p.a, p.d, theta)

        return T_local @ T @ T_tool

    def link_transforms(
        self, q: np.ndarray
    ) -> list[np.ndarray]:
        """
        各リンクの累積変換 T_0..i を返す（ヤコビアン計算用）。

        returns: [T_0, T_01, T_012, ..., T_0..n]  長さ n_joints+1
        """
        Ts = [np.eye(4)]
        for i, p in enumerate(self.dh_params):
            theta = q[i] + p.theta_offset
            Ts.append(Ts[-1] @ dh_transform(p.alpha, p.a, p.d, theta))
        return Ts

    def jacobian_position(
        self,
        q: np.ndarray,
        T_tool: np.ndarray = None,
        T_local: np.ndarray = None,
    ) -> np.ndarray:
        """
        エンドエフェクタ位置のヤコビアン ∂p_ee/∂θ ∈ R^{3 × n_joints}。

        連鎖律：
            ∂T_total/∂θ_i = T_local @ T_0..{i-1} @ (∂T_i/∂θ_i) @ T_{i+1}..n @ T_tool

        位置列は ∂T/∂θ_i の右上 3×1 ブロック。
        """
        if T_tool is None:
            T_tool = np.eye(4)
        if T_local is None:
            T_local = np.eye(4)

        Ts = self.link_transforms(q)

        # T_suffix[i] = T_i @ T_{i+1} @ ... @ T_n @ T_tool
        T_suffix = np.eye(4)
        T_suffix = self.dh_params  # dummy, will compute below
        suffixes = [T_tool]
        for i in range(self.n_joints - 1, -1, -1):
            p = self.dh_params[i]
            theta = q[i] + p.theta_offset
            suffixes.insert(0, dh_transform(p.alpha, p.a, p.d, theta) @ suffixes[0])

        J = np.zeros((3, self.n_joints))
        for i, p in enumerate(self.dh_params):
            theta = q[i] + p.theta_offset
            dTi_dtheta = dh_transform_dtheta(p.alpha, p.a, p.d, theta)

            # ∂T_total/∂θ_i = T_local @ T_0..{i-1} @ dTi/dθ @ T_{i+1}..n @ T_tool
            dT_total = T_local @ Ts[i] @ dTi_dtheta @ suffixes[i + 1]
            J[:, i] = dT_total[:3, 3]

        return J

    def jacobian_dh_params(
        self,
        q: np.ndarray,
        T_tool: np.ndarray = None,
        T_local: np.ndarray = None,
    ) -> dict[str, np.ndarray]:
        """
        DHパラメータに対する位置ヤコビアン。

        返り値: {
            "alpha_i": ∂p/∂alpha_i  (3,),
            "a_i":     ∂p/∂a_i      (3,),
            "d_i":     ∂p/∂d_i      (3,),
            "theta_offset_i": ∂p/∂theta_offset_i  (3,),
        }
        """
        if T_tool is None:
            T_tool = np.eye(4)
        if T_local is None:
            T_local = np.eye(4)

        Ts = self.link_transforms(q)

        suffixes = [T_tool]
        for i in range(self.n_joints - 1, -1, -1):
            p = self.dh_params[i]
            theta = q[i] + p.theta_offset
            suffixes.insert(0, dh_transform(p.alpha, p.a, p.d, theta) @ suffixes[0])

        grads = {}
        for i, p in enumerate(self.dh_params):
            theta = q[i] + p.theta_offset
            prefix = T_local @ Ts[i]
            suffix = suffixes[i + 1]

            for param_name, dTi in [
                (f"alpha_{i}", dh_transform_dalpha(p.alpha, p.a, p.d, theta)),
                (f"a_{i}",     dh_transform_da(p.alpha, p.a, p.d, theta)),
                (f"d_{i}",     dh_transform_dd(p.alpha, p.a, p.d, theta)),
                # theta_offset は θ と同じ偏微分
                (f"theta_offset_{i}", dh_transform_dtheta(p.alpha, p.a, p.d, theta)),
            ]:
                dT_total = prefix @ dTi @ suffix
                grads[param_name] = dT_total[:3, 3]

        return grads


def numerical_jacobian_dtheta(
    kin: RobotKinematics,
    q: np.ndarray,
    T_tool: np.ndarray = None,
    T_local: np.ndarray = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """数値微分で ∂p/∂θ を計算する。解析的ヤコビアンの検証用。"""
    if T_tool is None:
        T_tool = np.eye(4)
    if T_local is None:
        T_local = np.eye(4)

    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        q_plus = q.copy(); q_plus[i] += eps
        q_minus = q.copy(); q_minus[i] -= eps
        p_plus  = kin.forward(q_plus,  T_tool, T_local)[:3, 3]
        p_minus = kin.forward(q_minus, T_tool, T_local)[:3, 3]
        J[:, i] = (p_plus - p_minus) / (2 * eps)
    return J


def build_kinematic_from_params(
    dh_nominal: list[dict],
    params,
    param_lookup: dict,
) -> "RobotKinematics":
    """名目DH + 誤差パラメータから RobotKinematics を構築する。"""
    dh_list = []
    for i, dh in enumerate(dh_nominal):
        d_alpha = params.params[param_lookup[f"d_alpha_{i}"]].value
        d_a     = params.params[param_lookup[f"d_a_{i}"]].value
        d_d     = params.params[param_lookup[f"d_d_{i}"]].value
        d_theta = params.params[param_lookup[f"d_theta_offset_{i}"]].value
        dh_list.append(DHParams(
            alpha        = dh["alpha"]        + d_alpha,
            a            = dh["a"]            + d_a,
            d            = dh["d"]            + d_d,
            theta_offset = dh["theta_offset"] + d_theta,
        ))
    return RobotKinematics(dh_list)


def apply_transmission_error(
    q: np.ndarray,
    params,
    param_lookup: dict,
    n_joints: int,
) -> np.ndarray:
    """
    関節角度伝達誤差モデルを適用する。

    q_actual[i] = q[i] + amp[i] * sin(q[i] + phase[i])
    """
    q_corrected = q.copy()
    for i in range(n_joints):
        key_amp   = f"trans_err_amp_{i}"
        key_phase = f"trans_err_phase_{i}"
        if key_amp in param_lookup and key_phase in param_lookup:
            amp   = params.params[param_lookup[key_amp]].value
            phase = params.params[param_lookup[key_phase]].value
            q_corrected[i] += amp * np.sin(q[i] + phase)
    return q_corrected


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
