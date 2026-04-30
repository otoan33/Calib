"""
パラメータ管理。各パラメータは name, value, fixed, 事前分布, group を持つ。
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Parameter:
    name: str
    value: float
    fixed: bool = False
    prior_mean: float = 0.0
    prior_std: float = 1e3    # 大きいほど弱い事前分布（実質的に無制約）
    group: str = "default"


class ParameterSet:
    """
    Parameter のリストを管理し、推定ベクトルへの変換を担う。

    固定パラメータは最適化ベクトルに含めない。
    事前分布制約は残差末尾への付加項として get_prior_residuals() で取得する。
    """

    def __init__(self, params: list[Parameter]):
        self.params = params

    # --- インデックス管理 ---

    def free_indices(self, groups: list[str] | None = None) -> list[int]:
        """
        最適化対象（fixed=False かつ groups に含まれる）パラメータのインデックス。

        groups=None のとき全 free パラメータを返す。
        """
        return [
            i for i, p in enumerate(self.params)
            if not p.fixed and (groups is None or p.group in groups)
        ]

    def get_vector(self, indices: list[int]) -> np.ndarray:
        """指定インデックスのパラメータ値を配列として返す。"""
        return np.array([self.params[i].value for i in indices])

    def set_vector(self, indices: list[int], values: np.ndarray) -> None:
        """指定インデックスのパラメータ値を更新する。"""
        for i, v in zip(indices, values):
            self.params[i].value = v

    # --- 事前分布残差 ---

    def get_prior_residuals(self, indices: list[int]) -> np.ndarray:
        """
        事前分布制約の残差項：r_prior[j] = (θ[j] - μ[j]) / σ[j]

        最小二乗残差の末尾に連結することでL2正則化と等価になる。
        """
        r = np.array([
            (self.params[i].value - self.params[i].prior_mean) / self.params[i].prior_std
            for i in indices
        ])
        return r

    # --- ユーティリティ ---

    def summary(self) -> str:
        lines = ["Parameters:"]
        for p in self.params:
            status = "fixed" if p.fixed else f"free  [{p.group}]"
            lines.append(
                f"  {p.name:30s}  val={p.value:+.6f}  "
                f"prior=({p.prior_mean:.4f}±{p.prior_std:.4f})  {status}"
            )
        return "\n".join(lines)


def make_default_parameter_set(dh_nominal: list[dict]) -> ParameterSet:
    """
    6軸ロボットのデフォルトパラメータセットを生成する。

    dh_nominal: [{"alpha": ..., "a": ..., "d": ..., "theta_offset": ...}, ...]
    """
    params = []

    for i, dh in enumerate(dh_nominal):
        # DH 誤差パラメータ（名目値からのずれとして定義）
        # prior_std は非情報的（データが事前分布を十分上回るよう大きくとる）
        # prior_std は事実上の非情報的事前分布
        # データが支配するよう十分に大きくとる（≫ 想定最大誤差）
        params.append(Parameter(
            name=f"d_alpha_{i}", value=0.0, group="kinematic",
            prior_std=np.pi,     # ±π rad（完全非情報的）
        ))
        params.append(Parameter(
            name=f"d_a_{i}", value=0.0, group="kinematic",
            prior_std=1.0,       # ±1m（完全非情報的）
        ))
        params.append(Parameter(
            name=f"d_d_{i}", value=0.0, group="kinematic",
            prior_std=1.0,
        ))
        params.append(Parameter(
            name=f"d_theta_offset_{i}", value=0.0, group="kinematic",
            prior_std=np.pi,
        ))

    # ツール変換誤差（6DoF → tx, ty, tz, rx, ry, rz）
    for k in ["tx", "ty", "tz", "rx", "ry", "rz"]:
        std = 1.0 if k.startswith("t") else np.pi
        params.append(Parameter(
            name=f"tool_{k}", value=0.0, group="tool", prior_std=std,
        ))

    # ローカル（ベース）座標系誤差
    for k in ["tx", "ty", "tz", "rx", "ry", "rz"]:
        std = 1.0 if k.startswith("t") else np.pi
        params.append(Parameter(
            name=f"local_{k}", value=0.0, group="local", prior_std=std,
        ))

    # 時刻ずれ
    # prior_std=1.0: VelocityNorm/Identity 残差（m/s または m スケール）に対して
    # 事前分布が支配しないよう十分大きくとる（観測ヘッセ行列 >> 事前ヘッセ行列）
    params.append(Parameter(
        name="time_offset", value=0.0, group="time_offset",
        prior_std=1.0,
    ))

    # 関節角度伝達誤差（各軸、周期的誤差の振幅・位相）
    for i in range(6):
        params.append(Parameter(
            name=f"trans_err_amp_{i}", value=0.0,
            group="joint_transmission_error",
            prior_std=np.pi,   # 非情報的（振幅は観測データで決まる）
        ))
        params.append(Parameter(
            name=f"trans_err_phase_{i}", value=0.0,
            group="joint_transmission_error",
            prior_std=np.pi,
        ))

    return ParameterSet(params)
