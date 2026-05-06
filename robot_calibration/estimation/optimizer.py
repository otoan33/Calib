"""
段階的最適化。Stage のリストとして推定手順を定義する。
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from ..models.parameters import ParameterSet
from ..models.transforms import ObservationTransform, IdentityTransform
from ..models.residuals import compute_residuals


@dataclass
class Stage:
    name: str
    param_groups: list[str]
    transform: ObservationTransform
    data_subset: str | None = None   # "trajectory", "point_cloud", None=全部


@dataclass
class StageResult:
    stage_name: str
    x_opt: np.ndarray
    cost: float
    success: bool
    message: str
    jacobian: np.ndarray | None = None  # scipy least_squares の res.jac（不確かさ評価用）


def run_staged_optimization(
    stages: list[Stage],
    params: ParameterSet,
    dh_nominal: list[dict],
    param_lookup: dict,
    q_traj: np.ndarray,
    p_exp: np.ndarray,
    final_full_tune: bool = True,
    ls_kwargs: dict = None,
    q_timestamps: np.ndarray | None = None,
) -> list[StageResult]:
    """
    Stage リストを順に実行する段階的最適化。

    final_full_tune=True のとき、全ステージ完了後に全パラメータを解放して
    IdentityTransform で最終チューニングを行う。
    q_timestamps : 制御タイムスタンプ (N,)。time_offset 推定時に必要。
    """
    if ls_kwargs is None:
        ls_kwargs = {
            "method": "trf",
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-12,
            "x_scale": "jac",
            "max_nfev": 50000,
        }

    results = []

    for stage in stages:
        result = _run_single_stage(
            stage, params, dh_nominal, param_lookup, q_traj, p_exp, ls_kwargs,
            q_timestamps=q_timestamps,
        )
        results.append(result)
        print(f"[{stage.name}] cost={result.cost:.6f}  {result.message}")

    if final_full_tune:
        final_stage = Stage(
            name="final_full_tune",
            param_groups=None,   # None = 全グループ
            transform=IdentityTransform(),
        )
        result = _run_single_stage(
            final_stage, params, dh_nominal, param_lookup, q_traj, p_exp, ls_kwargs,
            q_timestamps=q_timestamps,
        )
        results.append(result)
        print(f"[final_full_tune] cost={result.cost:.6f}  {result.message}")

    return results


def _run_single_stage(
    stage: Stage,
    params: ParameterSet,
    dh_nominal: list[dict],
    param_lookup: dict,
    q_traj: np.ndarray,
    p_exp: np.ndarray,
    ls_kwargs: dict,
    q_timestamps: np.ndarray | None = None,
) -> StageResult:
    free_idx = params.free_indices(groups=stage.param_groups)

    if len(free_idx) == 0:
        return StageResult(
            stage_name=stage.name, x_opt=np.array([]),
            cost=0.0, success=True, message="no free parameters",
        )

    x0 = params.get_vector(free_idx)

    def fun(x):
        return compute_residuals(
            x, free_idx, params, dh_nominal, param_lookup,
            q_traj, p_exp, stage.transform, include_prior=True,
            q_timestamps=q_timestamps,
        )

    res = least_squares(fun, x0, **ls_kwargs)
    params.set_vector(free_idx, res.x)

    return StageResult(
        stage_name=stage.name,
        x_opt=res.x,
        cost=res.cost,
        success=res.success,
        message=res.message,
    )


def default_stages(transforms_override: dict = None) -> list[Stage]:
    """
    デフォルトの4ステージ設定を返す。

    transforms_override: {"time_offset": MyTransform(), ...} で個別上書き可。
    """
    from ..models.transforms import VelocityNormTransform, FFTAmplitudeTransform

    t = transforms_override or {}

    return [
        Stage(
            name="stage1_time_offset",
            param_groups=["time_offset"],
            transform=t.get("time_offset", VelocityNormTransform()),
            data_subset="trajectory",
        ),
        Stage(
            name="stage2_transmission_error",
            param_groups=["joint_transmission_error"],
            transform=t.get("joint_transmission_error", FFTAmplitudeTransform()),
        ),
        Stage(
            name="stage3_kinematics",
            param_groups=["kinematic", "tool", "local"],
            transform=t.get("kinematics", IdentityTransform()),
        ),
    ]
