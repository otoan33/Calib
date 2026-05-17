"""
段階的最適化。Stage のリストとして推定手順を定義する。
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

import numpy as np
from ..models.parameters import ParameterSet
from ..models.base import ObservationTransform
from ..models.matrix import IdentityTransform
from ..models.kinematics import build_kinematic_from_params, apply_transmission_error
from ..models.observation import vec6_to_se3


def compute_residuals(
    x: np.ndarray,
    free_indices: list[int],
    params: ParameterSet,
    dh_nominal: list[dict],
    param_lookup: dict,
    q_traj: np.ndarray,
    p_exp: np.ndarray,
    transform: ObservationTransform,
    include_prior: bool = True,
    q_timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """
    残差ベクトルを返す。scipy.optimize.least_squares の fun 引数として使用。

    "split"   : r = T(y_exp) - T(y_pred)
    "residual": r = T(||p_exp - p_pred||)
    """
    params.set_vector(free_indices, x)

    kin = build_kinematic_from_params(dh_nominal, params, param_lookup)
    T_tool  = vec6_to_se3(np.array([
        params.params[param_lookup[k]].value
        for k in ["tool_tx","tool_ty","tool_tz","tool_rx","tool_ry","tool_rz"]
    ]))
    T_local = vec6_to_se3(np.array([
        params.params[param_lookup[k]].value
        for k in ["local_tx","local_ty","local_tz","local_rx","local_ry","local_rz"]
    ]))
    n_joints = len(dh_nominal)

    if q_timestamps is not None and "time_offset" in param_lookup:
        from scipy.interpolate import interp1d as _interp
        dt_offset = params.params[param_lookup["time_offset"]].value
        t_shifted = q_timestamps + dt_offset
        q_traj_eff = np.column_stack([
            _interp(
                q_timestamps, q_traj[:, j], kind="cubic",
                bounds_error=False,
                fill_value=(q_traj[0, j], q_traj[-1, j]),
            )(t_shifted)
            for j in range(q_traj.shape[1])
        ])
    else:
        q_traj_eff = q_traj

    N = len(q_traj_eff)
    p_pred = np.zeros((N, 3))
    for t in range(N):
        q_eff = apply_transmission_error(q_traj_eff[t], params, param_lookup, n_joints)
        T = kin.forward(q_eff, T_tool, T_local)
        p_pred[t] = T[:3, 3]

    if getattr(transform, "transform_mode", "split") == "residual":
        r_norm = np.linalg.norm(p_exp - p_pred, axis=1)
        r_obs = transform.apply(r_norm)
    else:
        y_exp_flat  = p_exp.flatten()
        y_pred_flat = p_pred.flatten()
        r_obs = transform.apply(y_exp_flat) - transform.apply(y_pred_flat)

    if not include_prior:
        return r_obs

    r_prior = params.get_prior_residuals(free_indices)
    return np.concatenate([r_obs, r_prior])


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
    from ..models.matrix import VelocityNormTransform, FFTAmplitudeTransform

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
