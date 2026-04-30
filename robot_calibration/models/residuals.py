"""
残差関数。

r(t) = T(y_exp(t)) - T(y_pred(t))

観測残差と事前分布残差を連結して返す。
scipy.optimize.least_squares に渡す形式。
"""

import numpy as np
from .kinematics import RobotKinematics, DHParams
from .parameters import ParameterSet
from .transforms import ObservationTransform, IdentityTransform
from .observation import vec6_to_se3


def build_kinematic_from_params(
    dh_nominal: list[dict],
    params: ParameterSet,
    param_lookup: dict,
) -> RobotKinematics:
    """
    名目DHパラメータ + 誤差パラメータ（d_alpha, d_a, d_d, d_theta_offset）から
    RobotKinematics を構築する。
    """
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
    params: ParameterSet,
    param_lookup: dict,
    n_joints: int,
) -> np.ndarray:
    """
    関節角度伝達誤差モデルを適用する。

    実際の関節角度 = 指令値 + 周期的誤差：
        q_actual[i] = q[i] + amp[i] * sin(q[i] + phase[i])

    sin(q[i]) の周波数は関節角度に依存するため、
    ギア1回転ごとの周期誤差（1倍角）を基本モデルとする。
    amp[i] が 0 なら影響なし。
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

    x             : 最適化変数ベクトル（free_indices に対応）
    q_traj        : 関節角度時系列 (N, n_joints)
    p_exp         : 実験観測値 (N, 3)
    transform     : 観測変換 T
    q_timestamps  : 制御タイムスタンプ (N,)。指定時は time_offset を考慮した
                    時刻補間を行う（p_exp[i] = FK(q(t[i]+time_offset)) のモデル）
    """
    params.set_vector(free_indices, x)

    kin = build_kinematic_from_params(dh_nominal, params, param_lookup)
    T_tool  = _get_tool_transform(params, param_lookup)
    T_local = _get_local_transform(params, param_lookup)
    n_joints = len(dh_nominal)

    # time_offset を考慮した関節角度補間
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

    # 予測値を計算（伝達誤差補正後の関節角度で FK）
    N = len(q_traj_eff)
    p_pred = np.zeros((N, 3))
    for t in range(N):
        q_eff = apply_transmission_error(q_traj_eff[t], params, param_lookup, n_joints)
        T = kin.forward(q_eff, T_tool, T_local)
        p_pred[t] = T[:3, 3]

    # 変換モードに応じて残差を計算
    #   "split"   : r = T(y_exp) - T(y_pred)  ← Identity / VelocityNorm
    #   "residual": r = T(||p_exp - p_pred||)  ← FFTAmplitude（残差ノルムのFFT）
    if getattr(transform, "transform_mode", "split") == "residual":
        # 位置残差ノルムのスカラー時系列に変換してから T を適用
        r_norm = np.linalg.norm(p_exp - p_pred, axis=1)  # (N,)
        r_obs = transform.apply(r_norm)
    else:
        y_exp_flat  = p_exp.flatten()
        y_pred_flat = p_pred.flatten()
        r_obs = transform.apply(y_exp_flat) - transform.apply(y_pred_flat)

    if not include_prior:
        return r_obs

    r_prior = params.get_prior_residuals(free_indices)
    return np.concatenate([r_obs, r_prior])


def _get_tool_transform(params: ParameterSet, param_lookup: dict) -> np.ndarray:
    v = np.array([
        params.params[param_lookup[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ])
    return vec6_to_se3(v)


def _get_local_transform(params: ParameterSet, param_lookup: dict) -> np.ndarray:
    v = np.array([
        params.params[param_lookup[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ])
    return vec6_to_se3(v)
