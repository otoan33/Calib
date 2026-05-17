"""
統合キャリブレーションパイプライン。

レシピから KinematicModel・ObservationModel・Parameter リスト・Stage リストを
受け取り、段階的最適化を実行して推定済み ParameterSet と結果を返す。

骨格側はモデルインスタンスのメソッド（forward / predict）のみを呼ぶ。
レシピ側のクラス名・内部構造には一切依存しない。
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

from .models.base import KinematicModel, ObservationModel
from .models.parameters import ParameterSet, Parameter
from .models.base import ObservationTransform
from .models.matrix import IdentityTransform
from .estimation.optimizer import Stage, StageResult
from .estimation.uncertainty import laplace_uncertainty, UncertaintyResult


def _params_to_dict(ps: ParameterSet) -> dict:
    return {p.name: p.value for p in ps.params}


def _compute_residuals(
    x: np.ndarray,
    free_indices: list[int],
    params: ParameterSet,
    kin_model: KinematicModel,
    obs_model: ObservationModel,
    q_traj: np.ndarray,
    y_exp: np.ndarray,
    transform: ObservationTransform,
    include_prior: bool = True,
    q_timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """
    残差ベクトルを計算して返す。scipy.optimize.least_squares の fun 引数として使用。

    time_offset パラメータが params に存在し q_timestamps が与えられた場合、
    cubic 補間でシフトした関節角度を使って FK を評価する。
    """
    params.set_vector(free_indices, x)
    pdict = _params_to_dict(params)

    N = len(q_traj)

    # time_offset 補間
    if q_timestamps is not None and "time_offset" in pdict:
        dt = pdict["time_offset"]
        t_eff = q_timestamps + dt
        q_eff = np.column_stack([
            interp1d(
                q_timestamps, q_traj[:, j], kind="cubic",
                bounds_error=False,
                fill_value=(q_traj[0, j], q_traj[-1, j]),
            )(t_eff)
            for j in range(q_traj.shape[1])
        ])
    else:
        q_eff = q_traj

    # 予測値をモデル経由で計算
    preds = []
    for t in range(N):
        pose = kin_model.forward(q_eff[t], pdict)
        preds.append(obs_model.predict(pose, pdict))
    y_pred = np.concatenate(preds)

    # 変換モードで残差計算方法を切り替え
    if getattr(transform, "transform_mode", "split") == "residual":
        obs_dim = len(y_exp) // N
        r_raw = (y_exp - y_pred).reshape(N, obs_dim)
        r_norm = np.linalg.norm(r_raw, axis=1)
        r_obs = transform.apply(r_norm)
    else:
        r_obs = transform.apply(y_exp) - transform.apply(y_pred)

    if not include_prior:
        return r_obs

    r_prior = params.get_prior_residuals(free_indices)
    return np.concatenate([r_obs, r_prior])


def run_calibration(
    q_traj: np.ndarray,
    y_exp: np.ndarray,
    parameters: list[Parameter],
    kinematic_model: KinematicModel,
    observation_model: ObservationModel,
    stages: list[Stage],
    ls_kwargs: dict = None,
    q_timestamps: np.ndarray | None = None,
    final_full_tune: bool = False,
) -> tuple[ParameterSet, list[StageResult]]:
    """
    段階的キャリブレーションを実行する。

    Parameters
    ----------
    q_traj           : 関節角度時系列 (N, n_joints)
    y_exp            : 観測値時系列。(N, obs_dim) または (N*obs_dim,) のフラット配列
    parameters       : Parameter のリスト（レシピで定義）
    kinematic_model  : KinematicModel の実装
    observation_model: ObservationModel の実装
    stages           : Stage のリスト（レシピで定義）
    ls_kwargs        : scipy.optimize.least_squares へのキーワード引数
    q_timestamps     : 制御タイムスタンプ (N,)。time_offset パラメータ推定時に必要
    final_full_tune  : True のとき全ステージ後に全パラメータで最終調整を行う

    Returns
    -------
    params  : 推定済み ParameterSet
    results : 各ステージの StageResult リスト
    """
    if ls_kwargs is None:
        ls_kwargs = {
            "method": "lm",
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-12,
            "max_nfev": 50000,
        }

    params = ParameterSet(parameters)
    y_exp_flat = np.asarray(y_exp).flatten()
    results: list[StageResult] = []

    def _run_stage(stage: Stage) -> StageResult:
        free_idx = params.free_indices(groups=stage.param_groups)
        if not free_idx:
            return StageResult(stage.name, np.array([]), 0.0, True, "no free parameters")

        x0 = params.get_vector(free_idx)
        fun = lambda x: _compute_residuals(
            x, free_idx, params,
            kinematic_model, observation_model,
            q_traj, y_exp_flat, stage.transform,
            include_prior=True,
            q_timestamps=q_timestamps,
        )
        res = least_squares(fun, x0, **ls_kwargs)
        params.set_vector(free_idx, res.x)
        jac = res.jac if hasattr(res, "jac") else None
        return StageResult(stage.name, res.x, res.cost, res.success, res.message, jacobian=jac)

    for stage in stages:
        r = _run_stage(stage)
        results.append(r)
        print(f"[{r.stage_name}] cost={r.cost:.6f}  {r.message}")

    if final_full_tune:
        final = Stage("final_full_tune", param_groups=None, transform=IdentityTransform())
        r = _run_stage(final)
        results.append(r)
        print(f"[final_full_tune] cost={r.cost:.6f}  {r.message}")

    return params, results


def compute_uncertainty(
    params: ParameterSet,
    kin_model: KinematicModel,
    obs_model: ObservationModel,
    q_traj: np.ndarray,
    y_exp: np.ndarray,
    transform: ObservationTransform | None = None,
    q_timestamps: np.ndarray | None = None,
) -> UncertaintyResult:
    """
    キャリブレーション後の ParameterSet に対してラプラス近似で不確かさを計算する。

    最適解で least_squares を再実行してヤコビアン J を取得し、
    Cov ≈ (J^T J)^{-1} から標準偏差を推定する。

    Returns
    -------
    UncertaintyResult  (.param_names, .means, .stds, .cov)
    """
    if transform is None:
        transform = IdentityTransform()

    y_flat = np.asarray(y_exp).flatten()
    free_idx = params.free_indices(groups=None)

    fun = lambda x: _compute_residuals(
        x, free_idx, params, kin_model, obs_model,
        q_traj, y_flat, transform,
        include_prior=False,
        q_timestamps=q_timestamps,
    )
    x0 = params.get_vector(free_idx)
    res = least_squares(fun, x0, method="lm", max_nfev=1)  # 1 ステップで Jacobian だけ取得

    free_names = [params.params[i].name for i in free_idx]
    return laplace_uncertainty(res.jac, free_names, x0)
