"""
統合キャリブレーションパイプライン。

レシピから KinematicModel・ObservationModel・Parameter リスト・Stage リストを
受け取り、段階的最適化を実行して推定済み ParameterSet と結果を返す。

骨格側はモデルインスタンスのメソッド（forward / predict）のみを呼ぶ。
レシピ側のクラス名・内部構造には一切依存しない。
"""

import copy
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

from .models.base import KinematicModel, ObservationModel
from .models.parameters import ParameterSet, Parameter
from .models.base import ObservationTransform
from .models.matrix import IdentityTransform
from .estimation.optimizer import Stage, StageResult
from .estimation.uncertainty import laplace_uncertainty, UncertaintyResult, propagate_to_position


@dataclass
class SequentialStep:
    """run_sequential_calibration の1ステップの結果。"""
    n_data: int                  # このステップで使ったデータ点数
    group_idx: int               # グループ番号（0始まり）
    param_names: list[str]       # free パラメータ名（固定パラメータは含まない）
    param_values: np.ndarray     # 推定値
    param_stds: np.ndarray       # Laplace 近似による 1σ 不確かさ
    residual_rms: float          # 観測残差の RMS（観測値と同じ単位）
    pos_unc_mean: float          # 手先位置不確かさの評価点平均 sqrt(trace(Cov_p)) [m]
    pos_unc_xyz: np.ndarray      # 軸ごとの位置不確かさ平均 (obs_dim,) [m]


def _params_to_dict(ps: ParameterSet) -> dict:
    return {p.name: p.value for p in ps.params}


def _compute_param_jacobian(
    kin_model: KinematicModel,
    obs_model: ObservationModel,
    q_eval: np.ndarray,
    params: ParameterSet,
    free_indices: list[int],
    eps: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """
    パラメータに対する観測値のヤコビアン ∂y(q)/∂θ を数値微分で計算する。

    任意の KinematicModel・ObservationModel に対して動作する。
    forward_batch を使うので N 点分を各パラメータ1回の FK 呼び出しで評価する。

    Returns
    -------
    J        : (M * obs_dim, n_free)  各評価点でのパラメータヤコビアン
    obs_dim  : 観測次元（PoseObservation=3 など）
    """
    pdict0 = _params_to_dict(params)
    poses0 = kin_model.forward_batch(q_eval, pdict0)
    y0 = obs_model.predict_batch(poses0, pdict0)          # (M * obs_dim,)
    obs_dim = len(y0) // len(q_eval)

    J = np.zeros((len(y0), len(free_indices)))
    for j, idx in enumerate(free_indices):
        orig = params.params[idx].value
        params.params[idx].value = orig + eps
        pdict_p = _params_to_dict(params)
        poses_p = kin_model.forward_batch(q_eval, pdict_p)
        y_p = obs_model.predict_batch(poses_p, pdict_p)
        J[:, j] = (y_p - y0) / eps
        params.params[idx].value = orig               # 復元

    return J, obs_dim


def _build_interps(
    q_timestamps: np.ndarray,
    q_traj: np.ndarray,
) -> list:
    """
    関節角度補間器をまとめて生成する（スプライン係数は固定なので1度だけ作る）。

    戻り値: interp1d のリスト（関節数分）
    """
    return [
        interp1d(
            q_timestamps, q_traj[:, j], kind="cubic",
            bounds_error=False,
            fill_value=(q_traj[0, j], q_traj[-1, j]),
        )
        for j in range(q_traj.shape[1])
    ]


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
    interps: list | None = None,
) -> np.ndarray:
    """
    残差ベクトルを計算して返す。scipy.optimize.least_squares の fun 引数として使用。

    interps: _build_interps で事前生成した補間器リスト。
             None のときは q_timestamps から都度生成（互換性のため残す）。
    """
    params.set_vector(free_indices, x)
    pdict = _params_to_dict(params)

    N = len(q_traj)

    # time_offset 補間（補間器はキャッシュ済みのものを使う）
    if q_timestamps is not None and "time_offset" in pdict:
        dt = pdict["time_offset"]
        t_eff = q_timestamps + dt
        _interps = interps or _build_interps(q_timestamps, q_traj)
        q_eff = np.column_stack([f(t_eff) for f in _interps])
    else:
        q_eff = q_traj

    # 予測値をバッチで計算
    poses = kin_model.forward_batch(q_eff, pdict)   # (N, 4, 4)
    y_pred = obs_model.predict_batch(poses, pdict)   # (N * obs_dim,)

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

    # 補間器を1度だけ生成（スプライン係数はデータが変わらない限り再利用できる）
    interps = _build_interps(q_timestamps, q_traj) if q_timestamps is not None else None

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
            interps=interps,
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
    interps = _build_interps(q_timestamps, q_traj) if q_timestamps is not None else None

    fun = lambda x: _compute_residuals(
        x, free_idx, params, kin_model, obs_model,
        q_traj, y_flat, transform,
        include_prior=False,
        q_timestamps=q_timestamps,
        interps=interps,
    )
    x0 = params.get_vector(free_idx)
    res = least_squares(fun, x0, method="lm", max_nfev=1)  # 1 ステップで Jacobian だけ取得

    free_names = [params.params[i].name for i in free_idx]
    return laplace_uncertainty(res.jac, free_names, x0, residuals=res.fun)


def run_sequential_calibration(
    q_traj: np.ndarray,
    y_exp: np.ndarray,
    parameters: list[Parameter],
    kinematic_model: KinematicModel,
    observation_model: ObservationModel,
    stages: list[Stage],
    n_groups: int = 10,
    ls_kwargs: dict = None,
    q_timestamps: np.ndarray | None = None,
    update_prior: bool = True,
    q_eval: np.ndarray | None = None,
    n_eval: int = 100,
) -> list[SequentialStep]:
    """
    データを n_groups に分割し、累積データ量を増やしながら逐次推定する。

    Parameters
    ----------
    n_groups     : 分割数。各ステップで使うデータ数は N/n_groups ずつ増える。
    update_prior : True（デフォルト）のとき、前ステップの事後分布を次の事前分布に設定
                   する（逐次ベイズ推定）。False のとき毎ステップ独立推定（収束比較用）。
    q_eval       : 手先位置不確かさの評価に使う関節角度配列 (M, n_joints)。
                   None のとき訓練データから n_eval 点を等間隔サンプリングして使う。
    n_eval       : q_eval=None のとき使う評価点数（上限）。

    戻り値
    ------
    list[SequentialStep]  n_groups 個。各要素に推定値・不確かさ・RMS・位置不確かさを含む。
    """
    N = len(q_traj)
    y_flat_all = np.asarray(y_exp).flatten()
    obs_dim = len(y_flat_all) // N

    # 手先位置不確かさの評価点（固定。訓練データから等間隔サンプリング）
    if q_eval is None:
        idx_eval = np.round(np.linspace(0, N - 1, min(n_eval, N))).astype(int)
        q_eval_fixed = q_traj[idx_eval]
    else:
        q_eval_fixed = np.asarray(q_eval)

    # 各ステップのデータ終端インデックス（等間隔、最後は端まで）
    edges = np.round(np.linspace(0, N, n_groups + 1)).astype(int)

    working_params = copy.deepcopy(parameters)
    steps: list[SequentialStep] = []

    for g in range(n_groups):
        n_data = int(edges[g + 1])

        q_sub = q_traj[:n_data]
        y_sub = y_flat_all[:n_data * obs_dim]
        ts_sub = q_timestamps[:n_data] if q_timestamps is not None else None

        # 最適化（working_params の value が warm-start 初期値になる）
        params_g, _ = run_calibration(
            q_sub, y_sub, working_params,
            kinematic_model, observation_model, stages,
            ls_kwargs=ls_kwargs, q_timestamps=ts_sub,
        )

        # 不確かさ（Laplace 近似）
        unc = compute_uncertainty(
            params_g, kinematic_model, observation_model,
            q_sub, y_sub, q_timestamps=ts_sub,
        )

        # 残差 RMS
        pdict = _params_to_dict(params_g)
        poses = kinematic_model.forward_batch(q_sub, pdict)
        y_pred = observation_model.predict_batch(poses, pdict)
        rms = float(np.sqrt(np.mean((y_sub - y_pred) ** 2)))

        # パラメータ共分散 → 手先位置不確かさへの伝播
        free_idx = params_g.free_indices()
        J_param, obs_dim_eval = _compute_param_jacobian(
            kinematic_model, observation_model,
            q_eval_fixed, params_g, free_idx,
        )
        sigma_pos, sigma_xyz = propagate_to_position(unc.cov, J_param, obs_dim_eval)
        pos_unc_mean = float(np.mean(sigma_pos))
        pos_unc_xyz  = np.mean(sigma_xyz, axis=0)

        steps.append(SequentialStep(
            n_data=n_data,
            group_idx=g,
            param_names=unc.param_names,
            param_values=unc.means.copy(),
            param_stds=unc.stds.copy(),
            residual_rms=rms,
            pos_unc_mean=pos_unc_mean,
            pos_unc_xyz=pos_unc_xyz,
        ))

        if update_prior:
            val_map = dict(zip(unc.param_names, unc.means))
            std_map = dict(zip(unc.param_names, unc.stds))
            for p in params_g.params:
                if p.name in val_map:
                    p.prior_mean = val_map[p.name]
                    p.prior_std  = max(std_map[p.name], 1e-9)

        working_params = params_g.params

        print(
            f"[group {g+1}/{n_groups}]  n={n_data:5d}"
            f"  RMS={rms*1e3:.3f} mm"
            f"  pos_unc={pos_unc_mean*1e3:.3f} mm"
        )

    return steps
