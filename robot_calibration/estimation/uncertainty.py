"""
ラプラス近似による不確かさ定量化。

最適解近傍で事後分布をガウスで近似する：
    Cov ≈ (J^T J)^{-1}
    std = sqrt(diag(Cov))
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import OptimizeResult


@dataclass
class UncertaintyResult:
    param_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    cov: np.ndarray

    def summary(self) -> str:
        lines = ["Uncertainty (Laplace approximation):"]
        for name, mean, std in zip(self.param_names, self.means, self.stds):
            lines.append(f"  {name:30s}  {mean:+.6f} ± {std:.6f}")
        return "\n".join(lines)


def laplace_uncertainty(
    jac: np.ndarray,
    param_names: list[str],
    param_values: np.ndarray,
    residuals: np.ndarray | None = None,
) -> UncertaintyResult:
    """
    scipy least_squares が返す Jacobian からラプラス近似で共分散を計算する。

    jac       : (n_residuals, n_params) の Jacobian 行列
    residuals : 最適解での残差ベクトル。与えると σ²=(‖r‖²/(n_obs-n_params)) で
                スケールした正しい Cov_θ = σ² (J^T J)^{-1} を返す。
                None のとき σ²=1 とみなす（正規化済み残差を使う場合）。
    """
    H = jac.T @ jac   # フィッシャー情報行列の近似

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    if residuals is not None:
        n_obs, n_params = jac.shape
        dof = max(n_obs - n_params, 1)
        sigma2 = float(np.dot(residuals, residuals) / dof)
        cov = cov * sigma2

    stds = np.sqrt(np.maximum(np.diag(cov), 0))

    return UncertaintyResult(
        param_names=param_names,
        means=param_values,
        stds=stds,
        cov=cov,
    )


def propagate_to_position(
    cov_theta: np.ndarray,
    J_param: np.ndarray,
    obs_dim: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    パラメータ共分散 Cov_θ を観測空間の位置不確かさに伝播する。

        Cov_y(q) = J(q) @ Cov_θ @ J(q).T
        σ_pos(q) = sqrt( trace(Cov_y(q)) )

    Parameters
    ----------
    cov_theta : (n_params, n_params)  パラメータ共分散行列
    J_param   : (M * obs_dim, n_params)  各評価点でのパラメータヤコビアン
    obs_dim   : 観測次元（PoseObservation=3, DistanceObservation=1 など）

    Returns
    -------
    sigma_pos : (M,)         各評価点でのスカラー位置不確かさ [観測値と同単位]
    sigma_xyz : (M, obs_dim) 各評価点での軸ごとの標準偏差
    """
    M = J_param.shape[0] // obs_dim
    sigma_pos = np.zeros(M)
    sigma_xyz = np.zeros((M, obs_dim))

    for t in range(M):
        Jt = J_param[t * obs_dim : (t + 1) * obs_dim, :]   # (obs_dim, n_params)
        Cov_y = Jt @ cov_theta @ Jt.T                       # (obs_dim, obs_dim)
        diag = np.maximum(np.diag(Cov_y), 0.0)
        sigma_xyz[t] = np.sqrt(diag)
        sigma_pos[t] = np.sqrt(np.trace(Cov_y))

    return sigma_pos, sigma_xyz
