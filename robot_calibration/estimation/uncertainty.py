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
) -> UncertaintyResult:
    """
    scipy least_squares が返す Jacobian からラプラス近似で共分散を計算する。

    jac    : (n_residuals, n_params) の Jacobian 行列
    """
    H = jac.T @ jac   # フィッシャー情報行列の近似

    try:
        cov = np.linalg.inv(H)
        stds = np.sqrt(np.maximum(np.diag(cov), 0))  # 数値誤差で負になる場合を除外
    except np.linalg.LinAlgError:
        # ランク落ちの場合は疑似逆行列
        cov = np.linalg.pinv(H)
        stds = np.sqrt(np.maximum(np.diag(cov), 0))

    return UncertaintyResult(
        param_names=param_names,
        means=param_values,
        stds=stds,
        cov=cov,
    )
