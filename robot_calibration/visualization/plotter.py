"""
残差・収束・パラメータ結果の描画。
"""

import numpy as np
import matplotlib.pyplot as plt
from ..estimation.uncertainty import UncertaintyResult


def plot_residuals(
    r_before: np.ndarray,
    r_after: np.ndarray,
    title: str = "Residuals before/after calibration",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, r, label in zip(axes, [r_before, r_after], ["Before", "After"]):
        ax.plot(r)
        ax.set_title(f"{label} (RMS={np.sqrt(np.mean(r**2)):.4f})")
        ax.set_xlabel("sample")
        ax.set_ylabel("residual")
        ax.grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_parameter_comparison(
    true_values: dict[str, float],
    estimated: UncertaintyResult,
) -> plt.Figure:
    """真値と推定値を比較するバープロット。シミュレーションテスト用。"""
    names = estimated.param_names
    est   = estimated.means
    stds  = estimated.stds
    truth = np.array([true_values.get(n, 0.0) for n in names])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    ax.bar(x - 0.2, truth, 0.35, label="True",      alpha=0.7)
    ax.bar(x + 0.2, est,   0.35, label="Estimated", alpha=0.7, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("value")
    ax.set_title("True vs Estimated parameters")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_trajectory_comparison(
    p_exp: np.ndarray,
    p_pred_before: np.ndarray,
    p_pred_after: np.ndarray,
) -> plt.Figure:
    """観測軌道と予測軌道の比較（3D + 残差ノルム）。"""
    fig = plt.figure(figsize=(14, 5))

    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.plot(*p_exp.T,          label="Observed",        lw=1.5)
    ax3d.plot(*p_pred_before.T,  label="Pred (before)",   lw=1, ls="--")
    ax3d.plot(*p_pred_after.T,   label="Pred (after)",    lw=1, ls="-.")
    ax3d.set_title("3D trajectory")
    ax3d.legend(fontsize=7)

    for idx, (pred, label) in enumerate(
        [(p_pred_before, "Before"), (p_pred_after, "After")]
    ):
        ax = fig.add_subplot(1, 3, idx + 2)
        err = np.linalg.norm(p_exp - pred, axis=1) * 1e3  # mm
        ax.plot(err)
        ax.set_title(f"{label}  RMS={np.sqrt(np.mean(err**2)):.2f} mm")
        ax.set_xlabel("sample")
        ax.set_ylabel("error [mm]")
        ax.grid(True)

    plt.tight_layout()
    return fig
