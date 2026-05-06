"""
残差・収束・パラメータ結果の描画。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def plot_calibration_summary(
    p_exp: np.ndarray,
    p_pred_before: np.ndarray,
    p_pred_after: np.ndarray,
    uncertainty: UncertaintyResult | None = None,
    true_errors: dict[str, float] | None = None,
    title: str = "Calibration Summary",
) -> plt.Figure:
    """
    Before/After を一枚にまとめたサマリーグラフ。

    上段: XYZ 成分ごとの残差時系列（Before vs After）
    中段: 点ごとの誤差ノルム（Before vs After）＋ RMS 比較棒グラフ
    下段: パラメータ推定値 ± 1σ（真値があれば重ねて表示）

    Parameters
    ----------
    p_exp          : 観測位置 (N, 3) [m]
    p_pred_before  : キャリブレーション前の予測位置 (N, 3) [m]
    p_pred_after   : キャリブレーション後の予測位置 (N, 3) [m]
    uncertainty    : compute_uncertainty() の戻り値（省略可）
    true_errors    : 真値辞書（シミュレーション時のみ）
    """
    has_uncertainty = uncertainty is not None
    n_rows = 3 if has_uncertainty else 2
    fig = plt.figure(figsize=(16, 5 * n_rows), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 4, figure=fig)

    err_b = (p_exp - p_pred_before) * 1e3  # mm
    err_a = (p_exp - p_pred_after)  * 1e3
    norm_b = np.linalg.norm(err_b, axis=1)
    norm_a = np.linalg.norm(err_a, axis=1)
    rms_b = np.sqrt(np.mean(norm_b**2))
    rms_a = np.sqrt(np.mean(norm_a**2))
    labels_xyz = ["X", "Y", "Z"]
    colors = {"Before": "#E07B54", "After": "#4C9BE8"}

    # ── 上段: XYZ 残差時系列 ─────────────────────────────────────────────────
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(err_b[:, k], color=colors["Before"], lw=0.8, label="Before", alpha=0.8)
        ax.plot(err_a[:, k], color=colors["After"],  lw=0.8, label="After",  alpha=0.8)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"{labels_xyz[k]} residual [mm]")
        ax.set_xlabel("sample")
        ax.set_ylabel("error [mm]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)

    # 上段右: 3D scatter（残差の空間分布）
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.scatter(err_b[:, 0], err_b[:, 1], s=10, alpha=0.5, color=colors["Before"], label="Before")
    ax3.scatter(err_a[:, 0], err_a[:, 1], s=10, alpha=0.5, color=colors["After"],  label="After")
    ax3.set_xlabel("X error [mm]")
    ax3.set_ylabel("Y error [mm]")
    ax3.set_title("XY error scatter")
    ax3.legend(fontsize=7)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.4)

    # ── 中段: 誤差ノルム時系列 ＋ RMS 棒グラフ ──────────────────────────────
    ax_norm = fig.add_subplot(gs[1, :3])
    ax_norm.plot(norm_b, color=colors["Before"], lw=0.8, label=f"Before  RMS={rms_b:.2f} mm", alpha=0.8)
    ax_norm.plot(norm_a, color=colors["After"],  lw=0.8, label=f"After   RMS={rms_a:.2f} mm", alpha=0.8)
    ax_norm.set_xlabel("sample")
    ax_norm.set_ylabel("position error [mm]")
    ax_norm.set_title("Position error norm per sample")
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.4)

    ax_rms = fig.add_subplot(gs[1, 3])
    bars = ax_rms.bar(["Before", "After"], [rms_b, rms_a],
                      color=[colors["Before"], colors["After"]], width=0.5)
    for bar, val in zip(bars, [rms_b, rms_a]):
        ax_rms.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f} mm", ha="center", va="bottom", fontsize=9)
    ax_rms.set_ylabel("RMS error [mm]")
    ax_rms.set_title(f"RMS improvement\n{rms_b:.2f} → {rms_a:.2f} mm  ({(1-rms_a/rms_b)*100:.0f}% reduction)")
    ax_rms.grid(True, axis="y", alpha=0.4)

    # ── 下段: パラメータ推定値 ± 1σ ─────────────────────────────────────────
    if has_uncertainty:
        ax_p = fig.add_subplot(gs[2, :])
        names = uncertainty.param_names
        means = uncertainty.means
        stds  = uncertainty.stds
        x = np.arange(len(names))

        ax_p.bar(x, means, 0.5, color="#4C9BE8", alpha=0.7,
                 label="Estimated", yerr=stds, capsize=3, error_kw={"lw": 1.2})

        if true_errors is not None:
            truth = np.array([true_errors.get(n, 0.0) for n in names])
            ax_p.scatter(x, truth, marker="x", color="#E07B54", zorder=5, s=60,
                         linewidths=1.5, label="True")

        ax_p.axhline(0, color="k", lw=0.5, ls="--")
        ax_p.set_xticks(x)
        ax_p.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax_p.set_ylabel("parameter value")
        ax_p.set_title("Estimated parameters ± 1σ  (Laplace approximation)")
        ax_p.legend()
        ax_p.grid(True, axis="y", alpha=0.4)

    return fig
