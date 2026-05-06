"""
ボールバー（1点距離計測）によるキャリブレーション例。

距離スカラーのみからロボット DH 誤差・ツール・ローカル誤差を推定する。
Laplace 近似の標準偏差 σ を可視化することで、スカラー観測での識別可能性の
限界（σ が大きいパラメータ = 距離測定だけでは特定困難）を示す。

実行:
    python recipe_ballbar.py

出力:
    output/ballbar_summary.png  ← 残差・観測フィット・パラメータ識別可能性
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from robot_calibration.models.defaults import DHKinematics, DistanceObservation
from robot_calibration.models.parameters import Parameter
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.models.transforms import IdentityTransform
from robot_calibration.pipeline import run_calibration, compute_uncertainty

# ── ロボット定義 ──────────────────────────────────────────────────────────────
DH_NOMINAL = [
    {"alpha": 0.0,      "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi/2,  "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,      "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,      "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi/2,  "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi/2, "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]

# ボールバー固定端（計測原点）[m]。ロボットの作業空間内に配置する
ORIGIN = np.array([0.50, 0.00, 0.30])

KIN = DHKinematics(DH_NOMINAL)
OBS = DistanceObservation(origin=ORIGIN)

# ── パラメータ定義 ────────────────────────────────────────────────────────────
# tool_rz は距離観測では d_theta_offset_5 と完全縮退するため固定
PARAMETERS = [
    *[Parameter(f"d_alpha_{i}",        value=0.0, group="kinematic", prior_std=np.pi)
      for i in range(6)],
    *[Parameter(f"d_a_{i}",            value=0.0, group="kinematic", prior_std=1.0)
      for i in range(6)],
    *[Parameter(f"d_d_{i}",            value=0.0, group="kinematic", prior_std=1.0)
      for i in range(6)],
    *[Parameter(f"d_theta_offset_{i}", value=0.0, group="kinematic", prior_std=np.pi)
      for i in range(6)],
    Parameter("tool_tx", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_ty", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_tz", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_rx", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_ry", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_rz", value=0.0, group="tool",  prior_std=np.pi, fixed=True),
    Parameter("local_tx", value=0.0, group="local", prior_std=1.0),
    Parameter("local_ty", value=0.0, group="local", prior_std=1.0),
    Parameter("local_tz", value=0.0, group="local", prior_std=1.0),
    Parameter("local_rx", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_ry", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_rz", value=0.0, group="local", prior_std=np.pi),
]

# ── 推定ステージ ──────────────────────────────────────────────────────────────
STAGES = [
    Stage(
        name="full_calibration",
        param_groups=["kinematic", "tool", "local"],
        transform=IdentityTransform(),
    ),
]


def _plot_ballbar_summary(
    y_measured: np.ndarray,
    r_before: np.ndarray,
    r_after: np.ndarray,
    uncertainty,
    true_errors: dict,
    title: str = "Ballbar Calibration Summary",
) -> plt.Figure:
    """
    ボールバーキャリブレーション専用サマリーグラフ。

    上段: 残差時系列 / ヒストグラム / 計測 vs 予測散布図
    下段: パラメータ σ（対数スケール）— σ が大きいほど識別困難
    """
    rms_b = np.sqrt(np.mean((y_measured - r_before) ** 2)) * 1e3
    rms_a = np.sqrt(np.mean((y_measured - r_after) ** 2)) * 1e3
    err_b = (y_measured - r_before) * 1e3
    err_a = (y_measured - r_after) * 1e3
    colors = {"Before": "#E07B54", "After": "#4C9BE8"}

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35)

    # ── 上段左: 残差時系列 ───────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, :2])
    ax_r.plot(err_b, color=colors["Before"], lw=0.7, alpha=0.8,
              label=f"Before  RMS={rms_b:.3f} mm")
    ax_r.plot(err_a, color=colors["After"],  lw=0.7, alpha=0.8,
              label=f"After   RMS={rms_a:.3f} mm")
    ax_r.axhline(0, color="k", lw=0.5, ls="--")
    ax_r.set_xlabel("sample")
    ax_r.set_ylabel("distance residual [mm]")
    ax_r.set_title("Distance residuals (Before vs After)")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.4)

    # ── 上段中: ヒストグラム ─────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, 2])
    ax_h.hist(err_b, bins=35, color=colors["Before"], alpha=0.6, label="Before", density=True)
    ax_h.hist(err_a, bins=35, color=colors["After"],  alpha=0.6, label="After",  density=True)
    ax_h.set_xlabel("residual [mm]")
    ax_h.set_ylabel("density")
    ax_h.set_title("Residual distribution")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.4)

    # ── 上段右: 計測値 vs 予測値 ─────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0, 3])
    y_mm = y_measured * 1e3
    ax_s.scatter(y_mm, r_before * 1e3, s=4, alpha=0.35,
                 color=colors["Before"], label="Before")
    ax_s.scatter(y_mm, r_after  * 1e3, s=4, alpha=0.35,
                 color=colors["After"],  label="After")
    lo = y_mm.min() - 2; hi = y_mm.max() + 2
    ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax_s.set_xlim(lo, hi); ax_s.set_ylim(lo, hi)
    ax_s.set_xlabel("Measured [mm]")
    ax_s.set_ylabel("Predicted [mm]")
    ax_s.set_title("Measured vs Predicted")
    ax_s.legend(fontsize=8)
    ax_s.set_aspect("equal")
    ax_s.grid(True, alpha=0.4)

    # ── 下段: パラメータ σ（識別可能性インジケータ）──────────────────────────
    # σ が noise_std (0.05 mm ≈ 5e-5 m) に近い → 情報なし
    # σ ≪ prior_std → データが制約している
    ax_p = fig.add_subplot(gs[1, :])
    names  = uncertainty.param_names
    stds   = uncertainty.stds
    means  = uncertainty.means
    priors = {p.name: p.prior_std for p in PARAMETERS if not p.fixed}
    x = np.arange(len(names))

    # 推定値に対してσが10倍以上なら「識別困難」（赤）
    bar_colors = [
        "#E07B54" if stds[i] > 10 * abs(means[i]) + 1e-6 else "#4C9BE8"
        for i in range(len(names))
    ]
    ax_p.bar(x, stds, width=0.55, color=bar_colors, alpha=0.75,
             label="1σ uncertainty")

    if true_errors is not None:
        truth_abs = np.array([abs(true_errors.get(n, 0.0)) for n in names])
        ax_p.scatter(x, truth_abs + 1e-9, marker="x", color="#2CA02C",
                     zorder=5, s=60, linewidths=1.5, label="|True error|")

    ax_p.set_yscale("log")
    ax_p.set_ylim(bottom=1e-7)
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax_p.set_ylabel("σ  (log scale)")
    ax_p.set_title(
        "Parameter 1σ uncertainty  "
        "(red = σ ≫ |estimate|, poorly observable from distance-only data)"
    )
    ax_p.legend(fontsize=8)
    ax_p.grid(True, axis="y", alpha=0.4, which="both")

    return fig


# ── 実行 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NOISE_STD = 5e-5    # 0.05 mm（レーザートラッカー相当）

    TRUE_ERRORS = {
        "d_alpha_0":        np.deg2rad(0.05),
        "d_a_1":            0.0008,
        "d_d_3":            0.0005,
        "d_theta_offset_2": np.deg2rad(0.03),
        "tool_tx":          0.002,
        "tool_ty":         -0.001,
        "local_tx":        -0.003,
        "local_ry":         np.deg2rad(0.2),
    }

    np.random.seed(42)
    N = 500
    q_traj = np.random.uniform(
        low  = [-np.pi/2, -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi],
        high = [ np.pi/2,  np.pi/3,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi],
        size = (N, 6),
    )

    # 距離観測生成: OBS.predict() の (1,) を [0] でスカラー化
    y_exp = np.array([
        OBS.predict(KIN.forward(q_traj[i], TRUE_ERRORS), {})[0]
        for i in range(N)
    ]) + np.random.normal(0, NOISE_STD, N)

    print(f"サンプリング: {N} poses  ノイズ: {NOISE_STD*1e3:.3f} mm")
    print(f"距離範囲: {y_exp.min()*1e3:.1f} ~ {y_exp.max()*1e3:.1f} mm")

    # --- 推定 ----------------------------------------------------------------
    params_result, stage_results = run_calibration(
        q_traj=q_traj,
        y_exp=y_exp.reshape(N, 1),   # (N, 1): pipeline が (N,) にフラット化
        parameters=PARAMETERS,
        kinematic_model=KIN,
        observation_model=OBS,
        stages=STAGES,
    )

    # --- Before（名目パラメータ）・After（推定済み）距離予測 ------------------
    nominal_dict = {p.name: 0.0 for p in params_result.params}
    r_before = np.array([
        OBS.predict(KIN.forward(q_traj[i], nominal_dict), {})[0]
        for i in range(N)
    ])

    calib_dict = {p.name: p.value for p in params_result.params}
    r_after = np.array([
        OBS.predict(KIN.forward(q_traj[i], calib_dict), {})[0]
        for i in range(N)
    ])

    rms_b = np.sqrt(np.mean((y_exp - r_before) ** 2)) * 1e3
    rms_a = np.sqrt(np.mean((y_exp - r_after)  ** 2)) * 1e3
    print(f"\nRMS  Before: {rms_b:.3f} mm  →  After: {rms_a:.3f} mm")

    # --- 結果表示 ------------------------------------------------------------
    print("\n" + params_result.summary())

    print("\nTrue vs estimated (monitored parameters):")
    lk = {p.name: i for i, p in enumerate(params_result.params)}
    for name, truth in TRUE_ERRORS.items():
        if name in lk:
            est = params_result.params[lk[name]].value
            print(f"  {name:30s}  est={est:+.6f}  true={truth:+.6f}  diff={abs(est-truth):.2e}")

    # --- ラプラス近似による不確かさ評価 -------------------------------------
    uncertainty = compute_uncertainty(
        params=params_result,
        kin_model=KIN,
        obs_model=OBS,
        q_traj=q_traj,
        y_exp=y_exp.reshape(N, 1),
    )
    print("\nUncertainty (Laplace approximation):")
    print(uncertainty.summary())

    # --- グラフ出力 ----------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig = _plot_ballbar_summary(
        y_measured=y_exp,
        r_before=r_before,
        r_after=r_after,
        uncertainty=uncertainty,
        true_errors=TRUE_ERRORS,
        title=f"Ballbar Calibration  (Before {rms_b:.2f} mm → After {rms_a:.2f} mm)",
    )
    out_path = out_dir / "ballbar_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
