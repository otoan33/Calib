"""
関節伝達誤差（ギア周期誤差）を含む段階的キャリブレーション例。

伝達誤差モデル:
    q_actual[i] = q[i] + a_i * cos(q[i]) + b_i * sin(q[i])

これは amp * sin(q + phase) と等価（a = amp*sin(phase), b = amp*cos(phase)）。
フーリエ係数 (a, b) に変換することで初期値ゼロでも勾配がゼロにならず、
LM アルゴリズムが確実に収束する。

段階的アプローチ:
    Stage 1: DH + ツール + ローカル誤差を先に推定
    Stage 2/3: 各軸の掃引データで a_i, b_i を推定（DH 残差が小さい状態）
    Stage 4: 全パラメータ同時最終調整（final_full_tune=True）

実行:
    python recipe_transmission_error.py

出力:
    output/trans_err_sweep.png    ← 各軸掃引の位置誤差 vs 関節角（sin パターンの可視化）
    output/trans_err_summary.png  ← Before/After サマリーグラフ
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from robot_calibration.models.kinematics import DHKinematics
from robot_calibration.models.observation import PoseObservation
from robot_calibration.models.parameters import Parameter
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.models.matrix import IdentityTransform
from robot_calibration.pipeline import run_calibration, compute_uncertainty
from robot_calibration.visualization.plotter import plot_calibration_summary

# ── ロボット定義 ──────────────────────────────────────────────────────────────
DH_NOMINAL = [
    {"alpha": 0.0,      "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi/2,  "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,      "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,      "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi/2,  "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi/2, "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]


class DHKinematicsWithTransmissionError(DHKinematics):
    """
    DH 順運動学に関節伝達誤差を追加したサブクラス。

    伝達誤差モデル（フーリエ係数形式）:
        q_actual[i] = q[i] + a_i * cos(q[i]) + b_i * sin(q[i])

    (a, b) はいずれも初期値 0.0 から非ゼロ勾配で収束する線形パラメータ。
    amp/phase 形式への変換: amp = sqrt(a²+b²), phase = atan2(a, b)

    params キー:
        trans_err_a_i  — フーリエ余弦係数（= amp * sin(phase)）
        trans_err_b_i  — フーリエ正弦係数（= amp * cos(phase)）
    """

    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        q_eff = q.copy()
        for i in range(len(q)):
            a = params.get(f"trans_err_a_{i}", 0.0)
            b = params.get(f"trans_err_b_{i}", 0.0)
            q_eff[i] += a * np.cos(q[i]) + b * np.sin(q[i])
        return super().forward(q_eff, params)


KIN = DHKinematicsWithTransmissionError(DH_NOMINAL)
OBS = PoseObservation()

# ── パラメータ定義 ────────────────────────────────────────────────────────────
PARAMETERS = [
    # DH 誤差
    *[Parameter(f"d_alpha_{i}",        value=0.0, group="kinematic", prior_std=np.pi)
      for i in range(6)],
    *[Parameter(f"d_a_{i}",            value=0.0, group="kinematic", prior_std=1.0)
      for i in range(6)],
    *[Parameter(f"d_d_{i}",            value=0.0, group="kinematic", prior_std=1.0)
      for i in range(6)],
    *[Parameter(f"d_theta_offset_{i}", value=0.0, group="kinematic", prior_std=np.pi)
      for i in range(6)],
    # ツール変換誤差
    Parameter("tool_tx", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_ty", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_tz", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_rx", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_ry", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_rz", value=0.0, group="tool",  prior_std=np.pi, fixed=True),
    # ローカル座標系誤差
    Parameter("local_tx", value=0.0, group="local", prior_std=1.0),
    Parameter("local_ty", value=0.0, group="local", prior_std=1.0),
    Parameter("local_tz", value=0.0, group="local", prior_std=1.0),
    Parameter("local_rx", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_ry", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_rz", value=0.0, group="local", prior_std=np.pi),
    # 伝達誤差（フーリエ係数形式 — joint 0, 2 のみ推定）
    Parameter("trans_err_a_0", value=0.0, group="transmission_0", prior_std=0.1),
    Parameter("trans_err_b_0", value=0.0, group="transmission_0", prior_std=0.1),
    Parameter("trans_err_a_2", value=0.0, group="transmission_2", prior_std=0.1),
    Parameter("trans_err_b_2", value=0.0, group="transmission_2", prior_std=0.1),
]

# ── 推定ステージ ──────────────────────────────────────────────────────────────
# DH 誤差を先行推定してから伝達誤差を識別する。
# final_full_tune=True で最終的に全パラメータを同時収束させる。
STAGES = [
    Stage("kinematics",     param_groups=["kinematic", "tool", "local"],
          transform=IdentityTransform()),
    Stage("transmission_0", param_groups=["transmission_0"],
          transform=IdentityTransform()),
    Stage("transmission_2", param_groups=["transmission_2"],
          transform=IdentityTransform()),
]


def _ab_to_amp_phase(a: float, b: float):
    """フーリエ係数 (a, b) を振幅・位相に変換（表示用）。"""
    amp   = np.sqrt(a ** 2 + b ** 2)
    phase = np.arctan2(a, b)
    return amp, phase


def _plot_sweep_summary(
    q_j0: np.ndarray,
    q_j2: np.ndarray,
    err_j0_before: np.ndarray,
    err_j0_after: np.ndarray,
    err_j2_before: np.ndarray,
    err_j2_after: np.ndarray,
    est_a0: float, est_b0: float,
    est_a2: float, est_b2: float,
    true_amp0: float, true_amp2: float,
) -> plt.Figure:
    """
    関節掃引データの位置誤差 vs 関節角グラフ。

    Before（補正なし）では sin パターンが残り、
    After（伝達誤差補正後）ではパターンが消えることを示す。
    """
    colors = {"Before": "#E07B54", "After": "#4C9BE8"}
    est_amp0, est_phase0 = _ab_to_amp_phase(est_a0, est_b0)
    est_amp2, est_phase2 = _ab_to_amp_phase(est_a2, est_b2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transmission Error Identification from Joint Sweeps", fontsize=12,
                 fontweight="bold")

    for ax, q_col, err_b, err_a, jidx, est_a, est_b, true_amp, est_amp, est_phase in [
        (axes[0], q_j0[:, 0], err_j0_before, err_j0_after,
         0, est_a0, est_b0, true_amp0, est_amp0, est_phase0),
        (axes[1], q_j2[:, 2], err_j2_before, err_j2_after,
         2, est_a2, est_b2, true_amp2, est_amp2, est_phase2),
    ]:
        sort_idx = np.argsort(q_col)
        q_deg = np.rad2deg(q_col[sort_idx])

        ax.plot(q_deg, err_b[sort_idx], color=colors["Before"],
                lw=1.0, alpha=0.8, label="Before calibration")
        ax.plot(q_deg, err_a[sort_idx], color=colors["After"],
                lw=1.0, alpha=0.8, label="After calibration")

        # 推定した伝達誤差パターン（関節角度誤差 [mrad] → 目安として表示）
        q_fine = np.linspace(q_col.min(), q_col.max(), 300)
        trans_pattern = (est_a * np.cos(q_fine) + est_b * np.sin(q_fine)) * 1e3
        ax.plot(np.rad2deg(q_fine), trans_pattern, color="purple", lw=1.3, ls="--",
                label=f"Est. Δq joint {jidx} [mrad]  "
                      f"amp={np.rad2deg(est_amp)*1e3:.3f} mdeg")

        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel(f"Joint {jidx} angle [deg]")
        ax.set_ylabel("Position error [mm]")
        ax.set_title(
            f"Joint {jidx} sweep  "
            f"(true amp={np.rad2deg(true_amp)*1e3:.1f} mdeg  "
            f"est amp={np.rad2deg(est_amp)*1e3:.1f} mdeg)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    return fig


# ── 実行 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NOISE_STD = 2e-4   # 0.2 mm

    # フーリエ係数形式で真値を定義: a = amp*sin(phase), b = amp*cos(phase)
    TRUE_AMP_0   = np.deg2rad(0.30)   # 0.30°
    TRUE_PHASE_0 = np.deg2rad(30.0)
    TRUE_AMP_2   = np.deg2rad(0.20)   # 0.20°
    TRUE_PHASE_2 = np.deg2rad(-20.0)

    TRUE_ERRORS = {
        # DH 誤差
        "d_alpha_0":        np.deg2rad(0.05),
        "d_a_1":            0.0008,
        "d_d_3":            0.0005,
        "d_theta_offset_2": np.deg2rad(0.03),
        "tool_tx":          0.002,
        "tool_ty":         -0.001,
        "local_tx":        -0.003,
        "local_ry":         np.deg2rad(0.2),
        # 伝達誤差（フーリエ係数）
        "trans_err_a_0": TRUE_AMP_0 * np.sin(TRUE_PHASE_0),
        "trans_err_b_0": TRUE_AMP_0 * np.cos(TRUE_PHASE_0),
        "trans_err_a_2": TRUE_AMP_2 * np.sin(TRUE_PHASE_2),
        "trans_err_b_2": TRUE_AMP_2 * np.cos(TRUE_PHASE_2),
    }

    np.random.seed(7)

    # --- データ生成 -----------------------------------------------------------
    # joint j を掃引するときの固定姿勢
    Q_FIX = np.array([0.0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0.0])
    N_SWEEP = 240

    q_j0_sweep = np.tile(Q_FIX, (N_SWEEP, 1))
    q_j0_sweep[:, 0] = np.linspace(-np.pi / 2, np.pi / 2, N_SWEEP)

    q_j2_sweep = np.tile(Q_FIX, (N_SWEEP, 1))
    q_j2_sweep[:, 2] = np.linspace(-np.pi / 2, np.pi / 2, N_SWEEP)

    N_RAND = 200
    q_rand = np.random.uniform(
        low  = [-np.pi / 2, -np.pi / 3, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi],
        high = [ np.pi / 2,  np.pi / 3,  np.pi / 2,  np.pi / 2,  np.pi / 2,  np.pi],
        size = (N_RAND, 6),
    )

    # データ結合（各ステージで同一データを使用）
    N_J0, N_J2 = N_SWEEP, N_SWEEP
    q_traj  = np.vstack([q_j0_sweep, q_j2_sweep, q_rand])
    N_TOTAL = len(q_traj)
    idx_j0  = slice(0, N_J0)
    idx_j2  = slice(N_J0, N_J0 + N_J2)

    y_exp = np.array([
        OBS.predict(KIN.forward(q_traj[i], TRUE_ERRORS), {})
        for i in range(N_TOTAL)
    ]) + np.random.normal(0, NOISE_STD, (N_TOTAL, 3))

    print(f"サンプル数: J0={N_J0}  J2={N_J2}  rand={N_RAND}  total={N_TOTAL}")
    print(f"真の伝達誤差: joint0 amp={np.rad2deg(TRUE_AMP_0):.3f}°  "
          f"joint2 amp={np.rad2deg(TRUE_AMP_2):.3f}°")

    # --- 推定 ----------------------------------------------------------------
    params_result, stage_results = run_calibration(
        q_traj=q_traj,
        y_exp=y_exp,
        parameters=PARAMETERS,
        kinematic_model=KIN,
        observation_model=OBS,
        stages=STAGES,
        final_full_tune=True,
    )

    # --- Before・After 予測 --------------------------------------------------
    nominal_dict = {p.name: 0.0 for p in params_result.params}
    p_before = np.array([
        OBS.predict(KIN.forward(q_traj[i], nominal_dict), {})
        for i in range(N_TOTAL)
    ])

    calib_dict = {p.name: p.value for p in params_result.params}
    p_after = np.array([
        OBS.predict(KIN.forward(q_traj[i], calib_dict), {})
        for i in range(N_TOTAL)
    ])

    rms_b = np.sqrt(np.mean(np.linalg.norm(y_exp - p_before, axis=1) ** 2)) * 1e3
    rms_a = np.sqrt(np.mean(np.linalg.norm(y_exp - p_after,  axis=1) ** 2)) * 1e3
    print(f"\nRMS  Before: {rms_b:.3f} mm  ->  After: {rms_a:.3f} mm")

    # --- 結果表示 ------------------------------------------------------------
    print("\n" + params_result.summary())

    # 伝達誤差を amp/phase 形式でも表示
    lk = {p.name: i for i, p in enumerate(params_result.params)}
    def _get(name): return params_result.params[lk[name]].value if name in lk else 0.0

    est_a0 = _get("trans_err_a_0"); est_b0 = _get("trans_err_b_0")
    est_a2 = _get("trans_err_a_2"); est_b2 = _get("trans_err_b_2")
    est_amp0, est_phase0 = _ab_to_amp_phase(est_a0, est_b0)
    est_amp2, est_phase2 = _ab_to_amp_phase(est_a2, est_b2)

    print("\nTransmission error (recovered amp/phase):")
    print(f"  joint 0:  amp={np.rad2deg(est_amp0):.4f}°  phase={np.rad2deg(est_phase0):.2f}°  "
          f"(true amp={np.rad2deg(TRUE_AMP_0):.4f}°  phase={np.rad2deg(TRUE_PHASE_0):.2f}°)")
    print(f"  joint 2:  amp={np.rad2deg(est_amp2):.4f}°  phase={np.rad2deg(est_phase2):.2f}°  "
          f"(true amp={np.rad2deg(TRUE_AMP_2):.4f}°  phase={np.rad2deg(TRUE_PHASE_2):.2f}°)")

    print("\nTrue vs estimated (monitored parameters):")
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
        y_exp=y_exp,
    )
    print("\n" + uncertainty.summary())

    # --- グラフ出力 ----------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    def _pos_err_mm(sl):
        return np.linalg.norm(y_exp[sl] - p_before[sl], axis=1) * 1e3

    def _pos_err_after_mm(sl):
        return np.linalg.norm(y_exp[sl] - p_after[sl], axis=1) * 1e3

    fig_sweep = _plot_sweep_summary(
        q_j0=q_traj[idx_j0],
        q_j2=q_traj[idx_j2],
        err_j0_before=_pos_err_mm(idx_j0),
        err_j0_after =_pos_err_after_mm(idx_j0),
        err_j2_before=_pos_err_mm(idx_j2),
        err_j2_after =_pos_err_after_mm(idx_j2),
        est_a0=est_a0, est_b0=est_b0,
        est_a2=est_a2, est_b2=est_b2,
        true_amp0=TRUE_AMP_0,
        true_amp2=TRUE_AMP_2,
    )

    fig_summary = plot_calibration_summary(
        p_exp=y_exp,
        p_pred_before=p_before,
        p_pred_after=p_after,
        uncertainty=uncertainty,
        true_errors=TRUE_ERRORS,
        title=f"Transmission Error + DH Calibration  "
              f"(Before {rms_b:.2f} mm -> After {rms_a:.2f} mm)",
    )

    path_sweep   = out_dir / "trans_err_sweep.png"
    path_summary = out_dir / "trans_err_summary.png"
    fig_sweep.savefig(path_sweep,   dpi=150)
    fig_summary.savefig(path_summary, dpi=150)
    print(f"\nSaved: {path_sweep}")
    print(f"Saved: {path_summary}")
