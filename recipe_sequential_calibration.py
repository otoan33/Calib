"""
逐次キャリブレーションレシピ（データ数に対する収束確認）。

このファイルだけ読めば実験内容が全部わかる状態にする。
骨格コード（estimation/, io/, visualization/）は一切触らない。

実行:
    python recipe_sequential_calibration.py

概要:
    データを N_GROUPS に分割し、累積データ量を増やしながら逐次推定する。
    各ステップで手先位置不確かさ σ_pos を記録し、
    「何点データがあれば σ_pos が目標精度以下になるか」を可視化する。

    update_prior=True（デフォルト）のとき、前ステップの事後分布が
    次ステップの事前分布として引き継がれる（逐次ベイズ推定）。

データを用意する場合:
    DATA_DIR に joint_angles.csv (N×6) と tcp_positions.csv (N×3) を置いて
    「データ生成ブロック」を「データ読み込みブロック」に差し替える。

出力:
    output/sequential_convergence.png      ← 位置不確かさ・RMS 収束グラフ
    output/sequential_final_summary.png    ← 最終推定値の Before/After サマリー
"""

import numpy as np
from pathlib import Path

from robot_calibration.models.kinematics import DHKinematics
from robot_calibration.models.observation import PoseObservation
from robot_calibration.models.parameters import Parameter
from robot_calibration.models.matrix import IdentityTransform
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.pipeline import run_sequential_calibration, run_calibration, compute_uncertainty
from robot_calibration.visualization.plotter import plot_sequential_convergence, plot_calibration_summary

# ── ロボット定義 ──────────────────────────────────────────────────────────────
DH_NOMINAL = [
    {"alpha": 0.0,       "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi/2,   "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,       "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,       "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi/2,   "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi/2,  "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]

# ── モデル ────────────────────────────────────────────────────────────────────
KIN = DHKinematics(DH_NOMINAL)
OBS = PoseObservation()

# ── 推定パラメータ ─────────────────────────────────────────────────────────────
# 縮退: tool_rz ↔ d_theta_offset_5 は位置観測では完全縮退 → tool_rz を固定
PARAMETERS = [
    # DH 誤差（6軸 × 4 = 24 個）
    *[Parameter(f"d_alpha_{i}",        0.0, group="kinematic", prior_std=np.pi) for i in range(6)],
    *[Parameter(f"d_a_{i}",            0.0, group="kinematic", prior_std=1.0)   for i in range(6)],
    *[Parameter(f"d_d_{i}",            0.0, group="kinematic", prior_std=1.0)   for i in range(6)],
    *[Parameter(f"d_theta_offset_{i}", 0.0, group="kinematic", prior_std=np.pi) for i in range(6)],
    # ツール変換誤差（6 個）
    Parameter("tool_tx", 0.0, group="tool", prior_std=1.0),
    Parameter("tool_ty", 0.0, group="tool", prior_std=1.0),
    Parameter("tool_tz", 0.0, group="tool", prior_std=1.0),
    Parameter("tool_rx", 0.0, group="tool", prior_std=np.pi),
    Parameter("tool_ry", 0.0, group="tool", prior_std=np.pi),
    Parameter("tool_rz", 0.0, group="tool", prior_std=np.pi, fixed=True),  # 縮退
    # ベース座標系誤差（6 個）
    Parameter("local_tx", 0.0, group="local", prior_std=1.0),
    Parameter("local_ty", 0.0, group="local", prior_std=1.0),
    Parameter("local_tz", 0.0, group="local", prior_std=1.0),
    Parameter("local_rx", 0.0, group="local", prior_std=np.pi),
    Parameter("local_ry", 0.0, group="local", prior_std=np.pi),
    Parameter("local_rz", 0.0, group="local", prior_std=np.pi),
]

# ── 推定ステージ ─────────────────────────────────────────────────────────────
STAGES = [
    Stage("full_calibration",
          param_groups=["kinematic", "tool", "local"],
          transform=IdentityTransform()),
]

# ── 逐次推定の設定 ────────────────────────────────────────────────────────────
N_GROUPS          = 10       # 分割数（各ステップで N/N_GROUPS ずつデータを追加）
UPDATE_PRIOR      = True     # 前ステップの事後分布を次の事前分布に設定（逐次ベイズ）
SIGMA_POS_TARGET  = 0.1e-3   # 位置不確かさの目標値 [m]（100 μm）

# ── 収束プロットに表示するパラメータ ─────────────────────────────────────────
PARAM_FILTER = [
    "d_a_1",
    "d_d_3",
    "d_theta_offset_2",
    "tool_tx",
    "tool_ty",
    "local_tx",
    "local_ry",
]


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NOISE_STD = 2e-4   # 0.2 mm

    # ── データ生成（実機では CSV 読み込みに差し替える）──────────────────────
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

    np.random.seed(0)
    N = 300
    q_traj = np.random.uniform(
        low  = [-np.pi/2, -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi],
        high = [ np.pi/2,  np.pi/3,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi],
        size = (N, 6),
    )
    poses_true = KIN.forward_batch(q_traj, TRUE_ERRORS)
    y_exp = OBS.predict_batch(poses_true, {}) + np.random.normal(0, NOISE_STD, N * 3)

    # ── 実機データを使う場合はここを差し替える ──────────────────────────────
    # from robot_calibration.io.loader import load_joint_trajectory, load_position_observations
    # _, q_traj = load_joint_trajectory("data/joint_angles.csv")
    # _, y_exp  = load_position_observations("data/tcp_positions.csv")
    # y_exp = y_exp.flatten()
    # TRUE_ERRORS = None   # 実機では真値なし

    # ── 逐次推定 ─────────────────────────────────────────────────────────────
    print(f"逐次推定: N={N} 点を {N_GROUPS} グループに分割")
    print(f"  update_prior={UPDATE_PRIOR}  目標 σ_pos={SIGMA_POS_TARGET*1e3:.3f} mm\n")

    steps = run_sequential_calibration(
        q_traj=q_traj,
        y_exp=y_exp,
        parameters=PARAMETERS,
        kinematic_model=KIN,
        observation_model=OBS,
        stages=STAGES,
        n_groups=N_GROUPS,
        update_prior=UPDATE_PRIOR,
    )

    # ── 収束サマリー表示 ──────────────────────────────────────────────────────
    print("\n─── 収束サマリー ────────────────────────────────────────────────")
    print(f"{'グループ':>6}  {'データ数':>6}  {'σ_pos [mm]':>11}  {'RMS [mm]':>9}")
    first_converged = None
    for s in steps:
        sigma_mm = s.pos_unc_mean * 1e3
        rms_mm   = s.residual_rms * 1e3
        tag = ""
        if first_converged is None and s.pos_unc_mean <= SIGMA_POS_TARGET:
            first_converged = s.n_data
            tag = "  ← 目標達成"
        print(f"  {s.group_idx+1:4d}  {s.n_data:8d}  {sigma_mm:11.4f}  {rms_mm:9.4f}{tag}")

    if first_converged is not None:
        print(f"\n目標 σ_pos={SIGMA_POS_TARGET*1e3:.3f} mm を {first_converged} 点で達成")
    else:
        print(f"\n目標 σ_pos={SIGMA_POS_TARGET*1e3:.3f} mm は {N} 点では未達成")

    # ── 最終ステップの推定値 ─────────────────────────────────────────────────
    final = steps[-1]
    print("\n─── 最終推定値（最終グループ） ──────────────────────────────────")
    for name, val, std in zip(final.param_names, final.param_values, final.param_stds):
        truth_str = ""
        if TRUE_ERRORS is not None and name in TRUE_ERRORS:
            truth_str = f"  true={TRUE_ERRORS[name]:+.6f}  diff={abs(val - TRUE_ERRORS[name]):.2e}"
        print(f"  {name:30s}  est={val:+.6f} ± {std:.6f}{truth_str}")

    # ── 収束グラフ保存 ────────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig = plot_sequential_convergence(
        steps,
        param_filter=PARAM_FILTER,
        true_values=TRUE_ERRORS,
        title=f"Sequential Calibration Convergence  "
              f"(N={N}, noise={NOISE_STD*1e3:.2f} mm, groups={N_GROUPS})",
        save_path=out_dir / "sequential_convergence.png",
    )
    print(f"\nSaved: {out_dir}/sequential_convergence.png")

    # ── 最終ステップの Before/After サマリー ─────────────────────────────────
    # 最終推定値を dict に変換して Before/After 予測を計算
    final_param_dict = dict(zip(final.param_names, final.param_values))
    nominal_dict     = {n: 0.0 for n in final.param_names}

    # full データに対して評価（逐次推定は最終グループ = 全データで推定済み）
    poses_before = KIN.forward_batch(q_traj, nominal_dict)
    poses_after  = KIN.forward_batch(q_traj, final_param_dict)
    p_before = OBS.predict_batch(poses_before, {}).reshape(N, 3)
    p_after  = OBS.predict_batch(poses_after,  {}).reshape(N, 3)
    y_exp_2d = y_exp.reshape(N, 3)

    rms_b = np.sqrt(np.mean(np.linalg.norm(y_exp_2d - p_before, axis=1) ** 2)) * 1e3
    rms_a = np.sqrt(np.mean(np.linalg.norm(y_exp_2d - p_after,  axis=1) ** 2)) * 1e3
    print(f"\nRMS  Before: {rms_b:.3f} mm  →  After: {rms_a:.3f} mm")

    # 最終グループの不確かさを UncertaintyResult として再取得
    from robot_calibration.models.parameters import ParameterSet
    from robot_calibration.estimation.uncertainty import UncertaintyResult
    final_unc = UncertaintyResult(
        param_names=final.param_names,
        means=final.param_values,
        stds=final.param_stds,
        cov=np.diag(final.param_stds ** 2),   # 対角近似（簡易表示用）
    )

    fig2 = plot_calibration_summary(
        p_exp=y_exp_2d,
        p_pred_before=p_before,
        p_pred_after=p_after,
        uncertainty=final_unc,
        true_errors=TRUE_ERRORS,
        title=f"Sequential Calibration Summary  "
              f"(Before {rms_b:.2f} mm → After {rms_a:.2f} mm)",
        save_path=out_dir / "sequential_final_summary.png",
    )
    print(f"Saved: {out_dir}/sequential_final_summary.png")
