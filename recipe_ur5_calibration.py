"""
UR5 キャリブレーションレシピ（例）。

このファイルだけ読めば実験内容が全部わかる状態にする。
骨格コード（estimation/, io/, visualization/）は一切触らない。

実行:
    python recipe_ur5_calibration.py

データを用意する場合:
    DATA_DIR に joint_angles.csv (N×6) と tcp_positions.csv (N×3) を置いて
    シミュレーション生成ブロックをロードブロックに差し替える。

出力:
    output/ur5_calibration_summary.png  ← Before/After サマリーグラフ
"""

import numpy as np
from pathlib import Path

from robot_calibration.models.kinematics import DHKinematics
from robot_calibration.models.observation import PoseObservation
from robot_calibration.models.parameters import Parameter
from robot_calibration.models.matrix import IdentityTransform
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.pipeline import run_calibration, compute_uncertainty
from robot_calibration.visualization.plotter import plot_calibration_summary

# ── ロボット定義 ──────────────────────────────────────────────────────────────
DH_NOMINAL = [
    {"alpha": 0.0,        "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi/2,    "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,        "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,        "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi/2,    "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi/2,   "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]

# ── モデル定義 ───────────────────────────────────────────────────────────────
# 標準的な DH FK + 3D 位置観測。差分がなければサブクラス化不要。
KIN = DHKinematics(DH_NOMINAL)
OBS = PoseObservation()

# ── パラメータ定義 ───────────────────────────────────────────────────────────
# 縮退パラメータ:
#   tool_rz ↔ d_theta_offset_5 は位置のみ観測では完全縮退。
#   tool_rz を fixed=True にして d_theta_offset_5 に効果を吸収させる（慣例）。
PARAMETERS = [
    # DH 誤差（6軸 × 4 = 24 個）
    *[Parameter(f"d_alpha_{i}",        value=0.0, group="kinematic", prior_std=np.pi)
      for i in range(6)],
    *[Parameter(f"d_a_{i}",            value=0.0, group="kinematic", prior_std=1.0)
      for i in range(6)],
    *[Parameter(f"d_d_{i}",            value=0.0, group="kinematic", prior_std=1.0)
      for i in range(6)],
    *[Parameter(f"d_theta_offset_{i}", value=0.0, group="kinematic", prior_std=np.pi)
      for i in range(6)],
    # ツール変換誤差（6 個）
    Parameter("tool_tx", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_ty", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_tz", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_rx", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_ry", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_rz", value=0.0, group="tool",  prior_std=np.pi, fixed=True),  # 縮退
    # ローカル座標系誤差（6 個）
    Parameter("local_tx", value=0.0, group="local", prior_std=1.0),
    Parameter("local_ty", value=0.0, group="local", prior_std=1.0),
    Parameter("local_tz", value=0.0, group="local", prior_std=1.0),
    Parameter("local_rx", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_ry", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_rz", value=0.0, group="local", prior_std=np.pi),
]

# ── 推定ステージ ─────────────────────────────────────────────────────────────
STAGES = [
    Stage(
        name="full_calibration",
        param_groups=["kinematic", "tool", "local"],
        transform=IdentityTransform(),
    ),
]

# ── 実行 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NOISE_STD = 2e-4   # 0.2 mm

    # --- データ生成（実機では CSV 読み込みに差し替える） ----------------------
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
    N = 100
    q_traj = np.random.uniform(
        low  = [-np.pi/2, -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi],
        high = [ np.pi/2,  np.pi/3,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi],
        size = (N, 6),
    )
    y_exp = np.array([
        KIN.forward(q_traj[i], TRUE_ERRORS)[:3, 3]
        for i in range(N)
    ]) + np.random.normal(0, NOISE_STD, (N, 3))

    # --- 推定 ----------------------------------------------------------------
    params_result, stage_results = run_calibration(
        q_traj=q_traj,
        y_exp=y_exp,
        parameters=PARAMETERS,
        kinematic_model=KIN,
        observation_model=OBS,
        stages=STAGES,
    )

    # --- Before 予測（名目パラメータ、すべて 0.0）---------------------------
    nominal_dict = {p.name: 0.0 for p in params_result.params}
    p_before = np.array([
        OBS.predict(KIN.forward(q_traj[i], nominal_dict), {})
        for i in range(N)
    ])

    # --- After 予測（推定済みパラメータ）-------------------------------------
    calib_dict = {p.name: p.value for p in params_result.params}
    p_after = np.array([
        OBS.predict(KIN.forward(q_traj[i], calib_dict), {})
        for i in range(N)
    ])

    rms_b = np.sqrt(np.mean(np.linalg.norm(y_exp - p_before, axis=1)**2)) * 1e3
    rms_a = np.sqrt(np.mean(np.linalg.norm(y_exp - p_after,  axis=1)**2)) * 1e3
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
        y_exp=y_exp,
    )
    print("\n" + uncertainty.summary())

    # --- Before/After サマリーグラフ ----------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig = plot_calibration_summary(
        p_exp=y_exp,
        p_pred_before=p_before,
        p_pred_after=p_after,
        uncertainty=uncertainty,
        true_errors=TRUE_ERRORS,
        title=f"UR5 Calibration Summary  (Before {rms_b:.2f} mm → After {rms_a:.2f} mm)",
    )
    out_path = out_dir / "ur5_calibration_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
