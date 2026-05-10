"""
外部ライブラリのキネマモデルを使ったキャリブレーションレシピ。

想定シナリオ:
  - ロボットドライバや別ライブラリが独自の FK を提供している（内部構造は非公開）
  - その FK を信頼しつつ、ツール座標系・ベース座標系のオフセット誤差のみを同定したい

補正モデル:
    T_pred(q) = T_base_err @ T_fk_external(q) @ T_tool_err

パラメータ（12 DoF）:
    tool_tx/ty/tz, tool_rx/ry/rz  -- ツール座標系の位置・姿勢誤差 [m, rad]
    base_tx/ty/tz, base_rx/ry/rz  -- ベース座標系の位置・姿勢誤差 [m, rad]

実行:
    python recipe_external_model.py

データを用意する場合:
    データ生成ブロックを CSV 読み込みに差し替える。
    q_traj : (N, n_joints) 関節角度 [rad]
    y_exp  : (N, 3) 観測位置 [m]

出力:
    output/external_model_result.png
"""

import numpy as np
from pathlib import Path

from robot_calibration.models.base import KinematicModel, ObservationModel
from robot_calibration.models.defaults import PoseObservation
from robot_calibration.models.parameters import Parameter
from robot_calibration.models.observation import vec6_to_se3
from robot_calibration.models.transforms import IdentityTransform
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.pipeline import run_calibration, compute_uncertainty
from robot_calibration.visualization.plotter import (
    plot_residuals, plot_trajectory_comparison,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 外部ライブラリのキネマクラス（サードパーティ提供 API を想定）
#
#    実際は `import ext_robot_lib; robot = ext_robot_lib.Robot("my_robot")`
#    のようにライブラリを呼ぶ。ここでは動作確認用にスタブを定義する。
# ──────────────────────────────────────────────────────────────────────────────

class ExternalRobotFK:
    """
    外部ライブラリが提供するロボット FK クラスのスタブ。

    実際の外部ライブラリでは、このクラスは URDF やメーカー提供の
    パラメータを内部に持ち、fk() メソッドで SE(3) を返す。
    """

    def __init__(self, dh_nominal: list[dict]):
        self._dh = dh_nominal

    def fk(self, q: np.ndarray) -> np.ndarray:
        """関節角度から手先姿勢 T ∈ SE(3) を返す（外部ライブラリの公開 API）。"""
        # ---- ここが外部ライブラリの内部実装（こちらからは不可視） ----
        from robot_calibration.models.kinematics import RobotKinematics, DHParams
        dh_list = [
            DHParams(alpha=d["alpha"], a=d["a"], d=d["d"], theta_offset=d["theta_offset"])
            for d in self._dh
        ]
        return RobotKinematics(dh_list).forward(q)
        # ---- ここまで ----


# ──────────────────────────────────────────────────────────────────────────────
# 2. ラッパー: 外部 FK を KinematicModel に適合させる
# ──────────────────────────────────────────────────────────────────────────────

class ExternalLibKinematics(KinematicModel):
    """
    外部ライブラリの FK に座標系補正を上乗せするラッパー。

        T_pred = T_base_err @ robot.fk(q) @ T_tool_err

    外部ライブラリが解析的ヤコビアンを提供しないため、
    jacobian() は数値微分で実装する。
    """

    def __init__(self, robot: ExternalRobotFK):
        self._robot = robot

    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        T_fk = self._robot.fk(q)
        T_tool = vec6_to_se3(np.array([
            params.get("tool_tx", 0.0), params.get("tool_ty", 0.0), params.get("tool_tz", 0.0),
            params.get("tool_rx", 0.0), params.get("tool_ry", 0.0), params.get("tool_rz", 0.0),
        ]))
        T_base = vec6_to_se3(np.array([
            params.get("base_tx", 0.0), params.get("base_ty", 0.0), params.get("base_tz", 0.0),
            params.get("base_rx", 0.0), params.get("base_ry", 0.0), params.get("base_rz", 0.0),
        ]))
        return T_base @ T_fk @ T_tool

    def jacobian(self, q: np.ndarray, params: dict) -> np.ndarray:
        return self.numerical_jacobian(q, params)


# ──────────────────────────────────────────────────────────────────────────────
# 3. ロボット定義
# ──────────────────────────────────────────────────────────────────────────────

DH_NOMINAL = [
    {"alpha": 0.0,       "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi/2,   "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,       "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,       "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi/2,   "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi/2,  "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]

# ──────────────────────────────────────────────────────────────────────────────
# 4. パラメータ定義
# ──────────────────────────────────────────────────────────────────────────────

PARAMETERS = [
    # ツール変換誤差 -- 位置のみ観測では TCP 位置に T_tool.R は現れないため
    # 回転 3 成分は固定（観測不可能。姿勢観測モデルに変えれば推定可能）。
    Parameter("tool_tx", value=0.0, group="tool", prior_std=1.0),
    Parameter("tool_ty", value=0.0, group="tool", prior_std=1.0),
    Parameter("tool_tz", value=0.0, group="tool", prior_std=1.0),
    Parameter("tool_rx", value=0.0, group="tool", fixed=True),
    Parameter("tool_ry", value=0.0, group="tool", fixed=True),
    Parameter("tool_rz", value=0.0, group="tool", fixed=True),
    # ベース座標系誤差 -- 並進・回転とも観測可能
    Parameter("base_tx", value=0.0, group="base", prior_std=1.0),
    Parameter("base_ty", value=0.0, group="base", prior_std=1.0),
    Parameter("base_tz", value=0.0, group="base", prior_std=1.0),
    Parameter("base_rx", value=0.0, group="base", prior_std=np.pi),
    Parameter("base_ry", value=0.0, group="base", prior_std=np.pi),
    Parameter("base_rz", value=0.0, group="base", prior_std=np.pi),
]

# ──────────────────────────────────────────────────────────────────────────────
# 5. 推定ステージ
# ──────────────────────────────────────────────────────────────────────────────

STAGES = [
    Stage(
        name="tool_and_base",
        param_groups=["tool", "base"],
        transform=IdentityTransform(),
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# 6. 実行
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NOISE_STD = 2e-4   # 0.2 mm

    # --- 真値誤差（実機では不明; シミュレーション検証用）---------------------
    TRUE_ERRORS = {
        "tool_tx":  0.003,
        "tool_ty": -0.002,
        # tool_rz は位置観測では観測不可能のため真値に含めない
        "base_tx": -0.004,
        "base_ry":  np.deg2rad(0.3),
    }

    # --- データ生成（実機では CSV 読み込みに差し替える）----------------------
    robot = ExternalRobotFK(DH_NOMINAL)
    kin   = ExternalLibKinematics(robot)
    obs   = PoseObservation()

    np.random.seed(0)
    N = 120
    q_traj = np.random.uniform(
        low  = [-np.pi/2, -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi],
        high = [ np.pi/2,  np.pi/3,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi],
        size = (N, 6),
    )
    y_exp = np.array([
        kin.forward(q_traj[i], TRUE_ERRORS)[:3, 3]
        for i in range(N)
    ]) + np.random.normal(0, NOISE_STD, (N, 3))

    # --- 同定 ----------------------------------------------------------------
    params_result, stage_results = run_calibration(
        q_traj=q_traj,
        y_exp=y_exp,
        parameters=PARAMETERS,
        kinematic_model=kin,
        observation_model=obs,
        stages=STAGES,
    )

    # --- Before / After 予測 -------------------------------------------------
    nominal_dict = {p.name: 0.0 for p in params_result.params}
    calib_dict   = {p.name: p.value for p in params_result.params}

    p_before = np.array([obs.predict(kin.forward(q_traj[i], nominal_dict), {}) for i in range(N)])
    p_after  = np.array([obs.predict(kin.forward(q_traj[i], calib_dict),  {}) for i in range(N)])

    rms_b = np.sqrt(np.mean(np.linalg.norm(y_exp - p_before, axis=1)**2)) * 1e3
    rms_a = np.sqrt(np.mean(np.linalg.norm(y_exp - p_after,  axis=1)**2)) * 1e3
    print(f"\nRMS  Before: {rms_b:.3f} mm  →  After: {rms_a:.3f} mm")

    # --- 真値との比較 ---------------------------------------------------------
    print("\nTrue vs estimated:")
    lk = {p.name: i for i, p in enumerate(params_result.params)}
    for name, truth in TRUE_ERRORS.items():
        est  = params_result.params[lk[name]].value
        diff = abs(est - truth)
        print(f"  {name:12s}  true={truth:+.5f}  est={est:+.5f}  diff={diff:.2e}")

    # --- 不確かさ評価 ---------------------------------------------------------
    uncertainty = compute_uncertainty(
        params=params_result,
        kin_model=kin,
        obs_model=obs,
        q_traj=q_traj,
        y_exp=y_exp,
    )
    print("\n" + uncertainty.summary())

    # --- グラフ出力 -----------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig_r = plot_residuals(
        r_before=np.linalg.norm(y_exp - p_before, axis=1),
        r_after =np.linalg.norm(y_exp - p_after,  axis=1),
        title="External model calibration  residuals [m]",
    )
    fig_r.savefig(out_dir / "external_model_residuals.png", dpi=150)

    fig_t = plot_trajectory_comparison(y_exp, p_before, p_after)
    fig_t.savefig(out_dir / "external_model_trajectory.png", dpi=150)

    print(f"\nSaved: {out_dir}/external_model_*.png")
