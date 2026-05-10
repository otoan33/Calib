"""
外部ライブラリのキネマモデル自体にも同定パラメータがある場合のレシピ。

想定シナリオ:
  - 外部ライブラリが FK を提供し、かつライブラリが受け付けるパラメータが存在する
    （例: 関節角度オフセット = エンコーダゼロ点誤差）
  - それに加えて、ツール座標系・ベース座標系の補正も必要

補正モデル（レイヤー構造）:
    q_eff[i]   = q[i] + q_offset_i           ← 外部ライブラリに渡す補正
    T_fk       = robot.fk(q_eff)             ← 外部ライブラリの FK
    T_pred(q)  = T_base_err @ T_fk @ T_tool_err  ← 我々の補正レイヤー

パラメータ（合計 15 DoF）:
    q_offset_0..5   [group="joint_offset"]  外部ライブラリに渡す関節オフセット [rad]
    tool_tx/ty/tz   [group="tool"]          ツール並進誤差 [m]
    base_tx/ty/tz   [group="base"]          ベース並進誤差 [m]
    base_rx/ry/rz   [group="base"]          ベース回転誤差 [rad]
    （tool_rx/ry/rz は位置観測では不可観測のため固定）

実行:
    python recipe_external_model_mixed.py

出力:
    output/mixed_residuals.png
    output/mixed_trajectory.png
"""

import numpy as np
from pathlib import Path

from robot_calibration.models.base import KinematicModel
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
# 1. 外部ライブラリのキネマクラス
#
#    実際は `import ext_robot_lib; robot = ext_robot_lib.Robot("my_robot")`
#    のようにライブラリをロードする。ここでは動作確認用のスタブを置く。
#
#    ポイント: このライブラリは fk(q, model_params) という API を公開しており、
#    呼び出し側が一部のパラメータ（ここでは関節オフセット）を渡せる。
# ──────────────────────────────────────────────────────────────────────────────

class ExternalRobotFK:
    """外部ライブラリが提供する FK クラスのスタブ。"""

    # 外部ライブラリが受け付けるパラメータ名（ライブラリの仕様として公開済み）
    MODEL_PARAM_KEYS = [f"q_offset_{i}" for i in range(6)]

    def __init__(self, dh_nominal: list[dict]):
        self._dh = dh_nominal

    def fk(self, q: np.ndarray, model_params: dict | None = None) -> np.ndarray:
        """
        外部ライブラリの公開 API。

        model_params に含まれる q_offset_i を関節角度に加算してから FK を評価する。
        ライブラリ内部では URDF や独自パラメータで FK を計算する（不可視）。
        """
        # ---- 外部ライブラリの内部実装（こちらからは不可視）----
        from robot_calibration.models.kinematics import RobotKinematics, DHParams
        dh_list = [
            DHParams(alpha=d["alpha"], a=d["a"], d=d["d"], theta_offset=d["theta_offset"])
            for d in self._dh
        ]
        q_eff = q.copy()
        if model_params:
            for i in range(len(self._dh)):
                q_eff[i] += model_params.get(f"q_offset_{i}", 0.0)
        return RobotKinematics(dh_list).forward(q_eff)
        # ---- ここまで ----


# ──────────────────────────────────────────────────────────────────────────────
# 2. ラッパー: 外部 FK + 我々の補正レイヤーを KinematicModel に統合
# ──────────────────────────────────────────────────────────────────────────────

class ExternalLibKinematics(KinematicModel):
    """
    外部ライブラリの FK に、外部ライブラリ用パラメータと座標系補正を合わせて使うラッパー。

    params 辞書はすべてのパラメータを一元管理する。forward() 内で
    「外部ライブラリに渡すもの」と「我々の補正レイヤーで使うもの」に分割する。

        q_eff      = q + {q_offset_i ← 外部ライブラリへ}
        T_fk       = robot.fk(q_eff)
        T_pred     = T_base_err @ T_fk @ T_tool_err  ← 我々の補正レイヤー
    """

    def __init__(self, robot: ExternalRobotFK):
        self._robot = robot

    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        # --- 外部ライブラリ用パラメータを抽出して fk() に渡す ---
        ext_params = {k: params.get(k, 0.0) for k in ExternalRobotFK.MODEL_PARAM_KEYS}
        T_fk = self._robot.fk(q, ext_params)

        # --- 我々の補正レイヤーを上乗せ ---
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
    # ── 外部ライブラリに渡す関節オフセット ─────────────────────────────────────
    # 観測可能性の整理（UR5 系ロボット・位置のみ観測）:
    #
    #   q_offset_0: ベース z 軸回り → base_rz と完全縮退 → 固定
    #   q_offset_1: ショルダー、TCPへの影響大 (列ノルム ≈10.7) → 可観測
    #   q_offset_2: エルボー、TCPへの影響中 (列ノルム ≈5.8)  → 可観測
    #   q_offset_3: 手首1、TCPへの影響小 (列ノルム ≈1.6)    → SNR 低く固定推奨
    #   q_offset_4: 手首2、TCPへの影響小 (列ノルム ≈1.2)    → SNR 低く固定推奨
    #   q_offset_5: 手首 z 軸回り → 位置観測では不可観測 → 固定
    #
    # prior_std は観測ノイズに対して十分に非情報的な値を設定する（単位: rad）。
    Parameter("q_offset_0", value=0.0, group="joint_offset", fixed=True),
    Parameter("q_offset_1", value=0.0, group="joint_offset", prior_std=1.0),
    Parameter("q_offset_2", value=0.0, group="joint_offset", prior_std=1.0),
    Parameter("q_offset_3", value=0.0, group="joint_offset", fixed=True),
    Parameter("q_offset_4", value=0.0, group="joint_offset", fixed=True),
    Parameter("q_offset_5", value=0.0, group="joint_offset", fixed=True),

    # ── 我々の補正レイヤー: ツール変換 ──────────────────────────────────────────
    # 位置観測では T_tool.R は TCP 位置に現れないため回転 3 成分は固定。
    Parameter("tool_tx", value=0.0, group="tool", prior_std=1.0),
    Parameter("tool_ty", value=0.0, group="tool", prior_std=1.0),
    Parameter("tool_tz", value=0.0, group="tool", prior_std=1.0),
    Parameter("tool_rx", value=0.0, group="tool", fixed=True),
    Parameter("tool_ry", value=0.0, group="tool", fixed=True),
    Parameter("tool_rz", value=0.0, group="tool", fixed=True),

    # ── 我々の補正レイヤー: ベース座標系 ────────────────────────────────────────
    Parameter("base_tx", value=0.0, group="base", prior_std=1.0),
    Parameter("base_ty", value=0.0, group="base", prior_std=1.0),
    Parameter("base_tz", value=0.0, group="base", prior_std=1.0),
    Parameter("base_rx", value=0.0, group="base", prior_std=np.pi),
    Parameter("base_ry", value=0.0, group="base", prior_std=np.pi),
    Parameter("base_rz", value=0.0, group="base", prior_std=np.pi),
]

# ──────────────────────────────────────────────────────────────────────────────
# 5. 推定ステージ
#
#    全パラメータ（外部ライブラリ用 + 我々の補正レイヤー）を単一ステージで同時推定。
#    条件数解析により 13 パラメータすべてが観測可能（条件数 ≈ 22）なので
#    段階的推定は不要。分離不能な縮退の整理は PARAMETERS 側の fixed=True で行う。
# ──────────────────────────────────────────────────────────────────────────────

STAGES = [
    Stage(
        name="all_params",
        param_groups=None,   # None = 全 free パラメータを対象
        transform=IdentityTransform(),
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# 6. 実行
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NOISE_STD = 2e-4   # 0.2 mm

    # --- 真値誤差 ----------------------------------------------------------------
    TRUE_ERRORS = {
        # 外部ライブラリ側のパラメータ（エンコーダゼロ点ずれ）
        "q_offset_1":  np.deg2rad(0.4),    # ショルダー: +0.4°
        "q_offset_2":  np.deg2rad(-0.3),   # エルボー: -0.3°
        # 我々の補正レイヤー
        "tool_tx":     0.003,              # ツール先端: +3 mm
        "tool_ty":    -0.002,              # ツール先端: -2 mm
        "base_tx":    -0.004,              # ベース原点: -4 mm
        "base_ry":     np.deg2rad(0.3),    # ベース傾き: +0.3°
    }

    # --- データ生成（実機では CSV 読み込みに差し替える）--------------------------
    robot = ExternalRobotFK(DH_NOMINAL)
    kin   = ExternalLibKinematics(robot)
    obs   = PoseObservation()

    np.random.seed(0)
    N = 500
    # 関節オフセット（q_offset_1/2）と base_ry の相関を断ち切るには、
    # 関節 0 を全周 [-π, π] で動かすことが重要。
    # 半周 [-π/2, π/2] では両者が混同しやすくなる。
    q_traj = np.random.uniform(
        low  = [-np.pi,   -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi],
        high = [ np.pi,    np.pi/3,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi],
        size = (N, 6),
    )
    y_exp = np.array([
        kin.forward(q_traj[i], TRUE_ERRORS)[:3, 3]
        for i in range(N)
    ]) + np.random.normal(0, NOISE_STD, (N, 3))

    # --- 同定（段階的最適化 + 最終全パラメータ調整）----------------------------
    params_result, stage_results = run_calibration(
        q_traj=q_traj,
        y_exp=y_exp,
        parameters=PARAMETERS,
        kinematic_model=kin,
        observation_model=obs,
        stages=STAGES,
        ls_kwargs={
            "method": "trf",     # trust-region: 条件数が大きい問題でも安定
            "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12,
            "x_scale": "jac",    # パラメータスケールを自動調整（単位が混在するため重要）
            "max_nfev": 50000,
        },
    )

    # --- Before / After 予測 ---------------------------------------------------
    nominal_dict = {p.name: 0.0 for p in params_result.params}
    calib_dict   = {p.name: p.value for p in params_result.params}

    p_before = np.array([obs.predict(kin.forward(q_traj[i], nominal_dict), {}) for i in range(N)])
    p_after  = np.array([obs.predict(kin.forward(q_traj[i], calib_dict),  {}) for i in range(N)])

    rms_b = np.sqrt(np.mean(np.linalg.norm(y_exp - p_before, axis=1)**2)) * 1e3
    rms_a = np.sqrt(np.mean(np.linalg.norm(y_exp - p_after,  axis=1)**2)) * 1e3
    print(f"\nRMS  Before: {rms_b:.3f} mm  →  After: {rms_a:.3f} mm")

    # --- 真値との比較 -----------------------------------------------------------
    print("\nTrue vs estimated:")
    lk = {p.name: i for i, p in enumerate(params_result.params)}
    for name, truth in TRUE_ERRORS.items():
        est  = params_result.params[lk[name]].value
        diff = abs(est - truth)
        unit = "rad" if name.startswith(("q_", "base_r")) else "m"
        print(f"  {name:14s}  true={truth:+.5f} {unit}  est={est:+.5f} {unit}  diff={diff:.2e}")

    # --- 不確かさ評価 -----------------------------------------------------------
    uncertainty = compute_uncertainty(
        params=params_result,
        kin_model=kin,
        obs_model=obs,
        q_traj=q_traj,
        y_exp=y_exp,
    )
    print("\n" + uncertainty.summary())

    # --- グラフ出力 -------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig_r = plot_residuals(
        r_before=np.linalg.norm(y_exp - p_before, axis=1),
        r_after =np.linalg.norm(y_exp - p_after,  axis=1),
        title="Mixed calibration  residuals [m]",
    )
    fig_r.savefig(out_dir / "mixed_residuals.png", dpi=150)

    fig_t = plot_trajectory_comparison(y_exp, p_before, p_after)
    fig_t.savefig(out_dir / "mixed_trajectory.png", dpi=150)

    print(f"\nSaved: {out_dir}/mixed_*.png")
