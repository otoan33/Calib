"""
直線軌道の高レート時系列3次元位置データからのキャリブレーション（時刻ずれ推定込み）。

データ想定:
    ロボットコントローラ: 500 Hz 関節角度時系列
    外部計測（レーザートラッカー等）: 同レートで TCP 位置観測（0.2 mm ノイズ）
    ただし計測クロックとコントローラクロックの間に未知の時刻ずれがある。

軌道設計:
    3 開始姿勢 × 3 方向（X/Y/Z） = 9 直線
    各直線は台形速度プロファイル（加速→等速→減速）
    速度が時間変化するため time_offset の同定が可能。

縮退処理:
    tool_rz ↔ d_theta_offset_5 は位置のみ観測で完全縮退 → tool_rz = fixed

推定ステージ:
    Stage 1: time_offset のみ (IdentityTransform)
             位置残差は time_offset に対して単調な勾配を持つため確実に収束する。
    Stage 2: kinematic + tool + local (IdentityTransform)

実行:
    python recipe_straight_line_calib.py

出力:
    output/straight_line_summary.png      ← Before/After サマリー
    output/straight_line_per_line.png     ← 直線ごとの RMS 内訳
    output/straight_line_velocity.png     ← 速度プロファイルと時刻ずれの影響
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

from robot_calibration import (
    run_calibration, compute_uncertainty,
    DHKinematics, PoseObservation, Parameter, Stage,
)
from robot_calibration.models.matrix import IdentityTransform
from robot_calibration.visualization.plotter import plot_calibration_summary

# ── ロボット定義（UR5 近似） ──────────────────────────────────────────────────
DH_NOMINAL = [
    {"alpha": 0.0,       "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi/2,   "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,       "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,       "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi/2,   "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi/2,  "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]

KIN = DHKinematics(DH_NOMINAL)
OBS = PoseObservation()

# ── 計測設定 ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 500    # Hz（高レートデータ想定）
DT            = 1.0 / SAMPLE_RATE
SPEED         = 0.30   # m/s（TCP 最大速度）
LINE_LENGTH   = 0.15   # m
RAMP_FRAC     = 0.30   # 加速・減速区間の割合（各 30%）
BLEND_SECS    = 0.50   # s（ライン間の関節空間ブレンド時間）

# ── パラメータ定義 ────────────────────────────────────────────────────────────
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
    Parameter("tool_rz", value=0.0, group="tool",  prior_std=np.pi, fixed=True),  # 縮退
    Parameter("local_tx", value=0.0, group="local", prior_std=1.0),
    Parameter("local_ty", value=0.0, group="local", prior_std=1.0),
    Parameter("local_tz", value=0.0, group="local", prior_std=1.0),
    Parameter("local_rx", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_ry", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_rz", value=0.0, group="local", prior_std=np.pi),
    # 時刻ずれ（Stage 1 でのみ推定）
    Parameter("time_offset", value=0.0, group="time_offset", prior_std=1.0),
]

STAGES = [
    # Stage 1: time_offset のみ
    # IdentityTransform の位置残差は time_offset に対して単調な勾配 → 確実に収束
    Stage("stage1_time_offset",
          param_groups=["time_offset"],
          transform=IdentityTransform()),
    # Stage 2: time_offset を固定したまま DH / ツール / ベース誤差を推定
    Stage("stage2_kinematics",
          param_groups=["kinematic", "tool", "local"],
          transform=IdentityTransform()),
]


# ── 軌道生成ユーティリティ ───────────────────────────────────────────────────
def _blend_joints(
    q_from: np.ndarray,
    q_to: np.ndarray,
    duration: float = BLEND_SECS,
    sample_rate: float = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    関節空間でのスムーズステップブレンド（始終端で速度ゼロ、C1 連続）。

    s(t) = 3(t/T)^2 - 2(t/T)^3  (smooth step)
    ds/dt|_{t=0} = ds/dt|_{t=T} = 0
    """
    dt = 1.0 / sample_rate
    ts = np.arange(0.0, duration + dt / 2, dt)
    s = 3.0 * (ts / duration)**2 - 2.0 * (ts / duration)**3
    return q_from + np.outer(s, q_to - q_from), ts


def _make_line_traj_timed(
    q_start: np.ndarray,
    direction: np.ndarray,
    length: float = LINE_LENGTH,
    speed: float = SPEED,
    sample_rate: float = SAMPLE_RATE,
    ramp_frac: float = RAMP_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    台形速度プロファイル（加速→等速→減速）で直線移動する関節角度時系列を生成。

    速度プロファイル:
      各区間 ramp_frac × length の距離を加速/減速に使用。
      中間 (1-2*ramp_frac) × length は最大速度で等速移動。

    Returns
    -------
    q_traj  : (N, 6) 関節角度時系列
    ts      : (N,)   タイムスタンプ [s]
    """
    dt = 1.0 / sample_rate
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)

    # 台形の各区間の時間
    t_ramp  = 2.0 * ramp_frac * length / speed        # 加速（または減速）にかかる時間
    t_const = (1.0 - 2.0 * ramp_frac) * length / speed  # 等速区間の時間
    t_total = 2.0 * t_ramp + t_const
    a = speed / t_ramp                                 # 加速度

    def _s(t: float) -> float:
        """時刻 t での経路上の変位 [m]。"""
        if t <= t_ramp:
            return 0.5 * a * t**2
        elif t <= t_ramp + t_const:
            return 0.5 * speed * t_ramp + speed * (t - t_ramp)
        else:
            t_d = t - t_ramp - t_const
            return (0.5 * speed * t_ramp + speed * t_const
                    + speed * t_d - 0.5 * a * t_d**2)

    ts = np.arange(0.0, t_total + dt / 2, dt)
    q = np.array(q_start, dtype=float)
    qs = []
    for i, t in enumerate(ts):
        qs.append(q.copy())
        if i < len(ts) - 1:
            ds = _s(min(ts[i + 1], t_total)) - _s(t)
            J = KIN.jacobian(q, {})          # 名目パラメータで FK ヤコビアン
            q = q + np.linalg.pinv(J) @ (d * max(ds, 0.0))

    return np.array(qs), ts


# ── エントリーポイント ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NOISE_STD = 2e-4        # 0.2 mm（計測ノイズ）
    TRUE_TIME_OFFSET = 0.010  # 真の時刻ずれ 10 ms

    TRUE_ERRORS = {
        "d_alpha_0":         np.deg2rad(0.05),
        "d_a_1":             0.0008,
        "d_d_3":             0.0005,
        "d_theta_offset_2":  np.deg2rad(0.03),
        "tool_tx":           0.002,
        "tool_ty":          -0.001,
        "tool_rz":           np.deg2rad(0.5),   # 縮退パラメータ（固定）
        "local_tx":         -0.003,
        "local_ry":          np.deg2rad(0.2),
        "time_offset":       TRUE_TIME_OFFSET,
    }

    # --- 台形速度プロファイル付き直線軌道を生成 --------------------------------
    HOME_CONFIGS = [
        np.array([ 0.0,     -np.pi/4,  np.pi/2, -np.pi/4, -np.pi/2,  0.0    ]),
        np.array([ np.pi/4, -np.pi/3,  np.pi/3, -np.pi/4, -np.pi/3,  0.0    ]),
        np.array([-np.pi/4, -np.pi/3,  np.pi/3, -np.pi/6, -np.pi/2,  np.pi/4]),
    ]
    DIRECTIONS = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    # ライン間を滑らかな関節空間ブレンドで接続し、C1 連続な軌道を構築する。
    # 停止区間を不連続に挿入すると cubic 補間がスパイクを生じるため、
    # smooth-step 関数で前ラインの終端から次ラインの始端へ補間する。
    all_q, all_t = [], []
    t_cur = 0.0
    line_slices = []   # 各直線の (start_idx, end_idx)

    for q_home in HOME_CONFIGS:
        for d in DIRECTIONS:
            q_line, ts_line = _make_line_traj_timed(q_home, d)

            if all_q:
                # 前ライン終端 → 次ライン始端 をスムーズブレンドで接続
                q_prev_end = np.array(all_q[-1])
                q_blend, ts_blend = _blend_joints(q_prev_end, q_home)
                all_q.extend(q_blend.tolist())
                all_t.extend((t_cur + ts_blend).tolist())
                t_cur = all_t[-1] + DT

            s_idx = len(all_q)
            all_q.extend(q_line.tolist())
            all_t.extend((t_cur + ts_line).tolist())
            line_slices.append((s_idx, len(all_q)))
            t_cur = all_t[-1] + DT

    q_traj       = np.array(all_q)
    q_timestamps = np.array(all_t)
    N = len(q_traj)
    N_LINES = len(line_slices)
    pts_per_line = line_slices[0][1] - line_slices[0][0]
    print(f"サンプリング: {SAMPLE_RATE} Hz  総点数: {N}  直線数: {N_LINES}")
    print(f"各直線: {pts_per_line} pts  総計測時間: {q_timestamps[-1]:.2f} s")

    # --- 観測データ生成（真の時刻ずれ＋DH 誤差＋ノイズ） ---------------------
    # 観測時刻 t で計測されたデータは、実際には t + time_offset での FK 値
    np.random.seed(7)
    t_obs_eff = q_timestamps + TRUE_TIME_OFFSET
    q_obs = np.column_stack([
        interp1d(q_timestamps, q_traj[:, j], kind="cubic",
                 bounds_error=False,
                 fill_value=(q_traj[0, j], q_traj[-1, j]))(t_obs_eff)
        for j in range(6)
    ])
    y_exp = np.array([
        KIN.forward(q_obs[i], TRUE_ERRORS)[:3, 3]
        for i in range(N)
    ]) + np.random.normal(0, NOISE_STD, (N, 3))

    # --- Before 予測（名目パラメータ・時刻ずれ補正なし） ----------------------
    nominal_dict = {p.name: 0.0 for p in PARAMETERS}
    p_before = np.array([
        OBS.predict(KIN.forward(q_traj[i], nominal_dict), {})
        for i in range(N)
    ])

    # --- 推定 ----------------------------------------------------------------
    params_result, stage_results = run_calibration(
        q_traj=q_traj,
        y_exp=y_exp,
        parameters=PARAMETERS,
        kinematic_model=KIN,
        observation_model=OBS,
        stages=STAGES,
        q_timestamps=q_timestamps,
    )

    # --- After 予測（推定済みパラメータ・時刻ずれ補正済み） --------------------
    lk = {p.name: i for i, p in enumerate(params_result.params)}
    calib_dict = {p.name: p.value for p in params_result.params}
    est_time_offset = calib_dict["time_offset"]

    t_after_eff = q_timestamps + est_time_offset
    q_after = np.column_stack([
        interp1d(q_timestamps, q_traj[:, j], kind="cubic",
                 bounds_error=False,
                 fill_value=(q_traj[0, j], q_traj[-1, j]))(t_after_eff)
        for j in range(6)
    ])
    p_after = np.array([
        OBS.predict(KIN.forward(q_after[i], calib_dict), {})
        for i in range(N)
    ])

    rms_b = np.sqrt(np.mean(np.linalg.norm(y_exp - p_before, axis=1)**2)) * 1e3
    rms_a = np.sqrt(np.mean(np.linalg.norm(y_exp - p_after,  axis=1)**2)) * 1e3
    time_err_ms = abs(est_time_offset - TRUE_TIME_OFFSET) * 1e3
    print(f"\ntime_offset: est={est_time_offset*1e3:.3f} ms  true={TRUE_TIME_OFFSET*1e3:.1f} ms"
          f"  誤差={time_err_ms:.3f} ms")
    print(f"RMS  Before: {rms_b:.3f} mm  →  After: {rms_a:.3f} mm")

    # --- 推定結果表示 ---------------------------------------------------------
    print("\n" + params_result.summary())
    print("\nTrue vs estimated (monitored parameters):")
    for name, truth in TRUE_ERRORS.items():
        if name in lk:
            p = params_result.params[lk[name]]
            tag = "  [FIXED]" if p.fixed else ""
            print(f"  {name:30s}  est={p.value:+.6f}  true={truth:+.6f}"
                  f"  diff={abs(p.value-truth):.2e}{tag}")

    # --- ラプラス近似による不確かさ評価 ---------------------------------------
    uncertainty = compute_uncertainty(
        params=params_result,
        kin_model=KIN,
        obs_model=OBS,
        q_traj=q_traj,
        y_exp=y_exp,
        q_timestamps=q_timestamps,
    )
    print("\n" + uncertainty.summary())

    # --- グラフ出力 -----------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 1. Before/After サマリーグラフ
    fig = plot_calibration_summary(
        p_exp=y_exp,
        p_pred_before=p_before,
        p_pred_after=p_after,
        uncertainty=uncertainty,
        true_errors=TRUE_ERRORS,
        title=(f"Straight-line + Time Offset Calibration  "
               f"(Before {rms_b:.2f} mm → After {rms_a:.2f} mm, "
               f"Δt: {TRUE_TIME_OFFSET*1e3:.0f} ms → {est_time_offset*1e3:.2f} ms)"),
    )
    fig.savefig(out_dir / "straight_line_summary.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_dir}/straight_line_summary.png")

    # 2. 直線ごとの RMS 内訳グラフ
    dir_labels  = ["X", "Y", "Z"]
    dir_colors  = ["#4C9BE8", "#E07B54", "#4CAF50"]
    home_labels = ["H1", "H2", "H3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Straight-line Calibration: Per-line Analysis", fontsize=12, fontweight="bold")

    ax = axes[0]
    for li, (s, e) in enumerate(line_slices):
        col = dir_colors[li % 3]
        ax.plot(y_exp[s:e, 0]*1e3, y_exp[s:e, 1]*1e3, ".", ms=2, color="k", alpha=0.2)
        ax.plot(p_before[s:e, 0]*1e3, p_before[s:e, 1]*1e3, "--", lw=1, color=col, alpha=0.4,
                label=f"Before ({dir_labels[li%3]})" if li < 3 else None)
        ax.plot(p_after[s:e, 0]*1e3,  p_after[s:e, 1]*1e3,  "-",  lw=1.5, color=col, alpha=0.9,
                label=f"After ({dir_labels[li%3]})"  if li < 3 else None)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_title("Trajectories in XY plane\n(dots=obs, dashed=before, solid=after)")
    ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(True, alpha=0.4)

    ax2 = axes[1]
    x_labels = [f"{home_labels[i//3]}-{dir_labels[i%3]}" for i in range(N_LINES)]
    rms_b_lines = [
        np.sqrt(np.mean(np.linalg.norm(y_exp[s:e] - p_before[s:e], axis=1)**2)) * 1e3
        for s, e in line_slices
    ]
    rms_a_lines = [
        np.sqrt(np.mean(np.linalg.norm(y_exp[s:e] - p_after[s:e], axis=1)**2)) * 1e3
        for s, e in line_slices
    ]
    xi = np.arange(N_LINES)
    ax2.bar(xi - 0.2, rms_b_lines, 0.38, color="#E07B54", alpha=0.8, label="Before")
    ax2.bar(xi + 0.2, rms_a_lines, 0.38, color="#4C9BE8", alpha=0.8, label="After")
    ax2.axhline(NOISE_STD * 1e3, color="k", ls="--", lw=0.8, label=f"Noise ({NOISE_STD*1e3:.1f} mm)")
    ax2.set_xticks(xi); ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_ylabel("RMS error [mm]"); ax2.set_title("RMS per line")
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "straight_line_per_line.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir}/straight_line_per_line.png")

    # 3. 速度プロファイルと時刻ずれの影響
    # 最初の 1 本の直線のみ表示（説明用）
    s0, e0 = line_slices[0]
    t_line   = q_timestamps[s0:e0] - q_timestamps[s0]
    err_b    = np.linalg.norm(y_exp[s0:e0] - p_before[s0:e0], axis=1) * 1e3
    err_a    = np.linalg.norm(y_exp[s0:e0] - p_after[s0:e0],  axis=1) * 1e3

    # TCP 速度：ノイズのない名目 FK から計算（観測値からだとノイズが乗る）
    p_nom_line = np.array([
        OBS.predict(KIN.forward(q_traj[s0 + i], nominal_dict), {})
        for i in range(e0 - s0)
    ])
    tcp_vel = np.linalg.norm(np.diff(p_nom_line, axis=0), axis=1) / DT * 1e3  # mm/s

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Line #1 (X direction)  —  True Δt = {TRUE_TIME_OFFSET*1e3:.0f} ms, "
                 f"Est Δt = {est_time_offset*1e3:.2f} ms", fontsize=11, fontweight="bold")

    axes[0].plot(t_line[:-1], tcp_vel, color="#4CAF50", lw=1.2)
    axes[0].set_ylabel("TCP speed [mm/s]")
    axes[0].set_title("Velocity profile (trapezoidal: accel → const → decel)")
    axes[0].grid(True, alpha=0.4)
    # 区間境界を示す縦線
    t_ramp_end  = 2.0 * RAMP_FRAC * LINE_LENGTH / SPEED
    t_const_end = t_ramp_end + (1.0 - 2.0 * RAMP_FRAC) * LINE_LENGTH / SPEED
    for ax in axes:
        ax.axvline(t_ramp_end,  color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.axvline(t_const_end, color="gray", ls=":", lw=0.8, alpha=0.7)
    axes[0].annotate("accel", xy=(t_ramp_end/2, SPEED*1e3*0.5),
                     ha="center", color="gray", fontsize=8)
    axes[0].annotate("const", xy=((t_ramp_end+t_const_end)/2, SPEED*1e3*0.9),
                     ha="center", color="gray", fontsize=8)
    axes[0].annotate("decel", xy=((t_const_end+t_line[-1])/2, SPEED*1e3*0.5),
                     ha="center", color="gray", fontsize=8)

    axes[1].plot(t_line, err_b, color="#E07B54", lw=0.8, alpha=0.8,
                 label=f"Before  RMS={np.sqrt(np.mean(err_b**2)):.2f} mm")
    axes[1].plot(t_line, err_a, color="#4C9BE8", lw=0.8, alpha=0.8,
                 label=f"After   RMS={np.sqrt(np.mean(err_a**2)):.2f} mm")
    axes[1].set_ylabel("position error [mm]")
    axes[1].set_title("Position error vs time")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

    # XYZ 残差
    for k, (col, lbl) in enumerate(zip(["#E07B54","#4C9BE8","#4CAF50"], ["X","Y","Z"])):
        axes[2].plot(t_line, (y_exp[s0:e0, k] - p_after[s0:e0, k])*1e3,
                     color=col, lw=0.8, alpha=0.8, label=f"{lbl} (after)")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_xlabel("time [s]"); axes[2].set_ylabel("residual [mm]")
    axes[2].set_title("XYZ residuals after calibration")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "straight_line_velocity.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir}/straight_line_velocity.png")
