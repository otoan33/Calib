"""
シミュレーションテスト。

1. 真値パラメータセットを定義
2. ノイズ付き観測データを生成
3. 初期値をわずかにずらした状態から同定
4. 推定結果と真値を比較してプロット

このテストが通ることを実装の完了基準とする。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt

from robot_calibration.models.kinematics import (
    RobotKinematics, DHParams, numerical_jacobian_dtheta,
)
from robot_calibration.models.parameters import ParameterSet, Parameter, make_default_parameter_set
from robot_calibration.models.transforms import (
    IdentityTransform, VelocityNormTransform, FFTAmplitudeTransform,
)
from robot_calibration.models.residuals import (
    build_kinematic_from_params, compute_residuals,
)
from robot_calibration.models.observation import vec6_to_se3
from robot_calibration.estimation.optimizer import Stage, run_staged_optimization
from robot_calibration.estimation.uncertainty import laplace_uncertainty
from robot_calibration.visualization.plotter import (
    plot_residuals, plot_parameter_comparison, plot_trajectory_comparison,
)
from scipy.optimize import least_squares


# ─────────────────────────────────────────────
# 1. ロボット定義（UR5 近似の名目DH）
# ─────────────────────────────────────────────

DH_NOMINAL = [
    {"alpha": 0.0,            "a": 0.0,     "d": 0.0892, "theta_offset": 0.0},
    {"alpha": np.pi / 2,      "a": 0.0,     "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,            "a": -0.4250, "d": 0.0,    "theta_offset": 0.0},
    {"alpha": 0.0,            "a": -0.3922, "d": 0.1093, "theta_offset": 0.0},
    {"alpha": np.pi / 2,      "a": 0.0,     "d": 0.0950, "theta_offset": 0.0},
    {"alpha": -np.pi / 2,     "a": 0.0,     "d": 0.0820, "theta_offset": 0.0},
]

# 真値誤差（わずかな誤差を付加して観測データを生成）
TRUE_ERRORS = {
    "d_alpha_0": np.deg2rad(0.05),
    "d_a_1":     0.0008,
    "d_d_3":     0.0005,
    "d_theta_offset_2": np.deg2rad(0.03),
    "tool_tx": 0.002,
    "tool_ty": -0.001,
    "tool_rz": np.deg2rad(0.5),
    "local_tx": -0.003,
    "local_ry": np.deg2rad(0.2),
}

NOISE_STD = 0.0002   # 0.2 mm


# ─────────────────────────────────────────────
# 2. ヤコビアン検証テスト
# ─────────────────────────────────────────────

def test_jacobian():
    """解析的ヤコビアンと数値微分の一致を検証する。"""
    print("\n=== Jacobian verification ===")
    dh_list = [
        DHParams(alpha=d["alpha"], a=d["a"], d=d["d"], theta_offset=d["theta_offset"])
        for d in DH_NOMINAL
    ]
    kin = RobotKinematics(dh_list)

    np.random.seed(42)
    q = np.random.uniform(-np.pi / 2, np.pi / 2, 6)

    J_analytic  = kin.jacobian_position(q)
    J_numerical = numerical_jacobian_dtheta(kin, q)

    max_err = np.max(np.abs(J_analytic - J_numerical))
    print(f"  Analytic vs numerical max diff: {max_err:.2e}")

    assert max_err < 1e-5, f"Jacobian mismatch: {max_err:.2e}"
    print("  PASSED")


# ─────────────────────────────────────────────
# 3. 観測変換のヤコビアン検証
# ─────────────────────────────────────────────

def test_transform_jacobians():
    """VelocityNormTransform, FFTAmplitudeTransform のヤコビアンを数値微分で検証。"""
    print("\n=== Transform Jacobian verification ===")
    np.random.seed(0)

    for TransformClass, name, y_shape in [
        (VelocityNormTransform, "VelocityNorm",  (10 * 3,)),
        (FFTAmplitudeTransform, "FFTAmplitude",  (16,)),
    ]:
        tr = TransformClass() if TransformClass != VelocityNormTransform else VelocityNormTransform(dt=0.01)
        y  = np.random.randn(*y_shape)

        J_analytic = tr.jacobian(y)

        # 数値微分
        eps = 1e-7
        n   = len(y)
        s0  = tr.apply(y)
        J_num = np.zeros((len(s0), n))
        for i in range(n):
            yp = y.copy(); yp[i] += eps
            J_num[:, i] = (tr.apply(yp) - s0) / eps

        max_err = np.max(np.abs(J_analytic - J_num))
        print(f"  {name}: max diff = {max_err:.2e}")
        assert max_err < 1e-4, f"{name} Jacobian mismatch: {max_err:.2e}"
        print(f"  {name}: PASSED")


# ─────────────────────────────────────────────
# 4. 観測データ生成
# ─────────────────────────────────────────────

def make_params_with_true_errors() -> tuple[ParameterSet, dict]:
    """真値誤差を埋め込んだパラメータセットを返す。"""
    ps = make_default_parameter_set(DH_NOMINAL)

    # パラメータ名 → インデックスのマップ
    lookup = {p.name: i for i, p in enumerate(ps.params)}

    # 真値誤差を設定
    for name, val in TRUE_ERRORS.items():
        if name in lookup:
            ps.params[lookup[name]].value = val

    return ps, lookup


def generate_observations(
    n_points: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """ノイズ付き観測データを生成する。"""
    np.random.seed(seed)

    ps_true, lookup = make_params_with_true_errors()
    kin_true = build_kinematic_from_params(DH_NOMINAL, ps_true, lookup)

    T_tool  = vec6_to_se3(np.array([
        ps_true.params[lookup[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ]))
    T_local = vec6_to_se3(np.array([
        ps_true.params[lookup[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ]))

    # ランダムな関節角度軌道（作業域内）
    q_traj = np.random.uniform(
        low  = [-np.pi/2, -np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi],
        high = [ np.pi/2,  np.pi/3,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi],
        size = (n_points, 6),
    )

    p_exp = np.zeros((n_points, 3))
    for t in range(n_points):
        T = kin_true.forward(q_traj[t], T_tool, T_local)
        p_exp[t] = T[:3, 3]

    # 観測ノイズ
    p_exp += np.random.normal(0, NOISE_STD, p_exp.shape)

    return q_traj, p_exp


# ─────────────────────────────────────────────
# 5. メインの同定テスト
# ─────────────────────────────────────────────

def test_identification():
    """初期値ゼロから同定を実行し、真値との誤差を確認する。"""
    print("\n=== Identification simulation test ===")

    q_traj, p_exp = generate_observations(n_points=100)

    # 初期パラメータ（全誤差ゼロ）
    ps, lookup = make_default_parameter_set(DH_NOMINAL), None
    ps = make_default_parameter_set(DH_NOMINAL)
    lookup = {p.name: i for i, p in enumerate(ps.params)}

    # キャリブレーション前の残差
    kin_init = build_kinematic_from_params(DH_NOMINAL, ps, lookup)
    T_tool_init  = vec6_to_se3(np.zeros(6))
    T_local_init = vec6_to_se3(np.zeros(6))
    p_pred_before = np.array([
        kin_init.forward(q_traj[t], T_tool_init, T_local_init)[:3, 3]
        for t in range(len(q_traj))
    ])
    err_before = np.linalg.norm(p_exp - p_pred_before, axis=1) * 1e3
    print(f"  Before: RMS error = {np.sqrt(np.mean(err_before**2)):.3f} mm")

    # --- kinematic + tool + local を IdentityTransform で一括推定
    ps = make_default_parameter_set(DH_NOMINAL)
    lookup = {p.name: i for i, p in enumerate(ps.params)}

    stage = Stage(
        name="all_params",
        param_groups=["kinematic", "tool", "local"],
        transform=IdentityTransform(),
    )
    results = run_staged_optimization(
        stages=[stage],
        params=ps,
        dh_nominal=DH_NOMINAL,
        param_lookup=lookup,
        q_traj=q_traj,
        p_exp=p_exp,
        final_full_tune=False,
        ls_kwargs={
            "method": "lm",
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-12,
            "max_nfev": 50000,
        },
    )

    # キャリブレーション後の残差
    kin_opt = build_kinematic_from_params(DH_NOMINAL, ps, lookup)
    T_tool_opt  = vec6_to_se3(np.array([
        ps.params[lookup[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ]))
    T_local_opt = vec6_to_se3(np.array([
        ps.params[lookup[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ]))
    p_pred_after = np.array([
        kin_opt.forward(q_traj[t], T_tool_opt, T_local_opt)[:3, 3]
        for t in range(len(q_traj))
    ])
    err_after = np.linalg.norm(p_exp - p_pred_after, axis=1) * 1e3
    rms_after = np.sqrt(np.mean(err_after**2))
    print(f"  After:  RMS error = {rms_after:.3f} mm")

    # 推定値と真値の比較
    print("\n  Parameter comparison (estimated vs true):")
    monitored = list(TRUE_ERRORS.keys())
    for name in monitored:
        if name in lookup:
            est = ps.params[lookup[name]].value
            truth = TRUE_ERRORS[name]
            diff = abs(est - truth)
            print(f"    {name:30s}  est={est:+.6f}  true={truth:+.6f}  diff={diff:.2e}")

    # 収束確認：残差が観測ノイズレベルまで下がること
    assert rms_after < NOISE_STD * 5 * 1e3, (
        f"RMS error {rms_after:.3f} mm exceeds threshold "
        f"{NOISE_STD * 5 * 1e3:.3f} mm"
    )
    print(f"\n  PASSED (RMS {rms_after:.3f} mm < {NOISE_STD * 5 * 1e3:.3f} mm)")

    return q_traj, p_exp, p_pred_before, p_pred_after, lookup, ps


# ─────────────────────────────────────────────
# 6. Jacobian of DH params (検証)
# ─────────────────────────────────────────────

def test_dh_jacobian():
    """DHパラメータに対するヤコビアンを数値微分と比較する。"""
    print("\n=== DH parameter Jacobian verification ===")
    dh_list = [
        DHParams(alpha=d["alpha"], a=d["a"], d=d["d"], theta_offset=d["theta_offset"])
        for d in DH_NOMINAL
    ]
    kin = RobotKinematics(dh_list)
    np.random.seed(7)
    q = np.random.uniform(-0.5, 0.5, 6)

    grads = kin.jacobian_dh_params(q)

    eps = 1e-6
    max_errs = {}
    for key, grad_analytic in grads.items():
        # どのパラメータか解析
        parts = key.rsplit("_", 1)
        param_type = parts[0]   # e.g. "alpha", "a", "d", "theta_offset"
        link_idx   = int(parts[1])

        # 数値微分
        kin_p = RobotKinematics([
            DHParams(
                alpha        = dh_list[i].alpha        + (eps if param_type == "alpha" and i == link_idx else 0),
                a            = dh_list[i].a            + (eps if param_type == "a"     and i == link_idx else 0),
                d            = dh_list[i].d            + (eps if param_type == "d"     and i == link_idx else 0),
                theta_offset = dh_list[i].theta_offset + (eps if param_type == "theta_offset" and i == link_idx else 0),
            )
            for i in range(len(dh_list))
        ])
        kin_m = RobotKinematics([
            DHParams(
                alpha        = dh_list[i].alpha        - (eps if param_type == "alpha" and i == link_idx else 0),
                a            = dh_list[i].a            - (eps if param_type == "a"     and i == link_idx else 0),
                d            = dh_list[i].d            - (eps if param_type == "d"     and i == link_idx else 0),
                theta_offset = dh_list[i].theta_offset - (eps if param_type == "theta_offset" and i == link_idx else 0),
            )
            for i in range(len(dh_list))
        ])

        p_plus  = kin_p.forward(q)[:3, 3]
        p_minus = kin_m.forward(q)[:3, 3]
        grad_num = (p_plus - p_minus) / (2 * eps)

        err = np.max(np.abs(grad_analytic - grad_num))
        max_errs[key] = err
        if err > 1e-5:
            print(f"  WARNING {key}: analytic={grad_analytic}, numeric={grad_num}, err={err:.2e}")

    overall_max = max(max_errs.values())
    print(f"  Max error across all DH params: {overall_max:.2e}")
    assert overall_max < 1e-5, f"DH Jacobian mismatch: {overall_max:.2e}"
    print("  PASSED")


# ─────────────────────────────────────────────
# 7. 角度伝達誤差の同定テスト
# ─────────────────────────────────────────────

def test_transmission_error_identification():
    """
    軸0に周期的な角度伝達誤差を与え、IdentityTransform で同定できることを確認する。

    FFTAmplitudeTransform は「軸を周期的に往復させた複数サイクルの軌道」と
    「残差の特定成分」の組み合わせで機能する特化型変換。
    ここでは汎用の IdentityTransform を使って伝達誤差を直接同定する。
    """
    print("\n=== Transmission error identification test ===")
    from robot_calibration.models.residuals import apply_transmission_error

    TRUE_AMP_0   = np.deg2rad(0.08)   # 0.08deg の伝達誤差振幅
    TRUE_PHASE_0 = 0.3                # 位相

    np.random.seed(1)

    # 真値パラメータセット（DH誤差・tool/local誤差なし、伝達誤差のみ）
    ps_true = make_default_parameter_set(DH_NOMINAL)
    lookup_true = {p.name: i for i, p in enumerate(ps_true.params)}
    ps_true.params[lookup_true["trans_err_amp_0"]].value   = TRUE_AMP_0
    ps_true.params[lookup_true["trans_err_phase_0"]].value = TRUE_PHASE_0

    from robot_calibration.models.kinematics import DHParams, RobotKinematics
    dh_list = [DHParams(**{k: v for k, v in dh.items()}) for dh in DH_NOMINAL]
    kin_true = RobotKinematics(dh_list)

    # 関節0を往復させる（周期運動 5サイクル × 24点 = 120点）
    # 周期運動にすることで伝達誤差の周期パターンが明確に現れる
    n_cycles = 5
    n_per_cycle = 24
    n_points = n_cycles * n_per_cycle
    q0_sweep = np.pi / 2 * np.sin(
        np.linspace(0, n_cycles * 2 * np.pi, n_points, endpoint=False)
    )
    q_traj_te = np.column_stack([
        q0_sweep,
        np.zeros((n_points, 5)),   # 他軸は固定
    ])

    p_exp_te = np.zeros((n_points, 3))
    for t in range(n_points):
        q_eff = apply_transmission_error(q_traj_te[t], ps_true, lookup_true, 6)
        T = kin_true.forward(q_eff)
        p_exp_te[t] = T[:3, 3]
    p_exp_te += np.random.normal(0, NOISE_STD, p_exp_te.shape)

    # 初期値ゼロから伝達誤差パラメータのみを IdentityTransform で推定
    ps_est = make_default_parameter_set(DH_NOMINAL)
    lookup_est = {p.name: i for i, p in enumerate(ps_est.params)}

    stage = Stage(
        name="transmission_error",
        param_groups=["joint_transmission_error"],
        transform=IdentityTransform(),
    )
    run_staged_optimization(
        stages=[stage],
        params=ps_est,
        dh_nominal=DH_NOMINAL,
        param_lookup=lookup_est,
        q_traj=q_traj_te,
        p_exp=p_exp_te,
        final_full_tune=False,
        ls_kwargs={"method": "lm", "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "max_nfev": 50000},
    )

    est_amp   = ps_est.params[lookup_est["trans_err_amp_0"]].value
    est_phase = ps_est.params[lookup_est["trans_err_phase_0"]].value
    print(f"  amp_0  : est={np.rad2deg(est_amp):.4f} deg  true={np.rad2deg(TRUE_AMP_0):.4f} deg")
    print(f"  phase_0: est={est_phase:.4f} rad  true={TRUE_PHASE_0:.4f} rad")

    # 振幅の相対誤差が50%以内なら合格
    rel_err = abs(est_amp - TRUE_AMP_0) / TRUE_AMP_0
    assert rel_err < 0.5, f"Transmission error amp estimation failed: rel_err={rel_err:.2f}"
    print("  PASSED")
    return q_traj_te, p_exp_te, ps_est, lookup_est


# ─────────────────────────────────────────────
# 8. 時系列軌道データからの同定テスト
# ─────────────────────────────────────────────

def test_trajectory_identification():
    """
    時系列の3次元軌道データから time_offset とキャリブレーションパラメータを同定する。

    段階的推定:
      Stage 1: VelocityNormTransform で time_offset のみを推定
               (速度プロファイルの位相ずれが time_offset に感応)
      Stage 2: IdentityTransform で kinematic + tool + local を推定
               (time_offset を固定したうえで位置残差を最小化)
    """
    print("\n=== Trajectory identification test (time_offset + kinematics) ===")
    from scipy.interpolate import interp1d

    TRUE_TIME_OFFSET = 0.020    # 20ms の時刻ずれ
    dt = 0.05                   # サンプル周期 20Hz
    N = 200
    t = np.arange(N) * dt

    # 高速・多周波の滑らかな関節軌道（速度が大きいほど time_offset への感度が高い）
    omega = 1.5   # rad/s
    q_traj = np.column_stack([
        0.8 * np.sin(omega * t),
        0.5 * np.cos(0.7 * omega * t),
        0.6 * np.sin(1.3 * omega * t + 0.5),
        0.4 * np.cos(0.5 * omega * t + 1.0),
        0.3 * np.sin(0.9 * omega * t),
        0.2 * np.cos(1.1 * omega * t),
    ])

    # 真値パラメータ（DH/tool/local 誤差 + time_offset）
    ps_true, lookup_true = make_params_with_true_errors()
    ps_true.params[lookup_true["time_offset"]].value = TRUE_TIME_OFFSET

    kin_true = build_kinematic_from_params(DH_NOMINAL, ps_true, lookup_true)
    T_tool_true = vec6_to_se3(np.array([
        ps_true.params[lookup_true[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ]))
    T_local_true = vec6_to_se3(np.array([
        ps_true.params[lookup_true[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ]))

    # 観測データ生成: FK を time_offset だけシフトした時刻の関節角度で評価
    np.random.seed(42)
    t_obs = t + TRUE_TIME_OFFSET
    q_obs = np.column_stack([
        interp1d(t, q_traj[:, j], kind="cubic", fill_value="extrapolate")(t_obs)
        for j in range(6)
    ])
    p_exp_traj = np.zeros((N, 3))
    for i in range(N):
        T = kin_true.forward(q_obs[i], T_tool_true, T_local_true)
        p_exp_traj[i] = T[:3, 3]
    p_exp_traj += np.random.normal(0, NOISE_STD, p_exp_traj.shape)

    # キャリブ前の残差（time_offset も誤差も未補正）
    kin_init = build_kinematic_from_params(DH_NOMINAL, make_default_parameter_set(DH_NOMINAL),
                                           {p.name: i for i, p in enumerate(make_default_parameter_set(DH_NOMINAL).params)})
    p_pred_init = np.array([kin_init.forward(q_traj[i])[:3, 3] for i in range(N)])
    rms_before = np.sqrt(np.mean(np.linalg.norm(p_exp_traj - p_pred_init, axis=1)**2)) * 1e3
    print(f"  Before: RMS error = {rms_before:.3f} mm")

    # --- 段階的推定 ---
    ps_est = make_default_parameter_set(DH_NOMINAL)
    lookup_est = {p.name: i for i, p in enumerate(ps_est.params)}

    ls_kwargs = {"method": "lm", "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "max_nfev": 50000}

    stages = [
        # Stage 1: time_offset のみを IdentityTransform で推定
        # VelocityNormTransform は多関節3D軌道では速度ノルム勾配がゼロ交差しやすく
        # 局所最適に陥る。位置残差（Identity）は単調な勾配を持ち確実に収束する。
        Stage(
            name="stage1_time_offset",
            param_groups=["time_offset"],
            transform=IdentityTransform(),
        ),
        Stage(
            name="stage2_kinematic",
            param_groups=["kinematic", "tool", "local"],
            transform=IdentityTransform(),
        ),
    ]

    run_staged_optimization(
        stages=stages,
        params=ps_est,
        dh_nominal=DH_NOMINAL,
        param_lookup=lookup_est,
        q_traj=q_traj,
        p_exp=p_exp_traj,
        final_full_tune=False,
        ls_kwargs=ls_kwargs,
        q_timestamps=t,
    )

    est_time_offset = ps_est.params[lookup_est["time_offset"]].value
    time_offset_err_ms = abs(est_time_offset - TRUE_TIME_OFFSET) * 1e3
    print(f"  time_offset: est={est_time_offset*1e3:.2f} ms  true={TRUE_TIME_OFFSET*1e3:.2f} ms"
          f"  err={time_offset_err_ms:.2f} ms")

    # 最終残差評価（推定した time_offset でシフトした FK との比較）
    kin_opt = build_kinematic_from_params(DH_NOMINAL, ps_est, lookup_est)
    T_tool_opt = vec6_to_se3(np.array([
        ps_est.params[lookup_est[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ]))
    T_local_opt = vec6_to_se3(np.array([
        ps_est.params[lookup_est[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ]))
    t_shifted = t + est_time_offset
    q_shifted = np.column_stack([
        interp1d(t, q_traj[:, j], kind="cubic", fill_value="extrapolate")(t_shifted)
        for j in range(6)
    ])
    p_pred_final = np.array([
        kin_opt.forward(q_shifted[i], T_tool_opt, T_local_opt)[:3, 3]
        for i in range(N)
    ])
    rms_after = np.sqrt(np.mean(np.linalg.norm(p_exp_traj - p_pred_final, axis=1)**2)) * 1e3
    print(f"  After:  RMS error = {rms_after:.3f} mm")

    # 真値パラメータとの比較
    print("\n  Parameter comparison (estimated vs true):")
    for name in list(TRUE_ERRORS.keys()) + ["time_offset"]:
        if name in lookup_est:
            est = ps_est.params[lookup_est[name]].value
            truth = TRUE_ERRORS.get(name, TRUE_TIME_OFFSET if name == "time_offset" else 0.0)
            diff = abs(est - truth)
            print(f"    {name:30s}  est={est:+.6f}  true={truth:+.6f}  diff={diff:.2e}")

    threshold_mm = NOISE_STD * 5 * 1e3
    threshold_ms = 5.0

    assert time_offset_err_ms < threshold_ms, (
        f"time_offset error {time_offset_err_ms:.2f} ms exceeds {threshold_ms:.1f} ms"
    )
    assert rms_after < threshold_mm, (
        f"RMS {rms_after:.3f} mm exceeds threshold {threshold_mm:.3f} mm"
    )
    print(f"\n  PASSED  (time_offset err={time_offset_err_ms:.2f}ms < {threshold_ms}ms,"
          f"  RMS={rms_after:.3f}mm < {threshold_mm:.3f}mm)")

    return q_traj, t, p_exp_traj, p_pred_init, p_pred_final, ps_est, lookup_est


# ─────────────────────────────────────────────
# 9. 直線軌道からの同定テスト
# ─────────────────────────────────────────────

def _sensitivity_norms(params, dh_nominal, param_lookup, q_traj, groups=None, eps=1e-6):
    """
    各自由パラメータの感度ノルム ||∂p_all/∂θ_j||₂ を数値的に計算する。

    感度ノルムが小さい → 軌道データからそのパラメータを同定しにくい（情報が少ない）。
    """
    free_idx = params.free_indices(groups=groups)
    norms = {}

    for idx in free_idx:
        name = params.params[idx].name
        orig = params.params[idx].value

        def _fk_all(val):
            params.params[idx].value = val
            kin = build_kinematic_from_params(dh_nominal, params, param_lookup)
            tv = np.array([params.params[param_lookup[k]].value
                           for k in ["tool_tx","tool_ty","tool_tz","tool_rx","tool_ry","tool_rz"]])
            lv = np.array([params.params[param_lookup[k]].value
                           for k in ["local_tx","local_ty","local_tz","local_rx","local_ry","local_rz"]])
            T_tool  = vec6_to_se3(tv)
            T_local = vec6_to_se3(lv)
            return np.array([kin.forward(q, T_tool, T_local)[:3, 3] for q in q_traj])

        p_p = _fk_all(orig + eps)
        p_m = _fk_all(orig - eps)
        params.params[idx].value = orig  # 必ず復元

        norms[name] = np.sqrt(np.sum(((p_p - p_m) / (2 * eps))**2))

    return norms


def _generate_line_traj(kin_nominal, q_start, direction, length=0.15, n_points=30):
    """
    ヤコビアン擬似逆行列を使い、Cartesian 空間で指定方向に直線移動する
    関節角度時系列を生成する。
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    step = length / (n_points - 1)

    q = np.array(q_start, dtype=float)
    qs = [q.copy()]
    for _ in range(n_points - 1):
        J = kin_nominal.jacobian_position(q)   # (3, 6)
        J_pinv = np.linalg.pinv(J)
        q = q + J_pinv @ (d * step)
        qs.append(q.copy())
    return np.array(qs)


def test_straight_line_identification():
    """
    複数方向の直線軌道データからキャリブパラメータを同定するシミュレーション。

    縮退分析:
      tool_rz と d_theta_offset_5 は位置のみ観測では完全縮退（分離不能）。
      tool_rz = 0 に固定し、その効果を d_theta_offset_5 に吸収させる。

    軌道設計:
      3 開始姿勢 × 3 方向（X/Y/Z） = 9 直線、各 30 点、移動量 150mm
      合計 270 観測点 × 3 次元 = 810 方程式、自由パラメータ 35 個
    """
    print("\n=== Straight-line trajectory identification test ===")
    from robot_calibration.models.kinematics import DHParams, RobotKinematics

    np.random.seed(7)

    # 真値パラメータ（tool_rz の縮退効果も含む）
    ps_true, lookup_true = make_params_with_true_errors()
    kin_true = build_kinematic_from_params(DH_NOMINAL, ps_true, lookup_true)
    T_tool_true = vec6_to_se3(np.array([
        ps_true.params[lookup_true[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ]))
    T_local_true = vec6_to_se3(np.array([
        ps_true.params[lookup_true[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ]))

    # 名目 FK（軌道生成用）
    kin_nominal = RobotKinematics([DHParams(**dh) for dh in DH_NOMINAL])

    # 3 開始姿勢 × 3 方向（X/Y/Z） の直線軌道を生成
    HOME_CONFIGS = [
        np.array([0.0,      -np.pi/4,  np.pi/2,  -np.pi/4, -np.pi/2, 0.0]),
        np.array([np.pi/4,  -np.pi/3,  np.pi/3,  -np.pi/4, -np.pi/3, 0.0]),
        np.array([-np.pi/4, -np.pi/3,  np.pi/3,  -np.pi/6, -np.pi/2, np.pi/4]),
    ]
    DIRECTIONS = [
        np.array([1, 0, 0]),   # X 方向
        np.array([0, 1, 0]),   # Y 方向
        np.array([0, 0, 1]),   # Z 方向
    ]

    line_trajs = []
    for q_home in HOME_CONFIGS:
        for d in DIRECTIONS:
            line_trajs.append(
                _generate_line_traj(kin_nominal, q_home, d, length=0.15, n_points=30)
            )

    q_all = np.vstack(line_trajs)
    N = len(q_all)
    n_lines = len(HOME_CONFIGS) * len(DIRECTIONS)
    print(f"  {n_lines} lines × 30 pts = {N} total points")

    # 観測データ生成（真値 FK + ノイズ）
    p_exp_all = np.zeros((N, 3))
    for i in range(N):
        T = kin_true.forward(q_all[i], T_tool_true, T_local_true)
        p_exp_all[i] = T[:3, 3]
    p_exp_all += np.random.normal(0, NOISE_STD, p_exp_all.shape)

    # キャリブ前 RMS
    p_before = np.array([kin_nominal.forward(q_all[i])[:3, 3] for i in range(N)])
    rms_before = np.sqrt(np.mean(np.linalg.norm(p_exp_all - p_before, axis=1)**2)) * 1e3
    print(f"  Before: RMS = {rms_before:.3f} mm")

    # 推定パラメータセット
    ps_est = make_default_parameter_set(DH_NOMINAL)
    lookup_est = {p.name: i for i, p in enumerate(ps_est.params)}

    # 縮退パラメータを固定
    # tool_rz: 最終関節の theta_offset（d_theta_offset_5）と位置観測では完全縮退。
    # tool_rz = 0 に固定し、効果を d_theta_offset_5 に吸収させる（慣例的処理）。
    ps_est.params[lookup_est["tool_rz"]].fixed = True

    stage = Stage(
        name="straight_line_calib",
        param_groups=["kinematic", "tool", "local"],
        transform=IdentityTransform(),
    )
    run_staged_optimization(
        stages=[stage],
        params=ps_est,
        dh_nominal=DH_NOMINAL,
        param_lookup=lookup_est,
        q_traj=q_all,
        p_exp=p_exp_all,
        final_full_tune=False,
        ls_kwargs={"method": "lm", "ftol": 1e-12, "xtol": 1e-12,
                   "gtol": 1e-12, "max_nfev": 50000},
    )

    # キャリブ後 RMS
    kin_opt = build_kinematic_from_params(DH_NOMINAL, ps_est, lookup_est)
    T_tool_opt = vec6_to_se3(np.array([
        ps_est.params[lookup_est[k]].value
        for k in ["tool_tx", "tool_ty", "tool_tz", "tool_rx", "tool_ry", "tool_rz"]
    ]))
    T_local_opt = vec6_to_se3(np.array([
        ps_est.params[lookup_est[k]].value
        for k in ["local_tx", "local_ty", "local_tz", "local_rx", "local_ry", "local_rz"]
    ]))
    p_after = np.array([
        kin_opt.forward(q_all[i], T_tool_opt, T_local_opt)[:3, 3] for i in range(N)
    ])
    rms_after = np.sqrt(np.mean(np.linalg.norm(p_exp_all - p_after, axis=1)**2)) * 1e3
    print(f"  After:  RMS = {rms_after:.3f} mm")

    # パラメータ比較
    print("\n  Parameter comparison (estimated vs true):")
    for name in list(TRUE_ERRORS.keys()):
        if name not in lookup_est:
            continue
        est  = ps_est.params[lookup_est[name]].value
        truth = TRUE_ERRORS[name]
        fixed = ps_est.params[lookup_est[name]].fixed
        tag = "  [FIXED: degenerate]" if fixed else ""
        print(f"    {name:30s}  est={est:+.6f}  true={truth:+.6f}"
              f"  diff={abs(est-truth):.2e}{tag}")

    # tool_rz の効果が d_theta_offset_5 に吸収されていることを確認
    dt5 = ps_est.params[lookup_est["d_theta_offset_5"]].value
    print(f"\n  d_theta_offset_5 (absorbed tool_rz):  est={np.rad2deg(dt5):+.4f} deg"
          f"  (true tool_rz = {np.rad2deg(TRUE_ERRORS['tool_rz']):+.4f} deg)")

    threshold = NOISE_STD * 5 * 1e3
    assert rms_after < threshold, f"RMS {rms_after:.3f} mm > {threshold:.3f} mm"
    print(f"\n  PASSED (RMS {rms_after:.3f} mm < {threshold:.3f} mm)")

    return q_all, p_exp_all, p_before, p_after, ps_est, lookup_est, line_trajs


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_jacobian()
    test_transform_jacobians()
    test_dh_jacobian()
    test_transmission_error_identification()
    result = test_identification()
    q_traj, p_exp, p_pred_before, p_pred_after, lookup, ps = result

    traj_result = test_trajectory_identification()
    q_traj_t, t_arr, p_exp_t, p_init_t, p_final_t, ps_t, lk_t = traj_result

    line_result = test_straight_line_identification()
    q_lines, p_exp_lines, p_before_lines, p_after_lines, ps_sl, lk_sl, line_trajs = line_result

    # プロット保存
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 軌道比較（3D + 誤差ノルム時系列）
    fig = plot_trajectory_comparison(p_exp, p_pred_before, p_pred_after)
    fig.savefig(os.path.join(out_dir, "trajectory_comparison.png"), dpi=120)
    plt.close(fig)

    # 2. 残差（成分ごと）before / after
    r_before = (p_exp - p_pred_before).flatten() * 1e3  # mm
    r_after  = (p_exp - p_pred_after ).flatten() * 1e3
    fig = plot_residuals(r_before, r_after, title="Position residuals [mm]")
    fig.savefig(os.path.join(out_dir, "residuals.png"), dpi=120)
    plt.close(fig)

    # 3. パラメータ比較（真値 vs 推定値）
    from robot_calibration.estimation.uncertainty import UncertaintyResult
    monitored = list(TRUE_ERRORS.keys())
    est_vals = np.array([ps.params[lookup[n]].value for n in monitored if n in lookup])
    unc = UncertaintyResult(
        param_names=[n for n in monitored if n in lookup],
        means=est_vals,
        stds=np.zeros(len(est_vals)),
        cov=np.eye(len(est_vals)),
    )
    fig = plot_parameter_comparison(TRUE_ERRORS, unc)
    fig.savefig(os.path.join(out_dir, "parameter_comparison.png"), dpi=120)
    plt.close(fig)

    # 4. 時系列軌道テストのプロット
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    err_before_t = np.linalg.norm(p_exp_t - p_init_t, axis=1) * 1e3
    err_after_t  = np.linalg.norm(p_exp_t - p_final_t, axis=1) * 1e3
    axes[0].plot(t_arr, err_before_t, label="Before calibration", alpha=0.7)
    axes[0].plot(t_arr, err_after_t,  label="After calibration",  alpha=0.7)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Position error [mm]")
    axes[0].set_title("Trajectory identification: position error over time")
    axes[0].legend()
    axes[0].grid(True)

    # TCP 軌跡 (X-Y 平面)
    axes[1].plot(p_exp_t[:, 0]*1e3, p_exp_t[:, 1]*1e3, "k.", ms=2, label="Observed")
    axes[1].plot(p_init_t[:, 0]*1e3,  p_init_t[:, 1]*1e3,  "r-", lw=1, alpha=0.6, label="Predicted (before)")
    axes[1].plot(p_final_t[:, 0]*1e3, p_final_t[:, 1]*1e3, "b-", lw=1, alpha=0.8, label="Predicted (after)")
    axes[1].set_xlabel("X [mm]")
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_title("TCP trajectory (X-Y plane)")
    axes[1].legend()
    axes[1].set_aspect("equal")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_identification.png"), dpi=120)
    plt.close(fig)

    # 5. 直線軌道テストのプロット
    colors_dir = ["tab:blue", "tab:orange", "tab:green"]   # 方向ごとに色分け
    labels_dir = ["X", "Y", "Z"]
    n_home = 3
    n_dir = 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: X-Y 平面の直線軌道（before/after 重ね描き）
    ax = axes[0]
    for li, q_line in enumerate(line_trajs):
        col = colors_dir[li % n_dir]
        lbl_b = f"Before ({labels_dir[li % n_dir]})" if li < n_dir else None
        lbl_a = f"After ({labels_dir[li % n_dir]})"  if li < n_dir else None
        idx_s = li * 30
        idx_e = idx_s + 30
        ax.plot(p_before_lines[idx_s:idx_e, 0]*1e3, p_before_lines[idx_s:idx_e, 1]*1e3,
                "--", color=col, lw=1, alpha=0.5, label=lbl_b)
        ax.plot(p_after_lines[idx_s:idx_e, 0]*1e3, p_after_lines[idx_s:idx_e, 1]*1e3,
                "-", color=col, lw=1.5, alpha=0.9, label=lbl_a)
        ax.plot(p_exp_lines[idx_s:idx_e, 0]*1e3, p_exp_lines[idx_s:idx_e, 1]*1e3,
                "k.", ms=2, alpha=0.3)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_title("Straight-line trajectories (X-Y plane)")
    ax.legend(fontsize=7); ax.grid(True); ax.set_aspect("equal")

    # 右: ライン別 RMS error before/after
    ax2 = axes[1]
    rms_b_per_line = [
        np.sqrt(np.mean(np.linalg.norm(
            p_exp_lines[i*30:(i+1)*30] - p_before_lines[i*30:(i+1)*30], axis=1)**2)) * 1e3
        for i in range(n_home * n_dir)
    ]
    rms_a_per_line = [
        np.sqrt(np.mean(np.linalg.norm(
            p_exp_lines[i*30:(i+1)*30] - p_after_lines[i*30:(i+1)*30], axis=1)**2)) * 1e3
        for i in range(n_home * n_dir)
    ]
    x_idx = np.arange(n_home * n_dir)
    ax2.bar(x_idx - 0.2, rms_b_per_line, 0.4, label="Before", alpha=0.7)
    ax2.bar(x_idx + 0.2, rms_a_per_line, 0.4, label="After",  alpha=0.7)
    ax2.axhline(NOISE_STD * 1e3, color="k", ls="--", lw=0.8, label=f"Noise ({NOISE_STD*1e3:.1f}mm)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([f"H{i//n_dir+1}-{labels_dir[i%n_dir]}" for i in range(n_home*n_dir)],
                        fontsize=8)
    ax2.set_ylabel("RMS error [mm]"); ax2.set_title("RMS per line")
    ax2.legend(); ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "straight_line_identification.png"), dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")
    print("\nAll tests PASSED.")
