# Robot Calibration Library — 詳細設計書

## 目次

1. [設計思想](#1-設計思想)
2. [全体アーキテクチャ](#2-全体アーキテクチャ)
3. [データフロー](#3-データフロー)
4. [数学的基礎](#4-数学的基礎)
5. [モジュール詳細仕様](#5-モジュール詳細仕様)
6. [識別可能性と縮退](#6-識別可能性と縮退)
7. [設計判断の根拠](#7-設計判断の根拠)
8. [既知の制約と留意事項](#8-既知の制約と留意事項)

---

## 1. 設計思想

### 1.1 レシピ / 骨格の分離

本ライブラリの根本原則は「**実験ロジックをレシピファイルに閉じ込め、骨格コードは一切触らない**」ことです。

```
┌──────────────────────────────────────┐
│  recipe_*.py （実験者が書く）         │
│  ・DH 名目値                          │
│  ・パラメータ定義（group, prior_std） │
│  ・Stage リスト                        │
│  ・カスタムモデル（必要な場合のみ）   │
└──────────────┬───────────────────────┘
               │ run_calibration() を呼ぶ
┌──────────────▼───────────────────────┐
│  pipeline.py / estimation/ （骨格）  │
│  ・最適化ループ（vectorized FK）       │
│  ・残差計算                            │
│  ・ラプラス近似（σ_noise² 補正付き）   │
│  ・逐次推定・位置不確かさ伝播          │
└──────────────────────────────────────┘
```

骨格コードはモデルの**インタフェース**（`KinematicModel.forward()` / `forward_batch()`、`ObservationModel.predict()` / `predict_batch()`）のみに依存し、具体的なクラス名や内部構造を参照しません。

### 1.2 パラメータの設計思想

全パラメータは「真値からの**差分**」として定義されます。

```python
d_alpha_i, d_a_i, d_d_i, d_theta_offset_i   # DH 誤差: 名目値に対するずれ量
```

初期値はすべて `0.0`（名目ロボット = 誤差なし）。最適化で推定された値が誤差量に対応します。

---

## 2. 全体アーキテクチャ

### 2.1 モジュール構成

```
robot_calibration/
├── __init__.py           公開 API のエクスポート
├── pipeline.py           ◀ 統合パイプライン（中心的なファイル）
│
├── models/
│   ├── base.py           KinematicModel / ObservationModel / ObservationTransform（ABC）
│   ├── kinematics.py     DHKinematics / RobotKinematics / DHParams / 解析的ヤコビアン
│   ├── observation.py    PoseObservation / DistanceObservation / vec6_to_se3
│   ├── matrix.py         IdentityTransform / VelocityNormTransform / FFTAmplitudeTransform
│   └── parameters.py     Parameter / ParameterSet
│
├── estimation/
│   ├── optimizer.py      Stage / StageResult
│   └── uncertainty.py    laplace_uncertainty / UncertaintyResult / propagate_to_position
│
├── visualization/
│   └── plotter.py        plot_calibration_summary / plot_sequential_convergence など
│
├── io/
│   └── loader.py         CSV 読み込み / 保存
│
└── tests/
    └── simulation_test.py シミュレーションテスト
```

### 2.2 依存関係グラフ

```
recipe_*.py
    │
    ├─▶ pipeline.py
    │       ├─▶ models/base.py        （KinematicModel, ObservationModel, ObservationTransform）
    │       ├─▶ models/parameters.py  （ParameterSet）
    │       ├─▶ models/matrix.py      （IdentityTransform）
    │       ├─▶ estimation/optimizer.py（Stage, StageResult）
    │       └─▶ estimation/uncertainty.py（laplace_uncertainty, propagate_to_position）
    │
    ├─▶ models/kinematics.py          （DHKinematics, RobotKinematics, DHParams）
    │       └─▶ models/observation.py  （vec6_to_se3）
    │
    ├─▶ models/observation.py          （PoseObservation, DistanceObservation）
    ├─▶ models/matrix.py               （IdentityTransform など）
    │
    └─▶ visualization/plotter.py
            └─▶ estimation/uncertainty.py（UncertaintyResult）
```

---

## 3. データフロー

### 3.1 推定フロー全体図

```
入力: q_traj (N, n_joints), y_exp (N, obs_dim), PARAMETERS, STAGES
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ run_calibration()                                   │
│                                                     │
│  ParameterSet を構築                                │
│  interps = _build_interps()  ← スプライン係数を事前計算│
│         │                                           │
│  for stage in STAGES:                              │
│    1. free_idx = params.free_indices(stage.groups)  │
│    2. x0 = params.get_vector(free_idx)              │
│    3. least_squares(                                │
│         fun = _compute_residuals(x, ...),           │
│         x0  = x0,  method="lm",                   │
│       )                                             │
│    4. params.set_vector(free_idx, res.x)            │
│                                                     │
│  (final_full_tune=True なら全 free 一括で再実行)   │
└──────────────────────┬──────────────────────────────┘
                       │ params_result, stage_results
                       ▼
┌─────────────────────────────────────────────────────┐
│ compute_uncertainty()                               │
│                                                     │
│  最適解で least_squares を 1 ステップ実行して J 取得│
│  σ² = ‖r‖² / (n_obs - n_params)                   │
│  Cov = σ² (J^T J)^{-1}                            │
└──────────────────────┬──────────────────────────────┘
                       │ UncertaintyResult
                       ▼
            plot_calibration_summary()  → PNG
```

### 3.2 残差計算フロー（`_compute_residuals`）

```
x（最適化変数）
    │
    ├─ params.set_vector(free_idx, x)  ← params を x で更新
    │
    ├─ [time_offset 補間]（q_timestamps が与えられた場合）
    │   t_eff = q_timestamps + params["time_offset"]
    │   q_eff = cubic_interp(q_traj, q_timestamps)(t_eff)  ← interps を再利用
    │
    ├─ poses = kin_model.forward_batch(q_eff, pdict)   → (N, 4, 4)  ← 一括計算
    │  y_pred = obs_model.predict_batch(poses, pdict)  → (N*obs_dim,)
    │
    ├─ [transform_mode == "split"]
    │   r_obs = transform.apply(y_exp) - transform.apply(y_pred)
    │  [transform_mode == "residual"]
    │   r_raw = (y_exp - y_pred).reshape(N, obs_dim)
    │   r_obs = transform.apply(‖r_raw‖ per sample)
    │
    ├─ r_prior = (params[i].value - prior_mean) / prior_std
    │
    └─ return concat([r_obs, r_prior])   → (N*obs_dim + n_free,)
```

### 3.3 逐次推定フロー（`run_sequential_calibration`）

```
全データ (N 点) を n_groups に等分割
    edges = [0, N/G, 2N/G, ..., N]

for g in range(n_groups):
    q_sub = q_traj[:edges[g+1]]
    y_sub = y_exp[:edges[g+1]*obs_dim]

    params_g, _ = run_calibration(q_sub, y_sub, ...)   ← 累積データで最適化
    unc = compute_uncertainty(params_g, ...)            ← Laplace 近似

    J_param = _compute_param_jacobian(kin, obs, q_eval, params_g, free_idx)
                                        ↑ 数値微分 ∂y(q)/∂θ  (M*obs_dim, n_free)
    sigma_pos, sigma_xyz = propagate_to_position(unc.cov, J_param, obs_dim)
                                        ↑ Cov_pos(q) = J @ Cov_θ @ J^T

    if update_prior:
        各 params の prior_mean/std を今ステップの推定値/標準偏差で更新  ← 逐次ベイズ

    record SequentialStep(n_data, param_values, param_stds, pos_unc_mean, pos_unc_xyz, rms)
```

---

## 4. 数学的基礎

### 4.1 Modified DH 変換行列

Craig 規約（Modified DH）に従います。各リンクの変換行列は：

```
T_i = Rot_x(α_{i-1}) · Trans_x(a_{i-1}) · Rot_z(θ_i) · Trans_z(d_i)
```

全体変換:
```
T_total = T_local · T_0 · T_1 · ... · T_{n-1} · T_tool
```

- `T_local`: ベース座標系誤差（6DoF SE(3)）
- `T_tool`: エンドエフェクタ変換誤差（6DoF SE(3)）

### 4.2 一括順運動学（`forward_batch`）

N 点一括計算は NumPy の broadcast matmul を使ったジョイントループで実現されます。

```python
# _dh_transform_batch: (N,4,4) を一括生成
T = np.eye(4)[None].repeat(N, axis=0)           # (N, 4, 4)
for joint_i:
    Ti_batch = _dh_transform_batch(alpha, a, d, q_batch[:, i])   # (N,4,4)
    T = T @ Ti_batch                             # (N,4,4) @ (N,4,4) → NumPy broadcast
```

Python ループ逐次実行（O(N × n_joints) の行列積）に対して **約 30 倍** の高速化を達成しています。

### 4.3 解析的位置ヤコビアン（∂p/∂q）

連鎖律を用いて：

```
∂T_total/∂θ_i = T_local · T_{0..i-1} · (∂T_i/∂θ_i) · T_{i..n-1} · T_tool
```

前向き伝播（prefix）と後ろ向き伝播（suffix）を分離して O(n) で計算します：

```python
prefix = T_local @ Ts[i]   # 0 から i-1 までの累積変換
suffix = suffixes[i+1]     # i+1 から n までの累積変換
J[:, i] = (prefix @ dTi_dtheta @ suffix)[:3, 3]
```

### 4.4 回転の表現（Rodrigues 式）

ツール・ローカル変換の回転成分は軸角度 `[rx, ry, rz]` から SE(3) へ `vec6_to_se3()` で変換されます（小角度近似なし、任意の回転角に対して正確）。

### 4.5 非線形最小二乗法

最小化問題：

```
min_θ  ½ ‖r(θ)‖²

r(θ) = [r_obs(θ);  r_prior(θ)]

r_obs(θ)_t   = T(y_exp_t) - T(y_pred_t(θ))      (観測残差)
r_prior(θ)_i = (θ_i - μ_i) / σ_i                 (事前分布残差)
```

使用ソルバ: `scipy.optimize.least_squares` (method="lm"、Levenberg-Marquardt)

### 4.6 事前分布による正則化

事前分布残差は L2 正則化（Tikhonov 正則化）に相当し、ベイズ推定の MAP 推定値に対応します：

```
½ ‖r_obs‖² + ½ Σ_i ((θ_i - μ_i) / σ_i)²
```

`prior_std = np.pi` や `prior_std = 1.0` という設定は「事実上の非情報的事前分布」であり、データが事前分布をはるかに上回ります。

### 4.7 ラプラス近似による不確かさ推定

最適解 `θ*` 周辺で事後分布をガウスで近似します：

```
事後分布 ≈ N(θ*,  Cov_θ)
Cov_θ ≈ σ_noise² · (J^T J)^{-1}
σ_noise² = ‖r_obs‖² / (n_obs - n_params)    ← 自由度補正付き分散推定
```

`σ_noise²` を掛けることで、観測ノイズの大きさが不確かさに正しく反映されます。実装では `max_nfev=1` で `least_squares` を呼ぶことで **1 ステップ実行後のヤコビアンだけ**を取得しています（最適解を動かさないための最小評価）。

`σ_i` の意味：

| `σ_i` の値 | 意味 |
|-----------|------|
| `σ_i ≪ prior_std` | データがパラメータを強く制約している |
| `σ_i ≈ prior_std` | データがほとんど情報を与えていない（識別困難） |
| `σ_i ≫ prior_std` | 数値的縮退（ランク落ち状態） |

### 4.8 手先位置不確かさへの伝播

パラメータ共分散 `Cov_θ` から観測空間の位置不確かさを伝播します：

```
Cov_pos(q) = J_param(q) @ Cov_θ @ J_param(q)^T
σ_pos(q)   = √tr(Cov_pos(q))
```

ここで `J_param(q) = ∂y(q)/∂θ ∈ R^{obs_dim × n_params}` はパラメータに対する数値ヤコビアン（前向き差分）。  
`run_sequential_calibration` では評価点 `q_eval` での `σ_pos` の平均値と軸ごとの標準偏差を `SequentialStep` に記録し、データ数に対する収束曲線として可視化できます。

### 4.9 時刻ずれ推定（`time_offset`）

観測モデル: `y_obs(t) = FK(q(t + Δt)) + ε`

パイプライン内での実装:

```python
t_eff = q_timestamps + dt_offset          # シフトした評価時刻
q_eff = cubic_interp(q_timestamps, q_traj)(t_eff)  # 各軸独立に 3 次補間
```

スプライン係数は最適化ループ前に一度だけ `_build_interps()` で生成し、各評価で再利用します。

---

## 5. モジュール詳細仕様

### 5.1 `models/base.py` — 抽象インタフェース

#### `KinematicModel` (ABC)

```python
def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
    """関節角度 q → SE(3) 行列 (4, 4)"""

def forward_batch(self, q_batch: np.ndarray, params: dict) -> np.ndarray:
    """N 点一括順運動学 (N, n_joints) → (N, 4, 4)
    デフォルト実装はループ。オーバーライドで高速化可能。"""
```

#### `ObservationModel` (ABC)

```python
def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
    """SE(3) 行列 (4, 4) → 観測予測値 (obs_dim,)"""

def predict_batch(self, poses: np.ndarray, params: dict) -> np.ndarray:
    """(N, 4, 4) → (N*obs_dim,)
    デフォルト実装はループ。オーバーライドで高速化可能。"""
```

#### `ObservationTransform` (ABC)

```python
transform_mode: str   # "split" または "residual"

def apply(self, y: np.ndarray) -> np.ndarray:
    """観測値を変換する"""

def jacobian(self, y: np.ndarray) -> np.ndarray:
    """apply の解析的ヤコビアン（オプション）"""
```

### 5.2 `models/kinematics.py` — DH 順運動学

#### `DHParams` (dataclass)

```python
@dataclass
class DHParams:
    alpha: float        # z_{i-1} 軸に対する x_i 軸の傾き [rad]
    a: float            # z_{i-1} から z_i への x_i 方向距離 [m]
    d: float            # x_{i-1} から x_i への z_i 方向距離 [m]
    theta_offset: float # 関節角度のゼロ点オフセット [rad]
```

#### `RobotKinematics`

| メソッド | 役割 |
|----------|------|
| `forward(q, T_tool, T_local)` | 全体変換 `T_total = T_local · T_01 · T_tool` |
| `forward_batch(q_batch, T_tool, T_local)` | N 点一括 FK（NumPy broadcast matmul） |
| `link_transforms(q)` | 中間変換リスト — Jacobian 計算で再利用 |
| `jacobian_position(q, ...)` | 関節角度 Jacobian `∂p/∂q ∈ R^{3×n}` |
| `jacobian_dh_params(q, ...)` | DH パラメータ Jacobian（検証用） |

#### `DHKinematics(KinematicModel)`

`RobotKinematics` のラッパー。`params` 辞書から DH 誤差・ツール・ローカルパラメータを読み取り `RobotKinematics` を構築します。  
`_build_kin()` は最適化の各評価で毎回呼ばれますが、オーバーヘッドは小さいです。

```python
def _build_kin(self, params: dict):
    dh_list = [
        DHParams(
            alpha        = dh["alpha"] + params.get(f"d_alpha_{i}", 0.0),
            a            = dh["a"]     + params.get(f"d_a_{i}",     0.0),
            d            = dh["d"]     + params.get(f"d_d_{i}",     0.0),
            theta_offset = dh["theta_offset"] + params.get(f"d_theta_offset_{i}", 0.0),
        )
        for i, dh in enumerate(self.dh_nominal)
    ]
    T_tool  = vec6_to_se3([params.get(k, 0.0) for k in ["tool_tx", ...]])
    T_local = vec6_to_se3([params.get(k, 0.0) for k in ["local_tx", ...]])
    return RobotKinematics(dh_list), T_tool, T_local
```

### 5.3 `models/observation.py` — 観測モデル

| 要素 | 説明 |
|------|------|
| `PoseObservation` | `predict`: `x[:3, 3]` → (3,)。`predict_batch`: `poses[:,:3,3].flatten()` → (N*3,) |
| `DistanceObservation(origin)` | `predict`: `‖TCP - origin‖` → (1,)。`predict_batch`: 一括 L2 ノルム |
| `vec6_to_se3(v)` | `[tx,ty,tz,rx,ry,rz]` → SE(3)（Rodrigues 式、小角度近似なし） |
| `pose_to_position(T)` | `T[:3, 3]` |
| `pose_to_axis_angle(T)` | 位置 + 軸角度 `(6,)` |

### 5.4 `models/matrix.py` — 観測変換

残差計算には 2 つのモードがあり、`transform_mode` 属性で切り替えます：

```
"split"   : r = T(y_exp) - T(y_pred)         ← Identity, VelocityNorm
"residual": r = T(‖y_exp - y_pred‖_per_pt)   ← FFTAmplitude
```

#### `IdentityTransform`

`apply(y) = y`。標準的な使用法。

#### `VelocityNormTransform(dt)`

```
s[t] = ‖y[t+1] - y[t]‖ / dt    (N-1 次元)
```

座標系の offset（定数）成分を消去し、時刻ずれ `Δt` に対する感度を高めます。

#### `FFTAmplitudeTransform`

`transform_mode = "residual"` のため特殊な処理パスを通ります：

```python
r_raw  = p_exp - p_pred          # (N, obs_dim)
r_norm = ‖r_raw‖ per sample      # (N,)
r      = |FFT(r_norm)|            # (N//2+1,)
```

位置誤差ノルムの周波数スペクトルをゼロに近づけることで、周期的伝達誤差を同定します。  
ただし `IdentityTransform` + フーリエ係数パラメータ化（`a·cos + b·sin`）の方が収束が安定しているため、現在のレシピでは `IdentityTransform` を採用しています。

### 5.5 `models/parameters.py` — パラメータ管理

#### `Parameter` (dataclass)

```python
@dataclass
class Parameter:
    name      : str
    value     : float       # 現在の推定値（最適化で更新される）
    fixed     : bool = False
    prior_mean: float = 0.0
    prior_std : float = 1e3  # デフォルトは非情報的
    group     : str = "default"
```

#### `ParameterSet`

最適化変数ベクトル `x ∈ R^{n_free}` と `params` リストの間のマッピング層。  
固定パラメータは `free_indices()` で除外されるため最適化変数に含まれません。

```python
def free_indices(self, groups=None) -> list[int]
def get_vector(self, indices) -> np.ndarray
def set_vector(self, indices, values)
def get_prior_residuals(self, indices) -> np.ndarray   # (θ_i - μ_i) / σ_i
def summary(self) -> str
```

### 5.6 `pipeline.py` — 統合パイプライン

#### `_build_interps(q_timestamps, q_traj) -> list`

cubic spline 補間器を最適化ループ前に一度だけ生成します。スプライン係数はデータが変わらない限り変わらないため、毎評価での再計算は不要です。

#### `_compute_residuals(...)`

最適化の目的関数。`forward_batch` / `predict_batch` を使った一括 FK 計算で高速化されています。

#### `_compute_param_jacobian(kin_model, obs_model, q_eval, params, free_indices, eps=1e-6)`

パラメータに対する観測値の数値ヤコビアン `∂y(q)/∂θ` を前向き差分で計算します。  
`forward_batch` を使うため、M 評価点に対してパラメータ 1 個あたり 1 回の一括 FK 呼び出しで済みます。

```python
for j, idx in enumerate(free_indices):
    params.params[idx].value += eps
    y_p = obs_model.predict_batch(kin_model.forward_batch(q_eval, pdict_p), pdict_p)
    J[:, j] = (y_p - y0) / eps
    params.params[idx].value -= eps   # 復元
```

返値: `J ∈ R^{M*obs_dim × n_free}`

#### `run_calibration(...)`

段階的キャリブレーション。各ステージごとに `least_squares` を実行し、`params` を更新します。

#### `compute_uncertainty(...)`

`include_prior=False` で残差ヤコビアンを取得し、`laplace_uncertainty` に `residuals=res.fun` を渡します。

```python
return laplace_uncertainty(res.jac, free_names, x0, residuals=res.fun)
```

#### `run_sequential_calibration(...)`

データを `n_groups` グループに等分割し、累積データ量を増やしながら逐次推定を行います。  
各ステップで `SequentialStep`（推定値・不確かさ・手先位置不確かさ・RMS）を記録します。

### 5.7 `estimation/optimizer.py` — ステージ定義

#### `Stage` (dataclass)

```python
@dataclass
class Stage:
    name         : str
    param_groups : list[str] | None   # None = 全 free パラメータ
    transform    : ObservationTransform
```

#### `StageResult` (dataclass)

```python
@dataclass
class StageResult:
    stage_name : str
    x_opt      : np.ndarray       # 最適化された free パラメータベクトル
    cost       : float            # 最終コスト ½‖r‖²
    success    : bool
    message    : str
    jacobian   : np.ndarray | None
```

### 5.8 `estimation/uncertainty.py` — ラプラス近似

```python
def laplace_uncertainty(
    jac         : np.ndarray,           # (n_obs, n_params)
    param_names : list[str],
    param_values: np.ndarray,
    residuals   : np.ndarray | None = None,  # 最適解での残差
) -> UncertaintyResult:
    H    = jac.T @ jac
    cov  = inv(H)  # ランク落ち時は pinv
    if residuals is not None:
        sigma2 = dot(residuals, residuals) / max(n_obs - n_params, 1)
        cov   *= sigma2          # Cov_θ = σ_noise² (J^T J)^{-1}
    stds = sqrt(maximum(diag(cov), 0))
```

```python
def propagate_to_position(
    cov_theta : np.ndarray,   # (n_params, n_params)
    J_param   : np.ndarray,   # (M*obs_dim, n_params)
    obs_dim   : int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    # sigma_pos (M,), sigma_xyz (M, obs_dim)
    for t in range(M):
        Jt = J_param[t*obs_dim:(t+1)*obs_dim, :]  # (obs_dim, n_params)
        Cov_pos = Jt @ cov_theta @ Jt.T
        sigma_pos[t] = sqrt(trace(Cov_pos))
        sigma_xyz[t] = sqrt(diag(Cov_pos))
```

### 5.9 `visualization/plotter.py` — グラフ出力

全プロット関数に `save_path=None` パラメータがあり、指定時は `dpi=150` で保存して figure を閉じます。

#### `plot_calibration_summary`

`matplotlib.gridspec.GridSpec` による 3 段構成（XYZ 残差 / 誤差ノルム / パラメータ ±1σ）。

#### `plot_sequential_convergence(steps, ...)`

逐次推定の収束プロット。

```
Row 0: 手先位置不確かさ σ_pos の平均値 vs データ数  ← メイン指標
Row 1左: 軸ごとの σ_X / σ_Y / σ_Z vs データ数
Row 1右: 観測残差 RMS vs データ数（参考）
Row 2+:  パラメータ推定値 ± 1σ（param_filter 指定時）
```

---

## 6. 識別可能性と縮退

### 6.1 完全縮退パラメータ（位置のみ観測）

`tool_rz` と `d_theta_offset_5` は常に完全縮退します：

```
FK(q, tool_rz=δ) ≡ FK(q, d_theta_offset_5=-δ)  （位置成分のみ）
```

**解決策**: `tool_rz` を `fixed=True` に設定し、すべての最終軸回転誤差を `d_theta_offset_5` に吸収させます。

### 6.2 ボールバー（距離スカラー観測）の識別制限

単一固定点からの距離は 1 スカラーのため、3D 位置観測より情報量が少ないです。  
Laplace 近似の σ が `prior_std` に近いパラメータが識別困難です。複数の固定原点を使うと識別性が向上します。

### 6.3 関節伝達誤差の識別上の注意

`amp * sin(q + phase)` パラメータ化では `amp = 0` 付近で `phase` の勾配がゼロになり、LM が停留します。  
**解決策**: フーリエ係数形式 `a·cos(q) + b·sin(q)` に変換します（勾配が `amp=0` でも非ゼロ）。

### 6.4 DH 誤差と伝達誤差の相関

`d_theta_offset_i` は限られた関節可動範囲では伝達誤差 `a·cos(q) + b·sin(q)` と部分的に相関します。  
**対策**: 関節掃引を先行させてから DH を推定し、`final_full_tune=True` で最終収束させます。

---

## 7. 設計判断の根拠

### 7.1 `params: dict` vs. `ParameterSet` 直接渡し

`KinematicModel.forward(q, params: dict)` では Python 辞書を受け取ります。  
サブクラスで追加パラメータを `params.get(key, 0.0)` で簡単に参照できるため、`ParameterSet` への依存をモデル層から排除しています。

### 7.2 `method="lm"` の選択

ロボットキャリブレーション問題は密なヤコビアン（全パラメータが全観測に影響）のため、疎行列を仮定する `trf` / `dogbox` より LM が適しています。

### 7.3 `include_prior=False` での `compute_uncertainty`

```
include_prior=True: H = J_obs^T J_obs + Λ   → 事前分布が強いと σ_i が人工的に小さくなる
include_prior=False: H = J_obs^T J_obs        → データのみの不確かさを正しく反映
```

「データがどれだけ情報を与えているか」を評価するには事前情報を含めないのが正しいです。

### 7.4 `σ_noise²` スケーリングの適用

従来は `Cov = (J^T J)^{-1}` としていたため、ノイズスケールに依存しない不適切な共分散でした。  
`σ² = ‖r‖² / (n_obs - n_params)` で推定したノイズ分散を掛けることで、観測ノイズの大きさが正しく伝播します。  
N=300、ノイズ 0.3 mm の条件での検証で `σ_pos ≈ 0.07 mm`（修正前 ~191 mm）という物理的に妥当な結果を確認しています。

### 7.5 `forward_batch` をデフォルト実装込みで ABC に追加した理由

カスタム FK モデルへの追加コストをゼロにするためです。デフォルト実装（ループ）を `base.py` に持たせることで、サブクラスはオーバーライドなしでも動作し、性能が必要なときだけ実装を追加できます。

### 7.6 `final_full_tune` の位置づけ

段階的推定は先行ステージが誤りを引き込むと後続ステージが補正しきれない場合があります。`final_full_tune=True` は全パラメータを同時に解放して「最後の一押し」をします。ただし局所最適を保証するものではありません。

---

## 8. 既知の制約と留意事項

### 8.1 cubic 補間とタイムスタンプの連続性

`interp1d(kind="cubic", bounds_error=False, fill_value=(端点値, 端点値))` を使用します。

- **境界外**: 端点値で打ち切ります。評価時刻が `q_timestamps` の範囲外になる大きな `time_offset` では誤差が出ます
- **不連続点**: 関節角度に不連続がある場合（停止セグメントの終端など）は 3 次スプラインが大きな overshoot を生みます

### 8.2 ランク落ちと疑似逆行列

`laplace_uncertainty` は `LinAlgError` を捕捉して `pinv` で対処しますが、縮退している場合は特定の方向の `σ` が非常に大きい値になります。この場合は `fixed=True` で縮退パラメータを固定してから再評価してください。

### 8.3 数値精度とスケール

LM アルゴリズムはパラメータスケールの差が大きい場合（例: `d_alpha_i` の典型値 `~1e-3 rad` vs. `d_d_i` の典型値 `~1e-3 m`）に収束が遅くなる場合があります。`prior_std` を実際の誤差オーダーに合わせることで正則化がスケール補正の役割を持ちます。

### 8.4 `_compute_param_jacobian` のコスト

数値ヤコビアン `∂y/∂θ` の計算は `n_free` 回の `forward_batch` を要します。`run_sequential_calibration` では最終的な不確かさ評価の後に一度だけ呼ばれるため問題ありませんが、最適化ループ内に組み込むと計算コストが n_free 倍になります。

---

*本設計書は `robot_calibration/` の実装コードを正とします。コードと設計書に齟齬がある場合はコードが優先されます。*
