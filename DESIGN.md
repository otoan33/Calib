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
│  ・最適化ループ                        │
│  ・残差計算                            │
│  ・ラプラス近似                        │
└──────────────────────────────────────┘
```

骨格コードはモデルの**インタフェース**（`KinematicModel.forward()`, `ObservationModel.predict()`）のみに依存し、具体的なクラス名や内部構造を参照しません。これにより：

- 新しい観測系（距離計・IMU 等）の追加がレシピへのサブクラス追加だけで完結する
- 骨格コードの回帰テストが容易

### 1.2 パラメータの設計思想

全パラメータは「真値からの**差分**」として定義されます。

```python
# DH 誤差: 名目値に対するずれ量
d_alpha_i, d_a_i, d_d_i, d_theta_offset_i
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
│   ├── base.py           KinematicModel / ObservationModel（抽象基底クラス）
│   ├── kinematics.py     RobotKinematics（DH 順運動学・解析的ヤコビアン）
│   ├── observation.py    vec6_to_se3, PositionObservationModel（内部実装）
│   ├── defaults.py       DHKinematics / PoseObservation / DistanceObservation（公開 API）
│   ├── parameters.py     Parameter / ParameterSet
│   ├── transforms.py     ObservationTransform 各種
│   └── residuals.py      compute_residuals / apply_transmission_error（旧 API、残留）
│
├── estimation/
│   ├── optimizer.py      Stage / StageResult / run_staged_optimization（旧 API）
│   └── uncertainty.py    laplace_uncertainty / UncertaintyResult
│
├── visualization/
│   └── plotter.py        plot_calibration_summary など
│
├── io/
│   └── loader.py         CSV 読み込み / 保存
│
└── tests/
    └── simulation_test.py シミュレーションテスト
```

> **注**: `estimation/optimizer.py` と `models/residuals.py` は旧来の API（`RobotKinematics` 直接参照）で残存しています。現在のパイプライン（`pipeline.py`）は `models/base.py` のインタフェースを経由した新 API を使用しています。

### 2.2 依存関係グラフ

```
recipe_*.py
    │
    ├─▶ pipeline.py
    │       ├─▶ models/base.py        （KinematicModel, ObservationModel）
    │       ├─▶ models/parameters.py  （ParameterSet）
    │       ├─▶ models/transforms.py  （ObservationTransform）
    │       ├─▶ estimation/optimizer.py（Stage, StageResult）
    │       └─▶ estimation/uncertainty.py（laplace_uncertainty）
    │
    ├─▶ models/defaults.py
    │       ├─▶ models/kinematics.py  （RobotKinematics, DHParams）
    │       └─▶ models/observation.py  （vec6_to_se3）
    │
    └─▶ visualization/plotter.py
            └─▶ estimation/uncertainty.py（UncertaintyResult）
```

---

## 3. データフロー

### 3.1 推定フロー全体図

```
入力: q_traj (N, 6), y_exp (N, obs_dim), PARAMETERS, STAGES
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ run_calibration()                                   │
│                                                     │
│  ParameterSet を構築                                │
│         │                                           │
│  for stage in STAGES:                              │
│    1. free_idx = params.free_indices(stage.groups)  │
│    2. x0 = params.get_vector(free_idx)              │
│    3. least_squares(                                │
│         fun = _compute_residuals(x, ...),           │
│         x0  = x0,                                  │
│         method="lm",                               │
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
│  Cov = (J^T J)^{-1}  → stds = sqrt(diag(Cov))     │
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
    │   q_eff = cubic_interp(q_traj, q_timestamps)(t_eff)
    │
    └─ for t in range(N):
           pose = kin_model.forward(q_eff[t], params_dict)  → (4,4) SE(3)
           pred = obs_model.predict(pose, params_dict)       → (obs_dim,)
       y_pred = concat(preds)                               → (N*obs_dim,)
    
    r_obs = transform.apply(y_exp_flat) - transform.apply(y_pred)
    r_prior = (params[i].value - prior_mean) / prior_std  ← L2 正則化
    
    return concat([r_obs, r_prior])                        → (N*obs_dim + n_free,)
```

---

## 4. 数学的基礎

### 4.1 Modified DH 変換行列

Craig 規約（Modified DH）に従います。各リンクの変換行列は：

```
T_i = Rot_x(α_{i-1}) · Trans_x(a_{i-1}) · Rot_z(θ_i) · Trans_z(d_i)
```

行列形式では：

```
     | cos θ       -sin θ        0        a        |
T =  | sin θ cos α  cos θ cos α  -sin α   -d sin α  |
     | sin θ sin α  cos θ sin α   cos α    d cos α  |
     | 0            0             0        1        |
```

ここで `θ_i = q[i] + theta_offset_i`（関節角度 + 名目オフセット + 誤差）。

全体変換:
```
T_total = T_local · T_0 · T_1 · ... · T_{n-1} · T_tool
```

- `T_local`: ベース座標系誤差（6DoF SE(3)）
- `T_tool`: エンドエフェクタ変換誤差（6DoF SE(3)）

### 4.2 解析的位置ヤコビアン（∂p/∂q）

連鎖律を用いて：

```
∂T_total/∂θ_i = T_local · T_{0..i-1} · (∂T_i/∂θ_i) · T_{i..n-1} · T_tool
```

位置成分 `J[:, i] = (∂T_total/∂θ_i)[:3, 3]`

各パラメータの偏微分行列（`models/kinematics.py`に実装）:

| 偏微分 | 行列 |
|--------|------|
| `∂T/∂θ` | `(-sin θ, -cos θ, 0, 0; cos θ cos α, -sin θ cos α, 0, 0; ...)` |
| `∂T/∂α` | `(0,0,0,0; -sin θ sin α, -cos θ sin α, -cos α, -d cos α; ...)` |
| `∂T/∂a` | `(0,0,0,1; 0,0,0,0; 0,0,0,0; 0,0,0,0)` |
| `∂T/∂d` | `(0,0,0,0; 0,0,0,-sin α; 0,0,0,cos α; 0,0,0,0)` |

`theta_offset_i` は `θ_i` と同じ偏微分（`∂T/∂theta_offset = ∂T/∂θ`）。

### 4.3 DH パラメータ誤差に対するヤコビアン

`jacobian_dh_params()` は各 DH 誤差パラメータに対する位置感度を返します：

```python
∂p/∂d_alpha_i = (T_local · T_{0..i-1} · (∂T_i/∂α_i) · T_{i..n} · T_tool)[:3, 3]
```

これは最適化の Jacobian とは異なります（後者は `scipy` が数値微分で計算）。

### 4.4 回転の表現（Rodrigues 式）

ツール・ローカル変換の回転成分は軸角度 `[rx, ry, rz]` から SE(3) へ `vec6_to_se3()` で変換されます：

```
θ = ‖[rx, ry, rz]‖
k = [rx, ry, rz] / θ
R = I + sin(θ) K + (1 - cos θ) K²
```

ここで `K` は `k` の歪対称行列。小角度近似なし（任意の回転角に対して正確）。

### 4.5 非線形最小二乗法

最小化問題：

```
min_θ  ½ ‖r(θ)‖²

r(θ) = [r_obs(θ);  r_prior(θ)]

r_obs(θ)_t   = T(y_exp_t) - T(y_pred_t(θ))      (観測残差)
r_prior(θ)_i = (θ_i - μ_i) / σ_i                 (事前分布残差)
```

使用ソルバ: `scipy.optimize.least_squares` (method="lm"、Levenberg-Marquardt)

```
ls_kwargs = {
    "method": "lm",
    "ftol": 1e-12,    # 関数値の相対変化閾値
    "xtol": 1e-12,    # パラメータの相対変化閾値
    "gtol": 1e-12,    # 勾配の無限大ノルム閾値
    "max_nfev": 50000,
}
```

### 4.6 事前分布による正則化

事前分布残差は実質的に L2 正則化（Tikhonov 正則化）に相当します：

```
r_prior_i = (θ_i - μ_i) / σ_i    (prior_mean = μ_i, prior_std = σ_i)
```

コスト関数は：

```
½ ‖r_obs‖² + ½ Σ_i ((θ_i - μ_i) / σ_i)²
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             正則化項（σ が大きいほど弱い制約）
```

これはベイズ推定の MAP 推定値（事後最頻値）に対応します。  
`prior_std = np.pi` や `prior_std = 1.0` という設定は「事実上の非情報的事前分布」であり、データが事前分布をはるかに上回ります。

### 4.7 ラプラス近似による不確かさ推定

最適解 `θ*` 周辺で事後分布をガウスで近似します：

```
事後分布 ≈ N(θ*,  Cov)
Cov ≈ (J^T J)^{-1}    (J: 観測残差のヤコビアン、prior 項を除く)
σ_i = sqrt(Cov_{ii})
```

実装では `max_nfev=1` で `least_squares` を呼ぶことで **1 ステップ実行後のヤコビアンだけ**を取得しています（最適解を動かさないための最小評価）。

```python
res = least_squares(fun, x0, method="lm", max_nfev=1)
H   = res.jac.T @ res.jac   # フィッシャー情報行列
Cov = inv(H)                 # ランク落ちは pinv で対処
```

`σ_i` の意味：

| `σ_i` の値 | 意味 |
|-----------|------|
| `σ_i ≪ prior_std` | データがパラメータを強く制約している |
| `σ_i ≈ prior_std` | データがほとんど情報を与えていない（識別困難） |
| `σ_i ≫ prior_std` | 数値的縮退（ランク落ち状態） |

### 4.8 時刻ずれ推定（`time_offset`）

観測モデル:

```
y_obs(t) = FK(q(t + Δt)) + ε
```

パイプライン内での実装:

```python
t_eff = q_timestamps + dt_offset          # シフトした評価時刻
q_eff = cubic_interp(q_timestamps, q_traj)(t_eff)  # 各軸独立に 3 次補間
```

`IdentityTransform` を使う場合、コスト `‖y_obs - FK(q(t + Δt))‖²` の `Δt` に対する勾配は：

```
∂cost/∂Δt = -2 Σ_t r(t)^T · J_q(t) · dq/dt(t + Δt)
```

軌道が単調に変化しているとき（速度が一定方向）、この勾配は単調で局所最適解が生じにくいです。

> **C1 連続性の要件**: `cubic_interp` は入力点が C1 連続（速度連続）でないと補間後に大きなスパイクを生じます。関節の不連続な停止（速度ゼロ → 有限速度の 1 サンプル変化）は数万 mm/s の仮想速度を生みます。

---

## 5. モジュール詳細仕様

### 5.1 `models/base.py` — 抽象インタフェース

#### `KinematicModel` (ABC)

```python
def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
    """関節角度 q → SE(3) 行列 (4, 4)"""

def jacobian(self, q: np.ndarray, params: dict) -> np.ndarray:
    """∂p/∂q ∈ R^{3 × n_joints}（パス計画・特異点チェック用）"""

def numerical_jacobian(self, q, params, eps=1e-7) -> np.ndarray:
    """中心差分による数値 Jacobian（解析値との検証用、デフォルト実装）"""
```

`params` は `{パラメータ名: float}` の辞書。骨格コードはこの辞書を組み立てて `forward()` に渡します。未定義キーは `.get(key, 0.0)` でデフォルト処理するよう実装側に求めます。

#### `ObservationModel` (ABC)

```python
def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
    """SE(3) 行列 x (4, 4) → 観測予測値ベクトル (obs_dim,)"""
```

`obs_dim` は任意: `PoseObservation` は `(3,)`、`DistanceObservation` は `(1,)`。パイプラインは各点の出力を `np.concatenate` で結合するため、`obs_dim` が異なっても同じコードで処理されます。

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
| `link_transforms(q)` | 中間変換リスト `[I, T_0, T_01, ..., T_0..n]` — Jacobian 計算で再利用 |
| `jacobian_position(q, ...)` | 関節角度 Jacobian `∂p/∂q ∈ R^{3×n}` |
| `jacobian_dh_params(q, ...)` | DH パラメータ Jacobian（シミュレーション検証用） |

`jacobian_position` の実装では、前向き伝播（`link_transforms`）と後ろ向き伝播（`suffixes`）を分離して計算します：

```python
# O(n) でヤコビアンの全列を計算
prefix = T_local @ Ts[i]         # 0 から i-1 までの累積変換
suffix = suffixes[i+1]            # i+1 から n までの累積変換
J[:, i] = (prefix @ dTi_dtheta @ suffix)[:3, 3]
```

### 5.3 `models/observation.py` — 内部ユーティリティ

| 関数 | 説明 |
|------|------|
| `vec6_to_se3(v)` | `[tx,ty,tz,rx,ry,rz]` → SE(3)（Rodrigues 式、小角度近似なし） |
| `pose_to_position(T)` | `T[:3, 3]` — 位置抽出 |
| `pose_to_axis_angle(T)` | 位置 + 軸角度 `(6,)` — 姿勢観測時に使用 |

`vec6_to_se3` はゼロ回転（`‖[rx,ry,rz]‖ < 1e-9`）を単位行列として扱い、ゼロ除算を回避します。

### 5.4 `models/defaults.py` — 公開モデル実装

#### `DHKinematics(KinematicModel)`

`RobotKinematics` のラッパー。`params` 辞書から DH 誤差・ツール・ローカルパラメータを読み取り `RobotKinematics` を構築します：

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

最適化のたびに毎回 `RobotKinematics` を再構築します（LM の各評価ごと）。

#### `PoseObservation(ObservationModel)`

```python
def predict(self, x, params): return x[:3, 3]   # → (3,)
```

#### `DistanceObservation(ObservationModel)`

```python
def predict(self, x, params):
    return np.array([np.linalg.norm(x[:3, 3] - self.origin)])   # → (1,)
```

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

```python
class ParameterSet:
    params: list[Parameter]

    def free_indices(self, groups=None) -> list[int]:
        """fixed=False かつ groups に属するインデックスを返す"""

    def get_vector(self, indices) -> np.ndarray:
        """インデックス列の値を ndarray として取得"""

    def set_vector(self, indices, values):
        """最適化結果を params に書き戻す"""

    def get_prior_residuals(self, indices) -> np.ndarray:
        """(θ_i - μ_i) / σ_i を返す（コストの正則化項）"""

    def summary(self) -> str:
        """全パラメータを表形式で文字列化"""
```

`ParameterSet` は最適化変数ベクトル `x ∈ R^{n_free}` と `params` リストの間のマッピング層です。固定パラメータは `free_indices()` で除外されるため最適化変数に含まれません。

### 5.6 `models/transforms.py` — 観測変換

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

座標系の offset（定数）成分を消去し、時刻ずれ `Δt` に対する感度を高めます。ヤコビアンの解析式が実装されています（`scipy` に渡す際に数値微分の精度改善が期待できます）。

#### `FFTAmplitudeTransform`

`transform_mode = "residual"` のため特殊な処理パスを通ります：

```python
r_raw  = p_exp - p_pred          # (N, obs_dim)
r_norm = ‖r_raw‖ per sample      # (N,)
r      = |FFT(r_norm)|            # (N//2+1,)
```

位置誤差ノルムの周波数スペクトルをゼロに近づけることで、周期的伝達誤差を同定します。ただし `IdentityTransform` + フーリエ係数パラメータ化（`a·cos + b·sin`）の方が収束が安定しているため、現在のレシピでは `IdentityTransform` を採用しています。

### 5.7 `pipeline.py` — 統合パイプライン

#### `_compute_residuals(x, free_idx, params, kin_model, obs_model, q_traj, y_exp, transform, ...)`

最適化の目的関数。毎評価で呼ばれる最内ループのため、無駄な配列確保を避けています：

```python
params.set_vector(free_idx, x)    # params を x で更新（in-place）
pdict = _params_to_dict(params)   # {name: value} の辞書作成

# time_offset 補間（q_timestamps が与えられた場合のみ）
if "time_offset" in pdict:
    t_eff = q_timestamps + pdict["time_offset"]
    q_eff = cubic_column_interp(q_traj, q_timestamps, t_eff)
else:
    q_eff = q_traj

# 観測予測
preds = [obs_model.predict(kin_model.forward(q_eff[t], pdict), pdict) for t in range(N)]
y_pred = np.concatenate(preds)    # (N * obs_dim,)

# 残差計算（transform_mode で分岐）
r_obs  = transform.apply(y_exp) - transform.apply(y_pred)
r_prior = params.get_prior_residuals(free_idx)

return np.concatenate([r_obs, r_prior])
```

#### `run_calibration(...)`

```python
def run_calibration(q_traj, y_exp, parameters, kinematic_model, observation_model,
                    stages, ls_kwargs=None, q_timestamps=None, final_full_tune=False):
    params = ParameterSet(parameters)
    y_exp_flat = np.asarray(y_exp).flatten()   # (N * obs_dim,)

    for stage in stages:
        free_idx = params.free_indices(groups=stage.param_groups)
        x0 = params.get_vector(free_idx)
        fun = lambda x: _compute_residuals(x, free_idx, params, ...)
        res = least_squares(fun, x0, **ls_kwargs)
        params.set_vector(free_idx, res.x)
        # StageResult に res.jac を保存（不確かさ評価で使えることがある）

    if final_full_tune:
        # 全 free パラメータを一括で最終調整
        final = Stage("final_full_tune", param_groups=None, transform=IdentityTransform())
        ...
```

`y_exp_flat = np.asarray(y_exp).flatten()` により、(N, obs_dim) 形式でも (N*obs_dim,) 形式でも同一処理となります。

#### `compute_uncertainty(params, kin_model, obs_model, q_traj, y_exp, ...)`

```python
fun = lambda x: _compute_residuals(..., include_prior=False)
                 # ^^ prior 項を除外して純粋な観測 Jacobian だけ使う
res = least_squares(fun, x0, method="lm", max_nfev=1)   # 1 ステップだけ
return laplace_uncertainty(res.jac, free_names, x0)
```

`include_prior=False` にする理由: 事前分布残差のヤコビアンを含めると `H = J^T J` が正則化行列になり、不確かさが過小評価されます。純粋な観測情報だけから `σ` を計算することで「データがどれだけ情報を与えているか」が正しく評価されます。

### 5.8 `estimation/optimizer.py` — ステージ定義

#### `Stage` (dataclass)

```python
@dataclass
class Stage:
    name         : str
    param_groups : list[str] | None   # None = 全 free パラメータ
    transform    : ObservationTransform
    data_subset  : str | None = None  # 現在未使用（将来の部分データ対応用）
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
    jacobian   : np.ndarray | None  # scipy の res.jac（不確かさ評価で使用可）
```

> **注**: `estimation/optimizer.py` の `run_staged_optimization()` は旧 API。現在は `pipeline.py` の `run_calibration()` が推奨。

### 5.9 `estimation/uncertainty.py` — ラプラス近似

```python
def laplace_uncertainty(jac, param_names, param_values) -> UncertaintyResult:
    H   = jac.T @ jac                           # (n_free, n_free)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)                 # ランク落ち時は疑似逆行列
    stds = np.sqrt(np.maximum(np.diag(cov), 0)) # 数値誤差による負値を除外
    return UncertaintyResult(param_names, param_values, stds, cov)
```

### 5.10 `visualization/plotter.py` — グラフ出力

#### `plot_calibration_summary`

`matplotlib.gridspec.GridSpec` による 3 段構成:

```
GridSpec(n_rows, 4)

Row 0: [XY 残差] [Y 残差] [Z 残差] [XY 散布図]
Row 1: [誤差ノルム時系列 (span 3)] [RMS 棒グラフ]
Row 2: [パラメータ ± 1σ (span 4)] ← uncertainty が与えられた場合のみ
```

`true_errors` が与えられると Row 2 に `×` マーカーで真値を重ね描きします。

### 5.11 `io/loader.py` — CSV 入出力

```python
load_joint_trajectory(path)       → (times: (N,), q_traj: (N, 6))
load_position_observations(path)  → (times: (N,), p_exp: (N, 3))
save_results(path, names, means, stds)   # 推定結果の CSV 保存
```

フォーマット仮定: 1 行目はヘッダ、区切り文字はカンマ。列順は固定（t, q0-q5 / t, px, py, pz）。

---

## 6. 識別可能性と縮退

### 6.1 完全縮退パラメータ（位置のみ観測）

`tool_rz` と `d_theta_offset_5` は常に完全縮退します：

```
FK(q, tool_rz=δ) ≡ FK(q, d_theta_offset_5=-δ)  （位置成分のみ）
```

位置スカラーに影響しない最終軸まわりの回転は分離不能です。  
**解決策**: `tool_rz` を `fixed=True` に設定し、すべての最終軸回転誤差を `d_theta_offset_5` に吸収させます。

### 6.2 ボールバー（距離スカラー観測）の識別制限

単一固定点からの距離 `r = ‖TCP - origin‖` は 1 スカラーのため、3D 位置観測より情報量が少ないです。

識別できないパラメータの典型例（Laplace 近似の σ が prior_std に近い）:

| パラメータ | 識別困難な理由 |
|-----------|---------------|
| `d_alpha_5` | 最終軸の捩り方向は距離に非感応 |
| `d_theta_offset_4` | 手首付近の回転は作業空間での距離変化が小さい |
| `d_d_1` ~ `d_d_3` | リンク長方向が視線方向と直交する姿勢で相殺 |

複数の固定原点を使うと（`MultiOriginDistanceObservation`、K 点の距離を同時観測）、`predict()` が `(K,)` を返すことでパイプラインの変更なく識別性が向上します。

### 6.3 関節伝達誤差の識別上の注意

`amp * sin(q + phase)` パラメータ化では `amp = 0` 付近で `phase` の勾配がゼロになります（`∂/∂phase [amp·sin(q+phase)] = amp·cos(q+phase) → 0`）。LM アルゴリズムはこの停留点を抜け出せず、`phase` が任意値に収束します。

**解決策**: フーリエ係数形式 `a·cos(q) + b·sin(q)` に変換します：

```
a = amp * sin(phase)
b = amp * cos(phase)

∂/∂a [a·cos(q) + b·sin(q)] = cos(q)   ← a=0 でも非ゼロ
∂/∂b [a·cos(q) + b·sin(q)] = sin(q)   ← b=0 でも非ゼロ
```

変換後の amp/phase の回復:
```python
amp   = sqrt(a² + b²)
phase = atan2(a, b)
```

### 6.4 DH 誤差と伝達誤差の相関

DH パラメータ（特に `d_theta_offset_i`）は、限られた関節可動範囲では伝達誤差 `a·cos(q) + b·sin(q)` と部分的に相関します（`cos(q)` の多項式近似との類似性）。

**対策**:
1. 関節掃引を先行させ、その後 DH を推定する（ステージ順序）
2. `final_full_tune=True` で最終的な全パラメータ同時収束を行う
3. 掃引範囲を可能な限り広く取る（`-π` 〜 `π`）

---

## 7. 設計判断の根拠

### 7.1 `params: dict` vs. `ParameterSet` 直接渡し

`KinematicModel.forward(q, params: dict)` では `ParameterSet` でなく Python 辞書を受け取ります。この選択の理由:

- サブクラスで追加パラメータ（`trans_err_a_0` など）を `params.get(key, 0.0)` で簡単に参照できる
- `ParameterSet` への依存をモデル層から排除（モデルは骨格コードを知らなくてよい）
- シリアライズ・コピーが辞書として容易

### 7.2 `method="lm"` の選択

`scipy` の `least_squares` は `method="lm"`（Levenberg-Marquardt）を使います。

- ロボットキャリブレーション問題は密なヤコビアン（全パラメータが全観測に影響）なため、疎行列を仮定する `trf` / `dogbox` より LM の方が適しています
- LM は上限・下限制約を扱えませんが、キャリブレーション問題では `prior_std` による正則化で十分です

### 7.3 `include_prior=False` での `compute_uncertainty`

不確かさ計算時に事前分布残差を除外する理由:

```
include_prior=True の場合:
  H_total = J_obs^T J_obs + Λ    (Λ = diag(1/σ_i²))
  Cov = H_total^{-1}             ← 事前分布が強いと σ_i が人工的に小さくなる

include_prior=False の場合:
  H_obs = J_obs^T J_obs
  Cov = H_obs^{-1}               ← データのみの不確かさを正しく反映
```

「データがどれだけ情報を与えているか」を評価するには事前情報を含めないのが正しいです。

### 7.4 `final_full_tune` の位置づけ

段階的推定はしばしば局所最適解に陥りやすいです（先行ステージが誤りを引き込むと後続ステージが補正しきれない）。`final_full_tune=True` は全パラメータを同時に解放して「最後の一押し」をします。

ただし `final_full_tune` も局所最適を保証しません。出発点が大きく外れている場合は Stage 順序の見直しが必要です。

### 7.5 `y_exp` の形状自由度

```python
y_exp_flat = np.asarray(y_exp).flatten()
```

この 1 行で `(N, 3)`, `(N, 1)`, `(N,)`, `(3N,)` のいずれでも同一処理となります。レシピ側で形状を意識しなくて済みます。

---

## 8. 既知の制約と留意事項

### 8.1 cubic 補間とタイムスタンプの連続性

```python
interp1d(q_timestamps, q_traj[:, j], kind="cubic", bounds_error=False,
          fill_value=(q_traj[0, j], q_traj[-1, j]))
```

- タイムスタンプが等間隔でなくても動作します
- **境界外**: 端点値で打ち切ります（`fill_value`）。評価時刻が `q_timestamps` の範囲外になる大きな `time_offset` では誤差が出ます
- **不連続点**: 関節角度に不連続がある場合（停止セグメントの終端など）は 3 次スプラインが大きな overshoot を生みます

### 8.2 ランク落ちと疑似逆行列

`laplace_uncertainty` は `LinAlgError` を捕捉して `pinv` で対処しますが、縮退している場合は特定の方向の `σ` が非常に大きい値（または `inf`/`nan`）になります。この場合は `fixed=True` で縮退パラメータを固定してから再評価してください。

### 8.3 数値精度とスケール

LM アルゴリズムはパラメータスケールの差が大きい場合（例: `d_alpha_i` の典型値 `~1e-3 rad` vs. `d_d_i` の典型値 `~1e-3 m`）に収束が遅くなる場合があります。`prior_std` を実際の誤差オーダーに合わせることで正則化がスケール補正の役割を持ちます。

### 8.4 旧 API との共存

`estimation/optimizer.py` の `run_staged_optimization()` と `models/residuals.py` の `compute_residuals()` は旧来の実装が残っています。これらは `RobotKinematics` を直接参照しており、カスタム `KinematicModel` サブクラスには対応しません。新しい実装では `pipeline.py` の `run_calibration()` を使ってください。

---

*本設計書は `robot_calibration/` の実装コードを正とします。コードと設計書に齟齬がある場合はコードが優先されます。*
