# Robot Calibration Library

ロボットアームのキャリブレーションを行う Python ライブラリ。  
DH パラメータ誤差・ツール変換誤差・ベース座標系誤差・時刻ずれ・関節伝達誤差を最小二乗法で同定します。  
ラプラス近似による不確かさ評価・逐次推定収束分析・Before/After サマリーグラフ出力に対応しています。

---

## アーキテクチャ

実験ごとに **レシピファイル** (`recipe_*.py`) を 1 つ作成するだけで実験が完結します。  
骨格コード (`estimation/`, `io/`, `visualization/`) は一切触りません。

```
calib/
├── recipe_ur5_calibration.py      ← ランダム軌道からの DH / ツール / ベース誤差同定
├── recipe_straight_line_calib.py  ← 直線軌道 + 時刻ずれ同定
├── recipe_ballbar.py              ← ボールバー距離計測による同定・識別可能性評価
├── recipe_transmission_error.py   ← 関節伝達誤差を含む段階的キャリブレーション
├── recipe_external_model.py       ← カスタム KinematicModel のサンプル
└── robot_calibration/
    ├── pipeline.py                ← run_calibration / compute_uncertainty / run_sequential_calibration
    ├── models/
    │   ├── base.py                ← KinematicModel / ObservationModel / ObservationTransform (ABC)
    │   ├── kinematics.py          ← DHKinematics / RobotKinematics / DHParams
    │   ├── observation.py         ← PoseObservation / DistanceObservation / vec6_to_se3
    │   ├── matrix.py              ← IdentityTransform / VelocityNormTransform / FFTAmplitudeTransform
    │   └── parameters.py          ← Parameter / ParameterSet
    ├── estimation/
    │   ├── optimizer.py           ← Stage / StageResult
    │   └── uncertainty.py         ← laplace_uncertainty / UncertaintyResult / propagate_to_position
    ├── visualization/
    │   └── plotter.py             ← plot_calibration_summary / plot_sequential_convergence など
    └── io/
        └── loader.py              ← CSV 読み込み / 保存
```

---

## インストール

```bash
conda env create -f environment.yml
conda activate calib
```

**依存関係**: Python >= 3.11, NumPy, SciPy, Matplotlib

---

## 典型的なワークフロー

```python
import numpy as np
from robot_calibration import (
    run_calibration, compute_uncertainty,
    DHKinematics, PoseObservation, Parameter,
)
from robot_calibration.models.matrix import IdentityTransform
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.visualization.plotter import plot_calibration_summary

# 1. ロボット・モデル定義
DH_NOMINAL = [...]   # Modified DH パラメータ（6 リンク分）
KIN = DHKinematics(DH_NOMINAL)
OBS = PoseObservation()

# 2. 推定パラメータ定義
PARAMETERS = [
    *[Parameter(f"d_alpha_{i}", value=0.0, group="kinematic", prior_std=np.pi) for i in range(6)],
    *[Parameter(f"d_a_{i}",     value=0.0, group="kinematic", prior_std=1.0)   for i in range(6)],
    *[Parameter(f"d_d_{i}",     value=0.0, group="kinematic", prior_std=1.0)   for i in range(6)],
    *[Parameter(f"d_theta_offset_{i}", value=0.0, group="kinematic", prior_std=np.pi) for i in range(6)],
    Parameter("tool_tx", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_ty", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_tz", value=0.0, group="tool",  prior_std=1.0),
    Parameter("tool_rx", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_ry", value=0.0, group="tool",  prior_std=np.pi),
    Parameter("tool_rz", value=0.0, group="tool",  prior_std=np.pi, fixed=True),  # 縮退 → 固定
    Parameter("local_tx", value=0.0, group="local", prior_std=1.0),
    Parameter("local_ty", value=0.0, group="local", prior_std=1.0),
    Parameter("local_tz", value=0.0, group="local", prior_std=1.0),
    Parameter("local_rx", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_ry", value=0.0, group="local", prior_std=np.pi),
    Parameter("local_rz", value=0.0, group="local", prior_std=np.pi),
]

# 3. データ読み込み
from robot_calibration.io.loader import load_joint_trajectory, load_position_observations
q_times, q_traj = load_joint_trajectory("joint_angles.csv")      # (N,), (N, 6)
_,       y_exp  = load_position_observations("tcp_positions.csv") # (N,), (N, 3)

# 4. 推定実行
STAGES = [Stage("calib", param_groups=["kinematic","tool","local"], transform=IdentityTransform())]
params_result, stage_results = run_calibration(
    q_traj=q_traj, y_exp=y_exp,
    parameters=PARAMETERS, kinematic_model=KIN, observation_model=OBS,
    stages=STAGES,
)
print(params_result.summary())

# 5. 不確かさ評価（ラプラス近似 + σ_noise² スケーリング）
uncertainty = compute_uncertainty(
    params=params_result, kin_model=KIN, obs_model=OBS,
    q_traj=q_traj, y_exp=y_exp,
)
print(uncertainty.summary())   # 各パラメータの推定値 ± 1σ

# 6. Before/After サマリーグラフ出力
calib_dict   = {p.name: p.value for p in params_result.params}
nominal_dict = {p.name: 0.0     for p in params_result.params}
p_before = np.array([OBS.predict(KIN.forward(q_traj[i], nominal_dict), {}) for i in range(len(q_traj))])
p_after  = np.array([OBS.predict(KIN.forward(q_traj[i], calib_dict),   {}) for i in range(len(q_traj))])

fig = plot_calibration_summary(
    p_exp=y_exp, p_pred_before=p_before, p_pred_after=p_after,
    uncertainty=uncertainty,
    save_path="output/summary.png",
)
```

---

## サンプルレシピ

| ファイル | 内容 | 主な特徴 |
|---|---|---|
| [recipe_ur5_calibration.py](recipe_ur5_calibration.py) | ランダム姿勢 100 点からの DH / ツール / ベース誤差同定 | 基本ワークフロー・ラプラス近似 |
| [recipe_straight_line_calib.py](recipe_straight_line_calib.py) | 直線軌道 + 時刻ずれ同定 | 時刻ずれ推定・2 ステージ |
| [recipe_ballbar.py](recipe_ballbar.py) | ボールバー距離計測（スカラー観測）による同定 | `DistanceObservation`・識別可能性の可視化 |
| [recipe_transmission_error.py](recipe_transmission_error.py) | 関節伝達誤差を含む段階的キャリブレーション | カスタム `KinematicModel` サブクラス |
| [recipe_external_model.py](recipe_external_model.py) | 外部 FK モデルとの統合 | カスタム観測モデル |

```bash
python recipe_ur5_calibration.py
python recipe_straight_line_calib.py
python recipe_ballbar.py
python recipe_transmission_error.py
```

いずれも `output/` ディレクトリに Before/After サマリーグラフを保存します。

---

## `run_calibration` API

```python
params_result, stage_results = run_calibration(
    q_traj            : np.ndarray,                # (N, n_joints) 関節角度 [rad]
    y_exp             : np.ndarray,                # (N, obs_dim) または (N*obs_dim,) 観測値
    parameters        : list[Parameter],           # 推定パラメータリスト
    kinematic_model   : KinematicModel,            # 順運動学モデル
    observation_model : ObservationModel,          # 観測モデル
    stages            : list[Stage],               # 推定ステージリスト
    ls_kwargs         : dict | None = None,        # scipy.optimize.least_squares への追加引数
    q_timestamps      : np.ndarray | None = None,  # タイムスタンプ (N,) — time_offset 推定時に必要
    final_full_tune   : bool = False,              # 全ステージ後に全パラメータで最終調整
) -> tuple[ParameterSet, list[StageResult]]
```

内部で `forward_batch` / `predict_batch` を使った一括 FK 計算を行い、Python ループ逐次実行に対して **約 30 倍** の高速化を達成しています。

## `compute_uncertainty` API

```python
uncertainty = compute_uncertainty(
    params            : ParameterSet,              # run_calibration の戻り値
    kin_model         : KinematicModel,
    obs_model         : ObservationModel,
    q_traj            : np.ndarray,
    y_exp             : np.ndarray,
    transform         : ObservationTransform | None = None,  # 省略時 IdentityTransform
    q_timestamps      : np.ndarray | None = None,
) -> UncertaintyResult   # .param_names, .means, .stds, .cov
```

ヤコビアン `J` を最適解で評価し、`Cov ≈ σ_noise² (J^T J)^{-1}` からパラメータ標準偏差を計算します。  
`σ_noise²` は残差から自由度補正付きで推定されるため、スケールが正しく評価されます。  
`σ` が大きいパラメータは軌道から同定しにくい（情報不足・縮退に近い）ことを意味します。

## `run_sequential_calibration` API

データを `n_groups` グループに等分割し、累積データ量を増やしながら逐次推定を行います。  
各ステップの推定値・不確かさ・**手先位置不確かさ** を記録し、データ数に対する収束曲線を可視化できます。

```python
steps = run_sequential_calibration(
    q_traj            : np.ndarray,
    y_exp             : np.ndarray,
    parameters        : list[Parameter],
    kinematic_model   : KinematicModel,
    observation_model : ObservationModel,
    stages            : list[Stage],
    n_groups          : int = 10,               # 分割数
    ls_kwargs         : dict | None = None,
    q_timestamps      : np.ndarray | None = None,
    update_prior      : bool = True,            # 前ステップの事後分布を次の事前分布に設定
    q_eval            : np.ndarray | None = None,  # 位置不確かさの評価点
    n_eval            : int = 100,
) -> list[SequentialStep]
```

`update_prior=True`（デフォルト）のとき、前ステップの推定値と標準偏差が次ステップの事前分布として引き継がれます（逐次ベイズ推定）。

```python
from robot_calibration.visualization.plotter import plot_sequential_convergence

fig = plot_sequential_convergence(
    steps,
    param_filter=["d_a_1", "tool_tx", "local_ry"],  # 下段に表示するパラメータ
    true_values=TRUE_ERRORS,                         # 真値（シミュレーション時）
    save_path="output/sequential_convergence.png",
)
```

生成されるグラフ構成:

| 段 | 内容 |
|---|---|
| 上段 | 手先位置不確かさ `σ_pos = √tr(J Cov_θ Jᵀ)` の平均値 vs データ数 |
| 中段左 | XYZ 軸ごとの σ vs データ数 |
| 中段右 | 観測残差 RMS vs データ数（参考） |
| 下段 | パラメータ推定値 ± 1σ vs データ数（`param_filter` で絞り込み） |

---

## パラメータ定義

```python
Parameter(
    name      : str,          # DHKinematics が認識するキー名と対応
    value     : float,        # 初期値
    group     : str,          # "kinematic" | "tool" | "local" | "time_offset" | 任意
    prior_std : float,        # 事前分布の標準偏差（大きいほど非情報的）
    fixed     : bool = False, # True にすると推定から除外
)
```

### DHKinematics が認識するパラメータ名

| グループ | パラメータ名 | 意味 |
|---|---|---|
| `kinematic` | `d_alpha_i`, `d_a_i`, `d_d_i`, `d_theta_offset_i` | DH パラメータ誤差（i=0..n-1） |
| `tool` | `tool_tx/ty/tz` [m], `tool_rx/ry/rz` [rad] | ツール変換誤差 |
| `local` | `local_tx/ty/tz` [m], `local_rx/ry/rz` [rad] | ベース座標系誤差 |
| `time_offset` | `time_offset` [s] | 制御と観測の時刻ずれ |

### 縮退パラメータの扱い

位置のみ観測の場合、`tool_rz` と `d_theta_offset_5`（最終軸まわり回転）は完全縮退で分離不能です。  
慣例として `tool_rz` を `fixed=True` に設定し、効果を `d_theta_offset_5` に吸収させます。

---

## 時刻ずれ推定（`time_offset`）

ロボットコントローラと外部計測器のクロックずれを同定します。  
`q_timestamps` を渡すと、パイプライン内で cubic 補間を使い `q_eff(t) = q(t + time_offset)` として FK を評価します。

```python
PARAMETERS += [Parameter("time_offset", value=0.0, group="time_offset", prior_std=1.0)]

STAGES = [
    Stage("stage1_time_offset",
          param_groups=["time_offset"],
          transform=IdentityTransform()),
    Stage("stage2_kinematics",
          param_groups=["kinematic", "tool", "local"],
          transform=IdentityTransform()),
]

params_result, _ = run_calibration(
    ..., stages=STAGES, q_timestamps=q_timestamps,
)
```

> **注意**: `q_timestamps` に渡す関節角度時系列は **C1 連続**（速度連続）である必要があります。  
> ライン間に不連続な停止区間を挿入すると cubic 補間がスパイクを生じます。

---

## ボールバー（距離計測）による識別可能性評価

`DistanceObservation` を使うと、固定点からの距離スカラー（1 次元）のみで同定を行えます。

```python
from robot_calibration import DistanceObservation

ORIGIN = np.array([0.50, 0.00, 0.30])   # ボールバー固定端 [m]
OBS    = DistanceObservation(origin=ORIGIN)
```

Laplace 近似の `σ` を観察することで、**スカラー距離測定では同定困難なパラメータ**（σ が大きい）が一目でわかります。

| 典型的な観測 | 識別しやすい誤差 | 識別しにくい誤差 |
|---|---|---|
| 3D 位置（`PoseObservation`） | すべての DH / ツール / ベース誤差 | 最終軸回転（tool_rz ↔ d_theta_offset_5） |
| 距離スカラー（`DistanceObservation`） | 平行移動誤差、arm length 誤差 | 視線方向に直交する回転誤差、後段の軸誤差 |

---

## 関節伝達誤差モデル

ギアの周期的誤差などを模擬する伝達誤差モデルは、`DHKinematics` をサブクラス化して実装します。

### フーリエ係数形式（推奨）

`amp * sin(q + phase)` の直接パラメータ化は `amp=0` 付近で `phase` の勾配がゼロになり LM が停留します。  
代わりに **フーリエ係数 (a, b)** で線形パラメータ化します。

```python
class DHKinematicsWithTransmissionError(DHKinematics):
    """q_actual[i] = q[i] + a_i * cos(q[i]) + b_i * sin(q[i])"""
    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        q_eff = q.copy()
        for i in range(len(q)):
            a = params.get(f"trans_err_a_{i}", 0.0)
            b = params.get(f"trans_err_b_{i}", 0.0)
            q_eff[i] += a * np.cos(q[i]) + b * np.sin(q[i])
        return super().forward(q_eff, params)

PARAMETERS += [
    Parameter("trans_err_a_0", value=0.0, group="transmission_0", prior_std=0.1),
    Parameter("trans_err_b_0", value=0.0, group="transmission_0", prior_std=0.1),
]

# 推定後に amp/phase に変換
est_amp   = np.sqrt(est_a**2 + est_b**2)
est_phase = np.arctan2(est_a, est_b)
```

---

## 推定ステージ定義

```python
Stage(
    name         : str,              # ステージ名（ログ表示用）
    param_groups : list[str] | None, # 推定対象グループ（None = 全パラメータ）
    transform    : ObservationTransform,
)
```

---

## 観測変換

| クラス | 用途 |
|---|---|
| `IdentityTransform` | 位置残差を直接使用（汎用・`time_offset` 推定にも最適） |
| `VelocityNormTransform(dt)` | 速度ノルム時系列に変換。単軸回転運動での `time_offset` 推定に有効 |
| `FFTAmplitudeTransform` | 残差ノルムの FFT 振幅。周期的伝達誤差推定に有効 |

---

## カスタムモデルの実装

`DHKinematics` / `PoseObservation` で対応できない場合はサブクラス化します。

```python
from robot_calibration import KinematicModel, ObservationModel
import numpy as np

class MyKinematics(KinematicModel):
    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        # params は {名前: 値} の辞書。未定義キーは .get(key, 0.0) で取る
        return T   # (4, 4) SE(3)

    def forward_batch(self, q_batch: np.ndarray, params: dict) -> np.ndarray:
        # デフォルト実装（ループ）もあるが、オーバーライドで高速化できる
        return np.stack([self.forward(q, params) for q in q_batch])

class MyObservation(ObservationModel):
    def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
        return x[:3, 3]   # (3,) を返せば 3D 位置観測として処理される

    def predict_batch(self, poses: np.ndarray, params: dict) -> np.ndarray:
        return poses[:, :3, 3].flatten()   # (N*3,)
```

`forward_batch` / `predict_batch` をオーバーライドすると最適化ループが高速になります。デフォルト実装（ループ）も用意されているため、オーバーライドは必須ではありません。

---

## データフォーマット

### 関節角度軌道 (`joint_angles.csv`)

```
t,q0,q1,q2,q3,q4,q5
0.000,0.1,0.2,0.3,0.4,0.5,0.6
0.002,0.11,0.21,0.31,0.41,0.51,0.61
```

### TCP 位置観測 (`tcp_positions.csv`)

```
t,px,py,pz
0.000,0.412,0.031,0.553
0.002,0.413,0.031,0.554
```

```python
from robot_calibration.io.loader import load_joint_trajectory, load_position_observations

q_times, q_traj = load_joint_trajectory("joint_angles.csv")      # → (N,), (N, 6)
_,       y_exp  = load_position_observations("tcp_positions.csv") # → (N,), (N, 3)
```

---

## テスト実行

```bash
python -m robot_calibration.tests.simulation_test
```

| テスト名 | 内容 |
|---|---|
| `test_jacobian` | 解析的ヤコビアンと数値微分の一致確認 |
| `test_transform_jacobians` | VelocityNorm / FFTAmplitude 変換のヤコビアン確認 |
| `test_dh_jacobian` | DH パラメータ誤差ヤコビアンの確認 |
| `test_transmission_error_identification` | 周期的関節誤差の同定 |
| `test_identification` | 100 点ランダム姿勢から DH / ツール / ベース誤差を同定 |
| `test_trajectory_identification` | 6 関節サイン波軌道 + 時刻ずれ同定（20 ms 精度 < 5 ms） |
| `test_straight_line_identification` | 直線軌道 9 本から DH / ツール / ベース誤差同定 |

テスト成功時、プロット画像が `robot_calibration/tests/output/` に保存されます。

---

## API リファレンス

### `robot_calibration` トップレベルエクスポート

| 名前 | 種別 | 説明 |
|---|---|---|
| `run_calibration` | 関数 | 段階的キャリブレーションパイプライン |
| `compute_uncertainty` | 関数 | ラプラス近似による不確かさ評価（σ_noise² スケーリング付き） |
| `run_sequential_calibration` | 関数 | データを分割して逐次推定・収束分析 |
| `SequentialStep` | dataclass | 逐次推定の 1 ステップ結果（param_values, param_stds, pos_unc_mean, pos_unc_xyz, residual_rms） |
| `KinematicModel` | ABC | 順運動学モデルのインタフェース（`forward`, `forward_batch`） |
| `ObservationModel` | ABC | 観測モデルのインタフェース（`predict`, `predict_batch`） |
| `DHKinematics` | クラス | Modified DH パラメータによる順運動学 |
| `PoseObservation` | クラス | 3D 位置観測モデル |
| `DistanceObservation` | クラス | 固定点からの距離スカラー観測モデル |
| `Parameter` | クラス | 推定パラメータ（name, value, group, prior_std, fixed） |
| `ParameterSet` | クラス | パラメータコレクション（`.summary()`, `.free_indices()` など） |
| `Stage` | クラス | 推定ステージ定義（name, param_groups, transform） |
| `StageResult` | クラス | ステージ結果（cost, success, message, jacobian） |
| `UncertaintyResult` | クラス | 不確かさ評価結果（param_names, means, stds, cov） |
