# Robot Calibration Library

ロボットアームのキャリブレーションを行うPythonライブラリ。DHパラメータ誤差、ツール変換誤差、ローカル変換誤差、時刻ずれ、関節伝達誤差を最小二乗法で同定します。

## 特徴

- **段階的最適化**: パラメータをグループ化して順次推定
- **不確かさ評価**: ラプラス近似によるパラメータの標準偏差推定
- **柔軟な観測モデル**: 位置観測、速度ノルム、FFT周波数成分に対応
- **シミュレーションテスト**: 包括的な検証スイート

## インストール

### Conda環境のセットアップ

```bash
conda env create -f environment.yml
conda activate calib
```

### 依存関係

- Python >= 3.11
- NumPy
- SciPy
- Matplotlib

## 使い方

### 基本的なキャリブレーション手順

1. **データ準備**: 関節角度軌道と対応するエンドエフェクタ位置観測データをCSV形式で準備
2. **パラメータセット作成**: 推定対象パラメータを定義
3. **最適化実行**: 段階的最適化でパラメータを推定
4. **結果評価**: 残差改善とパラメータ不確かさを確認

### サンプルコード

```python
import numpy as np
from robot_calibration.models.parameters import make_default_parameter_set
from robot_calibration.models.kinematics import DH_NOMINAL  # または独自のDHパラメータ
from robot_calibration.estimation.optimizer import run_staged_optimization, default_stages
from robot_calibration.io.loader import load_joint_trajectory, load_position_observations

# 1. データ読み込み
q_times, q_traj = load_joint_trajectory("joint_trajectory.csv")  # (N, 6)
obs_times, p_exp = load_position_observations("position_observations.csv")  # (N, 3)

# 2. パラメータセット作成
ps = make_default_parameter_set(DH_NOMINAL)
param_lookup = {p.name: i for i, p in enumerate(ps.params)}

# 3. 最適化実行
stages = default_stages()  # デフォルトの3ステージ
results = run_staged_optimization(
    stages=stages,
    params=ps,
    dh_nominal=DH_NOMINAL,
    param_lookup=param_lookup,
    q_traj=q_traj,
    p_exp=p_exp,
    q_timestamps=q_times,  # time_offset推定時は必要
)

# 4. 結果表示
for result in results:
    print(f"{result.stage_name}: cost={result.cost:.6f}")

# 5. 不確かさ評価（オプション）
from robot_calibration.estimation.uncertainty import laplace_uncertainty
from scipy.optimize import least_squares

# 最終ステージのヤコビアンから不確かさを計算
def fun(x):
    from robot_calibration.models.residuals import compute_residuals
    free_idx = ps.free_indices()
    return compute_residuals(x, free_idx, ps, DH_NOMINAL, param_lookup, q_traj, p_exp)

res = least_squares(fun, ps.get_vector(ps.free_indices()))
uncertainty = laplace_uncertainty(res.jac, [p.name for p in ps.params if not p.fixed], res.x)
print(uncertainty.summary())
```

### データフォーマット

#### 関節角度軌道 (joint_trajectory.csv)
```csv
t,q0,q1,q2,q3,q4,q5
0.0,0.1,0.2,0.3,0.4,0.5,0.6
0.01,0.11,0.21,0.31,0.41,0.51,0.61
...
```

#### 位置観測 (position_observations.csv)
```csv
t,px,py,pz
0.0,0.1,0.2,0.3
0.01,0.11,0.21,0.31
...
```

### パラメータグループ

- `kinematic`: DHパラメータ誤差 (α, a, d, θ_offset)
- `tool`: ツール変換誤差 (tx, ty, tz, rx, ry, rz)
- `local`: ベース座標系誤差 (tx, ty, tz, rx, ry, rz)
- `time_offset`: 時刻ずれ
- `joint_transmission_error`: 関節伝達誤差 (振幅・位相)

### 観測変換

- `IdentityTransform`: 位置残差を直接使用
- `VelocityNormTransform`: 速度ノルムの時系列に変換（時刻ずれ推定に有効）
- `FFTAmplitudeTransform`: 残差ノルムのFFT周波数成分（周期的誤差推定に有効）

### 高度な使い方

#### カスタムステージ定義

```python
from robot_calibration.estimation.optimizer import Stage
from robot_calibration.models.transforms import VelocityNormTransform, IdentityTransform

stages = [
    Stage("time_offset", ["time_offset"], VelocityNormTransform(dt=0.01)),
    Stage("kinematics", ["kinematic", "tool", "local"], IdentityTransform()),
]
```

#### パラメータの固定/解放

```python
ps = make_default_parameter_set(DH_NOMINAL)
# 特定のツールパラメータを固定
ps.params[param_lookup["tool_tx"]].fixed = True
ps.params[param_lookup["tool_ty"]].fixed = True
```

## テスト実行

```bash
python -m robot_calibration.tests.simulation_test
```

テストでは以下の検証を行います：
- ヤコビアンの解析解と数値微分の一致
- 観測変換のヤコビアン検証
- DHパラメータヤコビアン検証
- 伝達誤差同定
- 軌道データからの同定
- 直線軌道データからの同定

テスト成功時は `output/` ディレクトリにプロット画像が保存されます。

## API リファレンス

### 主要クラス

- `RobotKinematics`: DHパラメータによる順運動学
- `ParameterSet`: 推定パラメータ管理
- `ObservationTransform`: 観測データの変換
- `Stage`: 最適化ステージ定義

### 主要関数

- `run_staged_optimization()`: 段階的最適化実行
- `laplace_uncertainty()`: 不確かさ評価
- `load_joint_trajectory()` / `load_position_observations()`: データ読み込み

## ライセンス

このプロジェクトはオープンソースです。適切なライセンスを追加してください。

## 貢献

バグ報告や機能リクエストはGitHub Issuesでお願いします。