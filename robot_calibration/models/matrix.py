"""
観測変換の具象実装。

  IdentityTransform      — 変換なし
  VelocityNormTransform  — 3D 軌道 → 速度ノルム時系列
  FFTAmplitudeTransform  — 残差ノルム → 周波数振幅（伝達誤差同定用）
"""

import numpy as np
from .base import ObservationTransform


class IdentityTransform(ObservationTransform):
    """変換なし。全パラメータ一括推定時に使用。"""
    transform_mode = "split"

    def apply(self, y: np.ndarray) -> np.ndarray:
        return y

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        return np.eye(len(y))


class VelocityNormTransform(ObservationTransform):
    """
    3次元軌道 y ∈ R^{N×3} を速度ノルムのスカラー時系列に変換する。

    変換後: s[t] = ||y[t+1] - y[t]|| / dt

    座標系ずれの影響を排除し、時刻ずれ Δt のみを感度よく推定できる。
    y の形状: (N*3,) のフラット配列（N 点 × 3次元）
    """
    transform_mode = "split"

    def __init__(self, dt: float = 1.0):
        self.dt = dt

    def apply(self, y: np.ndarray) -> np.ndarray:
        Y = y.reshape(-1, 3)
        diff = np.diff(Y, axis=0)            # (N-1, 3)
        return np.linalg.norm(diff, axis=1) / self.dt  # (N-1,)

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        """
        s[t] = ||Y[t+1] - Y[t]|| / dt

        ∂s[t]/∂Y[t+1, k] =  (Y[t+1,k] - Y[t,k]) / (s[t] * dt^2)
        ∂s[t]/∂Y[t,   k] = -(Y[t+1,k] - Y[t,k]) / (s[t] * dt^2)
        """
        Y = y.reshape(-1, 3)
        N = len(Y)
        M = N - 1
        J = np.zeros((M, N * 3))

        diff = np.diff(Y, axis=0)           # (M, 3)
        norms = np.linalg.norm(diff, axis=1)  # (M,)
        norms = np.where(norms < 1e-12, 1e-12, norms)  # ゼロ除算回避

        for t in range(M):
            for k in range(3):
                v = diff[t, k] / (norms[t] * self.dt)
                J[t, (t + 1) * 3 + k] =  v   # Y[t+1, k]
                J[t,  t      * 3 + k] = -v   # Y[t,   k]

        return J


class FFTAmplitudeTransform(ObservationTransform):
    """
    位置残差ノルムの時系列をFFTして周波数成分の振幅に変換する。

    用途：関節の角度伝達誤差（周期的誤差）のパラメータ推定。

    transform_mode = "residual" のため pipeline は
        r_raw = p_exp - p_pred  (N, 3)
        s[t]  = ||r_raw[t]||   スカラー時系列 (N,)
        r     = |FFT(s)|       ← これをゼロに近づける
    という処理を行う。
    """
    transform_mode = "residual"

    def apply(self, y: np.ndarray) -> np.ndarray:
        """y : スカラー時系列 (N,)  →  振幅スペクトル (N//2 + 1,)"""
        N = len(y)
        Y = np.fft.rfft(y) / N
        return np.abs(Y)

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        """
        |FFT(y)[k]| の y に対するヤコビアン。

        FFT(y)[k] = Σ_n y[n] exp(-2πi kn/N)
        連鎖律: ∂|Y_k|/∂y[n] = (Y_k.real * cos - Y_k.imag * sin) / (N * |Y_k|)
        """
        N = len(y)
        Y = np.fft.rfft(y) / N
        M = len(Y)  # N//2 + 1
        J = np.zeros((M, N))

        abs_Y = np.abs(Y)
        abs_Y = np.where(abs_Y < 1e-15, 1e-15, abs_Y)

        n = np.arange(N)
        for k in range(M):
            phase = 2 * np.pi * k * n / N
            J[k, :] = (Y[k].real * np.cos(phase) + Y[k].imag * (-np.sin(phase))) / (N * abs_Y[k])

        return J
