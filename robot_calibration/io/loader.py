"""
CSV読み込み。

想定フォーマット:
  関節角度軌道: t, q0, q1, q2, q3, q4, q5
  観測位置:     t, px, py, pz
"""

import numpy as np
import csv
from pathlib import Path


def load_joint_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    関節角度軌道 CSV を読み込む。

    Returns:
        times  : (N,) 時刻 [s]
        q_traj : (N, 6) 関節角度 [rad]
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    times  = data[:, 0]
    q_traj = data[:, 1:7]
    return times, q_traj


def load_position_observations(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    位置観測 CSV を読み込む。

    Returns:
        times : (N,) 時刻 [s]
        p_exp : (N, 3) 観測位置 [m]
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    times = data[:, 0]
    p_exp = data[:, 1:4]
    return times, p_exp


def save_results(path: str | Path, param_names: list[str], means: np.ndarray, stds: np.ndarray) -> None:
    """推定結果を CSV に保存する。"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "mean", "std"])
        for name, mean, std in zip(param_names, means, stds):
            writer.writerow([name, f"{mean:.8f}", f"{std:.8f}"])
