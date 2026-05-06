"""
キャリブレーションモデルの抽象基底クラス。

レシピファイルはこれを継承し、計算部分だけを実装する。
骨格側（pipeline.py）はこのインタフェースのみに依存し、
レシピ固有のクラス名・内部構造には一切依存しない。
"""

import numpy as np
from abc import ABC, abstractmethod


class KinematicModel(ABC):
    """
    順運動学モデルのインタフェース。

    params: dict は {パラメータ名: 値} の辞書。
    未定義キーは 0.0 としてデフォルト処理するよう実装側で .get() を使う。
    """

    @abstractmethod
    def forward(self, q: np.ndarray, params: dict) -> np.ndarray:
        """
        関節角度 q とパラメータ辞書から手先姿勢 T ∈ SE(3) (4×4) を返す。
        """
        ...

    @abstractmethod
    def jacobian(self, q: np.ndarray, params: dict) -> np.ndarray:
        """
        関節角度に対する位置ヤコビアン ∂p/∂q ∈ R^{3 × n_joints}。

        パスプランニングや特異点チェック、軌道生成に使用する。
        検証は numerical_jacobian() と比較する。
        """
        ...

    def numerical_jacobian(
        self,
        q: np.ndarray,
        params: dict,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """数値微分による位置ヤコビアン（解析的ヤコビアンの検証用）。"""
        J = np.zeros((3, len(q)))
        for i in range(len(q)):
            q_p = q.copy(); q_p[i] += eps
            q_m = q.copy(); q_m[i] -= eps
            p_p = self.forward(q_p, params)[:3, 3]
            p_m = self.forward(q_m, params)[:3, 3]
            J[:, i] = (p_p - p_m) / (2 * eps)
        return J


class ObservationModel(ABC):
    """
    観測モデルのインタフェース。

    手先姿勢 x（4×4 SE(3)）から観測予測値ベクトルを生成する。
    観測の種類（位置・姿勢・距離・角度など）をレシピ側で定義する。
    """

    @abstractmethod
    def predict(self, x: np.ndarray, params: dict) -> np.ndarray:
        """
        手先姿勢 x (4×4) とパラメータ辞書から観測予測値ベクトルを返す。

        戻り値の長さは観測の種類により異なる：
          位置のみ: (3,)
          距離スカラー: (1,)
          姿勢込み: (6,) など
        """
        ...
