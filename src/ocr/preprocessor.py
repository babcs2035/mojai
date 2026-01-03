"""
画像前処理モジュール．

手書き文字認識の精度を最大化するため，ノイズ除去，コントラスト強調，および適応的二値化を行なう．
また，二値化画像に基づいた行領域の検出機能を提供する．
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


class Preprocessor:
    """
    高品質な画像前処理パイプラインクラス．

    以下の手法を組み合わせ，ノイズの多い手書き文書から文字成分を抽出する：
    1. Bilateral Filter: エッジを保持したまま紙面のノイズを低減．
    2. CLAHE: 局所的なコントラストを補正し，文字の濃淡を均一化．
    3. Adaptive Thresholding: 照明ムラに対応した局所的な二値化．
    """

    def __init__(self, block_size: int = 21, c_value: int = 10):
        """
        プロセッサの初期化を行なう．

        Args:
            block_size (int): 二値化を行なう局所領域のサイズ．奇数である必要がある．
            c_value (int): 二値化の際の閾値微調整パラメータ．
        """
        self.block_size = block_size
        self.c_value = c_value

    def process(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        """
        画像に対して一連の前処理を適用する．

        Args:
            image (Image.Image): 入力画像．

        Returns:
            tuple[Image.Image, np.ndarray]: 表示用の白黒反転画像と，解析用の二値化配列．
        """
        # グレースケール変換
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 1. Bilateral Filter: エッジを保存しながら高周波ノイズを除去
        # 紙の質感や不要な汚れを抑制し，文字の輪郭を際立たせる
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization):
        # 局所的なヒストグラム平坦化を行ない，背景の照明ムラを抑えつつ文字のコントラストを最適化する
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Adaptive Thresholding:
        # 周辺画素の加重平均に基づいた動的な閾値処理を行なう．
        # 手書き文書特有の「部分的なかすれ」や「照明の偏り」に対して堅牢な二値化を実現する
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.block_size,
            C=self.c_value,
        )

        # 4. モルフォロジー演算（オープニング処理）:
        # 孤立点ノイズ（胡麻塩ノイズ）を除去し，文字の形状を整える
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return Image.fromarray(255 - binary), binary

    def detect_lines(self, binary: np.ndarray) -> list[tuple[int, int]]:
        """
        垂直方向への投影（水平積分）を用いて，文書内の行境界を特定する．

        Args:
            binary (np.ndarray): 前処理済みの二値化画像．

        Returns:
            list[tuple[int, int]]: 各行の開始座標と終了座標（y1, y2）のリスト．
        """
        # 水平方向への投影（各行のピクセル値を累積）
        projection = np.sum(binary, axis=1)
        # ガウシアンフィルタによる信号の平滑化．文字配置の疎密による変動を抑える
        smoothed = gaussian_filter1d(projection, sigma=3)

        # 全体のピーク値に基づいた動的な閾値設定
        threshold = np.max(smoothed) * 0.05
        active = smoothed > threshold

        lines = []
        start = None
        for i, val in enumerate(active):
            # 文字成分の開始点
            if val and start is None:
                start = i
            # 文字成分の終了点
            elif not val and start is not None:
                # 一定以上の高さを持つ領域のみを行として認定
                if (i - start) > 15:
                    # 上下に 5 ピクセルのマージンを付与
                    lines.append((max(0, start - 5), min(binary.shape[0], i + 5)))
                start = None

        # 最終行の処理
        if start is not None and (binary.shape[0] - start) > 15:
            lines.append((max(0, start - 5), binary.shape[0]))

        # 行が検出されなかった場合は画像全体を一文字行として扱う（フォールバック）
        if not lines:
            lines = [(0, binary.shape[0])]

        return lines
