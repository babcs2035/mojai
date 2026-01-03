"""
前処理モジュール

手書き文字認識に最適化された画像前処理を提供する。
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


class Preprocessor:
    """
    高品質前処理パイプライン

    - Bilateral Filter: エッジ保存ノイズ除去
    - CLAHE: コントラスト正規化
    - Adaptive Thresholding: 局所的二値化
    """

    def __init__(self, block_size: int = 21, c_value: int = 10):
        self.block_size = block_size
        self.c_value = c_value

    def process(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        """
        画像を前処理し、表示用画像と二値化画像を返す
        """
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 1. Bilateral Filter: エッジを保存しながらノイズ除去
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. CLAHE: コントラスト正規化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Adaptive Thresholding: 局所的二値化
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.block_size,
            C=self.c_value,
        )

        # 4. モルフォロジー: 小さなノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return Image.fromarray(255 - binary), binary

    def detect_lines(self, binary: np.ndarray) -> list[tuple[int, int]]:
        """
        水平投影による行検出
        """
        # 水平投影
        projection = np.sum(binary, axis=1)
        smoothed = gaussian_filter1d(projection, sigma=3)

        # 動的閾値
        threshold = np.max(smoothed) * 0.05
        active = smoothed > threshold

        lines = []
        start = None
        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if (i - start) > 15:
                    lines.append((max(0, start - 5), min(binary.shape[0], i + 5)))
                start = None

        if start is not None and (binary.shape[0] - start) > 15:
            lines.append((max(0, start - 5), binary.shape[0]))

        if not lines:
            lines = [(0, binary.shape[0])]

        return lines
