"""
文字単位セグメンテーションモジュール

射影プロファイル法を使用して、行画像から個々の文字を分割する。
PaddleOCRの行レベル認識結果を、文字レベルに細分化する。
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class CharacterBox:
    """文字のバウンディングボックス"""

    x: int  # 左上X座標 (行画像内)
    y: int  # 左上Y座標 (行画像内)
    width: int  # 幅
    height: int  # 高さ
    image: np.ndarray  # 切り出された文字画像


class CharacterSegmenter:
    """
    射影プロファイル法による文字セグメンテーション

    計画書の「4.2 文字単位セグメンテーションの実装戦略」に基づく実装。
    Deep Learningの行検出能力と古典的画像処理の精密な文字切り出しを融合。
    """

    def __init__(
        self,
        min_char_width: int = 10,
        min_gap_width: int = 3,
        threshold_ratio: float = 0.1,
    ):
        """
        セグメンターを初期化

        Args:
            min_char_width: 最小文字幅 (これより小さい領域は無視)
            min_gap_width: 最小ギャップ幅 (これより小さい空白は文字内とみなす)
            threshold_ratio: 射影プロファイルの閾値比率
        """
        self.min_char_width = min_char_width
        self.min_gap_width = min_gap_width
        self.threshold_ratio = threshold_ratio

    def segment(
        self,
        line_image: np.ndarray | Image.Image,
        orientation: str = "horizontal",
    ) -> list[CharacterBox]:
        """
        行画像から文字を分割

        Args:
            line_image: 行画像 (BGR or RGB)
            orientation: 書き方向 ("horizontal": 横書き, "vertical": 縦書き)

        Returns:
            文字ボックスのリスト
        """
        # PIL Imageの場合はnumpy配列に変換
        if isinstance(line_image, Image.Image):
            line_image = np.array(line_image)

        # グレースケール変換
        if len(line_image.shape) == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()

        # 二値化 (Otsu法)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 射影プロファイルを計算
        if orientation == "horizontal":
            # 横書き: 垂直射影 (列ごとの白画素数)
            projection = np.sum(binary, axis=0)
        else:
            # 縦書き: 水平射影 (行ごとの白画素数)
            projection = np.sum(binary, axis=1)

        # 区間を抽出
        intervals = self._extract_intervals(projection)

        # 文字ボックスを生成
        char_boxes: list[CharacterBox] = []
        for start, end in intervals:
            if orientation == "horizontal":
                # 横書き: X方向で分割
                char_img = line_image[:, start:end]
                box = CharacterBox(
                    x=start,
                    y=0,
                    width=end - start,
                    height=line_image.shape[0],
                    image=char_img,
                )
            else:
                # 縦書き: Y方向で分割
                char_img = line_image[start:end, :]
                box = CharacterBox(
                    x=0,
                    y=start,
                    width=line_image.shape[1],
                    height=end - start,
                    image=char_img,
                )
            char_boxes.append(box)

        return char_boxes

    def _extract_intervals(self, projection: np.ndarray) -> list[tuple[int, int]]:
        """
        射影プロファイルから文字区間を抽出

        Args:
            projection: 射影プロファイル

        Returns:
            (開始位置, 終了位置) のタプルリスト
        """
        # 閾値を計算 (平均値の一定割合)
        threshold = np.mean(projection) * self.threshold_ratio

        # 閾値以上の位置をマーク
        is_character = projection > threshold

        # 連続する非ゼロ区間を抽出
        intervals: list[tuple[int, int]] = []
        in_char = False
        start = 0

        for i, is_char in enumerate(is_character):
            if is_char and not in_char:
                # 文字区間の開始
                start = i
                in_char = True
            elif not is_char and in_char:
                # 文字区間の終了
                if i - start >= self.min_char_width:
                    intervals.append((start, i))
                in_char = False

        # 最後の区間を処理
        if in_char and len(projection) - start >= self.min_char_width:
            intervals.append((start, len(projection)))

        # 短いギャップをマージ
        intervals = self._merge_short_gaps(intervals)

        return intervals

    def _merge_short_gaps(
        self,
        intervals: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        短いギャップを持つ隣接区間をマージ

        Args:
            intervals: 区間リスト

        Returns:
            マージ後の区間リスト
        """
        if len(intervals) <= 1:
            return intervals

        merged: list[tuple[int, int]] = [intervals[0]]

        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]

            if start - prev_end < self.min_gap_width:
                # ギャップが短い場合はマージ
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def segment_from_bbox(
        self,
        full_image: np.ndarray | Image.Image,
        bbox: list[tuple[float, float]],
        orientation: str = "horizontal",
    ) -> list[CharacterBox]:
        """
        全体画像とバウンディングボックスから文字を分割

        Args:
            full_image: 全体画像
            bbox: PaddleOCRのバウンディングボックス (4点座標)
            orientation: 書き方向

        Returns:
            文字ボックスのリスト (座標は全体画像基準に変換済み)
        """
        # PIL Imageの場合はnumpy配列に変換
        if isinstance(full_image, Image.Image):
            full_image = np.array(full_image)

        # バウンディングボックスから行画像を切り出し
        pts = np.array(bbox, dtype=np.float32)
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)

        # パディングを追加
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(full_image.shape[1], x_max + padding)
        y_max = min(full_image.shape[0], y_max + padding)

        line_image = full_image[y_min:y_max, x_min:x_max]

        # 文字分割
        char_boxes = self.segment(line_image, orientation)

        # 座標を全体画像基準に変換
        for box in char_boxes:
            box.x += x_min
            box.y += y_min

        return char_boxes
