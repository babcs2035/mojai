"""
文字分割モジュール

純粋な画像処理ベースの谷点検出による文字分割を提供する。
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class CharacterSegmenter:
    """
    垂直投影プロファイルベースの文字分割

    画像の垂直投影から谷（局所最小値）を検出し、
    アノテーション文字数に基づいて文字境界を特定する。
    """

    def segment(self, line_binary: np.ndarray, num_chars: int) -> list[tuple[int, int]]:
        """
        行画像から文字境界を検出

        Args:
            line_binary: 二値化された行画像 (白=インク)
            num_chars: アノテーションの文字数

        Returns:
            文字境界のリスト [(x_start, x_end), ...]
        """
        height, width = line_binary.shape

        # 垂直投影
        projection = np.sum(line_binary, axis=0).astype(float)
        smoothed = gaussian_filter1d(projection, sigma=3)

        # インク領域の検出
        ink_threshold = np.max(smoothed) * 0.02
        ink_mask = smoothed > ink_threshold

        # インク領域の開始・終了を検出
        ink_start = 0
        ink_end = width
        for i in range(width):
            if ink_mask[i]:
                ink_start = max(0, i - 3)
                break
        for i in range(width - 1, -1, -1):
            if ink_mask[i]:
                ink_end = min(width, i + 3)
                break

        # インク領域が無効な場合はフォールバック
        if ink_end <= ink_start:
            return self._equal_divide(width, num_chars)

        ink_region = smoothed[ink_start:ink_end]

        # 谷 (局所最小値) を検出
        estimated_char_width = len(ink_region) / num_chars
        min_distance = max(5, int(estimated_char_width * 0.3))

        valleys, _ = find_peaks(
            -ink_region, distance=min_distance, prominence=np.max(ink_region) * 0.05
        )

        # 絶対座標に変換
        valleys = valleys + ink_start

        # 必要な境界数: num_chars - 1
        needed = num_chars - 1

        if len(valleys) >= needed:
            # 谷の深さでソートし、最も深い needed 個を選択
            depths = smoothed[valleys]
            sorted_indices = np.argsort(depths)[:needed]
            boundaries = sorted(valleys[sorted_indices])
        elif len(valleys) > 0:
            # 谷が足りない: 既存の谷を使いつつ、残りを補間
            boundaries = list(valleys)
            all_points = [ink_start] + list(boundaries) + [ink_end]
            while len(boundaries) < needed:
                gaps = [(all_points[i + 1] - all_points[i], i) for i in range(len(all_points) - 1)]
                gaps.sort(reverse=True)
                widest_idx = gaps[0][1]
                mid = (all_points[widest_idx] + all_points[widest_idx + 1]) // 2
                boundaries.append(mid)
                boundaries.sort()
                all_points = [ink_start] + list(boundaries) + [ink_end]
            boundaries = sorted(boundaries)
        else:
            # 谷が見つからない: 等分割
            ink_width = ink_end - ink_start
            boundaries = [ink_start + int(ink_width * (i + 1) / num_chars) for i in range(needed)]

        # 境界を (x_start, x_end) のリストに変換
        all_bounds = [ink_start] + list(boundaries) + [ink_end]
        return [(all_bounds[i], all_bounds[i + 1]) for i in range(num_chars)]

    def _equal_divide(self, width: int, num_chars: int) -> list[tuple[int, int]]:
        """等分割のフォールバック"""
        char_width = width / num_chars
        return [(int(i * char_width), int((i + 1) * char_width)) for i in range(num_chars)]
