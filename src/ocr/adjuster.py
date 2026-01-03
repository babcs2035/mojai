"""
境界調整モジュール

検証結果に基づいて文字境界を調整するアルゴリズムを提供する。
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class BoundaryAdjuster:
    """
    検証結果に基づく境界調整

    - 分離エラー (認識が1文字未満): 隣接領域とマージ
    - 合体エラー (認識が2文字以上): 領域を分割
    """

    def adjust(
        self,
        boundaries: list[tuple[int, int]],
        line_binary: np.ndarray,
        expected_chars: list[str],
        recognition_results: list[tuple[bool, str]],
    ) -> list[tuple[int, int]]:
        """
        検証結果に基づいて境界を調整

        Args:
            boundaries: 現在の境界 [(x_start, x_end), ...]
            line_binary: 二値化された行画像
            expected_chars: 期待される文字リスト
            recognition_results: [(一致したか, 認識結果), ...]

        Returns:
            調整後の境界
        """
        adjusted = []
        i = 0

        while i < len(expected_chars):
            if i >= len(boundaries):
                break

            is_match, recognized = recognition_results[i]
            current_bound = boundaries[i]

            if is_match:
                # 正解: そのまま
                adjusted.append(current_bound)
                i += 1
            elif len(recognized) == 0 or recognized == "":
                # 認識失敗: 隣とマージ試行
                if i + 1 < len(boundaries):
                    merged = self._merge_boundaries(current_bound, boundaries[i + 1])
                    adjusted.append(merged)
                    i += 2  # 2文字分消費するので次の expected も1つスキップ
                else:
                    adjusted.append(current_bound)
                    i += 1
            elif len(recognized) > 1:
                # 複数文字認識: 分割試行
                sub_bounds = self._split_boundary(current_bound, line_binary, len(recognized))
                adjusted.extend(sub_bounds)
                i += 1
            else:
                # 1文字だが不一致: そのまま（文字の誤認識は境界問題ではない可能性）
                adjusted.append(current_bound)
                i += 1

        return adjusted

    def _merge_boundaries(
        self, bound1: tuple[int, int], bound2: tuple[int, int]
    ) -> tuple[int, int]:
        """2つの境界をマージ"""
        return (bound1[0], bound2[1])

    def _split_boundary(
        self, bound: tuple[int, int], line_binary: np.ndarray, num_splits: int
    ) -> list[tuple[int, int]]:
        """
        境界を谷点検出で分割

        Args:
            bound: 分割する境界 (x_start, x_end)
            line_binary: 二値化された行画像
            num_splits: 分割数

        Returns:
            分割後の境界リスト
        """
        x_start, x_end = bound
        width = x_end - x_start

        if width < num_splits * 5:
            # 幅が狭すぎる場合は等分割
            char_width = width / num_splits
            return [
                (int(x_start + i * char_width), int(x_start + (i + 1) * char_width))
                for i in range(num_splits)
            ]

        # この領域の垂直投影
        region = line_binary[:, x_start:x_end]
        projection = np.sum(region, axis=0).astype(float)
        smoothed = gaussian_filter1d(projection, sigma=2)

        # 谷点検出
        min_distance = max(3, width // (num_splits + 1))
        valleys, _ = find_peaks(-smoothed, distance=min_distance)

        # 必要な分割点数
        needed = num_splits - 1

        if len(valleys) >= needed:
            # 最も深い谷を選択
            depths = smoothed[valleys]
            sorted_indices = np.argsort(depths)[:needed]
            split_points = sorted(valleys[sorted_indices])
        else:
            # 等分割
            split_points = [int(width * (i + 1) / num_splits) for i in range(needed)]

        # 境界リストを構築
        result = []
        prev = 0
        for sp in split_points:
            result.append((x_start + prev, x_start + sp))
            prev = sp
        result.append((x_start + prev, x_end))

        return result


class IterativeRefiner:
    """
    反復的境界調整

    検証→調整→再検証を繰り返して精度を向上させる。
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.adjuster = BoundaryAdjuster()

    def refine(
        self,
        initial_boundaries: list[tuple[int, int]],
        line_binary: np.ndarray,
        line_image: "Image.Image",
        expected_chars: list[str],
        verifier: "CharacterVerifier",
    ) -> list[tuple[int, int]]:
        """
        反復的に境界を調整

        Args:
            initial_boundaries: 初期境界
            line_binary: 二値化された行画像
            line_image: 元の行画像
            expected_chars: 期待される文字リスト
            verifier: 検証器

        Returns:
            最終的な境界
        """
        from PIL import Image

        boundaries = initial_boundaries
        best_boundaries = boundaries
        best_accuracy = 0.0

        for iteration in range(self.max_iterations):
            # 文字画像を切り出し
            char_images = []
            for x_start, x_end in boundaries:
                if x_end <= x_start:
                    char_images.append(Image.new("RGB", (10, 10), "white"))
                else:
                    char_images.append(line_image.crop((x_start, 0, x_end, line_image.height)))

            # 検証
            if len(char_images) != len(expected_chars):
                # 境界数が不一致の場合は調整をスキップ
                break

            results = verifier.verify_batch(char_images, expected_chars)

            # 精度計算
            matches = sum(1 for is_match, _ in results if is_match)
            accuracy = matches / len(expected_chars) if expected_chars else 0

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_boundaries = boundaries

            if accuracy >= 1.0:
                # 完全一致: 終了
                break

            # 境界調整
            boundaries = self.adjuster.adjust(boundaries, line_binary, expected_chars, results)

            # 境界数が変わった場合は調整をリセット
            if len(boundaries) != len(expected_chars):
                boundaries = best_boundaries
                break

        return best_boundaries
