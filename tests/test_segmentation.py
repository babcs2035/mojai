"""
文字セグメンテーションモジュールのテスト
"""

import numpy as np
import pytest
from PIL import Image

from src.load.segmentation import CharacterSegmenter


@pytest.fixture
def segmenter():
    """セグメンターのフィクスチャ"""
    return CharacterSegmenter()


@pytest.fixture
def sample_line_image():
    """サンプル行画像のフィクスチャ (3文字分)"""
    # 白背景に黒い矩形を3つ配置
    width, height = 150, 50
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 3つの「文字」を配置
    for i in range(3):
        x_start = 10 + i * 50
        x_end = 40 + i * 50
        img[10:40, x_start:x_end] = 0  # 黒い矩形

    return img


def test_segmenter_init():
    """セグメンターの初期化テスト"""
    seg = CharacterSegmenter(min_char_width=20, min_gap_width=5)

    assert seg.min_char_width == 20
    assert seg.min_gap_width == 5


def test_segment_horizontal(segmenter, sample_line_image):
    """横書きセグメンテーションのテスト"""
    result = segmenter.segment(sample_line_image, orientation="horizontal")

    # 3つの文字が検出されるはず
    assert len(result) == 3

    # 各文字のバウンディングボックスが妥当か
    for char_box in result:
        assert char_box.width > 0
        assert char_box.height > 0
        assert char_box.image is not None


def test_segment_pil_image(segmenter, sample_line_image):
    """PIL Imageでの入力テスト"""
    pil_image = Image.fromarray(sample_line_image)
    result = segmenter.segment(pil_image, orientation="horizontal")

    assert len(result) > 0


def test_empty_image(segmenter):
    """空画像のテスト"""
    empty_img = np.ones((50, 150, 3), dtype=np.uint8) * 255
    result = segmenter.segment(empty_img, orientation="horizontal")

    # 空画像からは文字が検出されない
    assert len(result) == 0
