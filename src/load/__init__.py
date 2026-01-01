"""
Core A: load (IARE - 知的取得・認識エンジン)

PaddleOCRを使用した手書き文書のOCR認識を行うモジュール。
"""

from src.load.ocr import OCREngine
from src.load.segmentation import CharacterSegmenter

__all__ = ["OCREngine", "CharacterSegmenter"]
