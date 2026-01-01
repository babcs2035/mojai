"""
PaddleOCR ラッパーモジュール

PP-OCRv4モデルを使用した日本語OCR認識を提供する。
縦書き/横書きの両方に対応。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import settings


@dataclass
class OCRResult:
    """OCR認識結果"""

    text: str  # 認識されたテキスト
    confidence: float  # 確信度 (0.0-1.0)
    bbox: list[tuple[float, float]]  # バウンディングボックス (4点の座標)
    line_index: int  # 行インデックス


@dataclass
class LineResult:
    """行レベルのOCR結果"""

    text: str  # 行全体のテキスト
    confidence: float  # 平均確信度
    bbox: list[tuple[float, float]]  # 行のバウンディングボックス
    characters: list[OCRResult]  # 文字レベルの結果


class OCREngine:
    """
    PaddleOCRを使用したOCRエンジン

    PP-OCRv4アーキテクチャを使用し、日本語の縦書き/横書きに対応。
    RTX 3090上でのGPU推論に最適化。
    """

    def __init__(
        self,
        lang: str | None = None,
        det_model_dir: Path | None = None,
        rec_model_dir: Path | None = None,
    ):
        """
        OCRエンジンを初期化

        Args:
            lang: 言語設定 (デフォルト: japan)
            det_model_dir: 検出モデルディレクトリ
            rec_model_dir: 認識モデルディレクトリ
        """
        self.lang = lang or settings.ocr_lang
        self.det_model_dir = det_model_dir or settings.ocr_det_model_dir
        self.rec_model_dir = rec_model_dir or settings.ocr_rec_model_dir

        self._engine: Any = None

    def _init_engine(self) -> None:
        """PaddleOCRエンジンを遅延初期化"""
        if self._engine is not None:
            return

        from paddleocr import PaddleOCR

        # PaddleOCR v3.x API対応
        # 注: v3.xではuse_gpu, show_logパラメータは廃止
        kwargs: dict[str, Any] = {
            "lang": self.lang,
        }

        if self.det_model_dir:
            kwargs["det_model_dir"] = str(self.det_model_dir)
        if self.rec_model_dir:
            kwargs["rec_model_dir"] = str(self.rec_model_dir)

        self._engine = PaddleOCR(**kwargs)

    def recognize(self, image: Image.Image | np.ndarray | Path | str) -> list[LineResult]:
        """
        画像からテキストを認識

        Args:
            image: 入力画像 (PIL Image, numpy配列, またはファイルパス)

        Returns:
            行レベルのOCR結果リスト
        """
        self._init_engine()

        # 入力を適切な形式に変換
        if isinstance(image, (Path, str)):
            img_input = str(image)
        elif isinstance(image, Image.Image):
            img_input = np.array(image)
        else:
            img_input = image

        # OCR実行
        result = self._engine.ocr(img_input, cls=True)

        if result is None or len(result) == 0:
            return []

        # 結果を解析
        lines: list[LineResult] = []
        for line_idx, line_data in enumerate(result[0] or []):
            if line_data is None:
                continue

            bbox, (text, confidence) = line_data

            line_result = LineResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
                characters=[],  # 文字単位の結果はsegmentationモジュールで追加
            )
            lines.append(line_result)

        return lines

    def recognize_batch(
        self,
        images: list[Image.Image | np.ndarray | Path | str],
    ) -> list[list[LineResult]]:
        """
        複数画像をバッチ処理

        Args:
            images: 入力画像のリスト

        Returns:
            各画像のOCR結果リスト
        """
        return [self.recognize(img) for img in images]

    def release(self) -> None:
        """
        GPUメモリを解放

        生成フェーズ(Core C)に移行する前に呼び出す。
        """
        if self._engine is not None:
            del self._engine
            self._engine = None

            # CUDAメモリを解放
            try:
                import paddle

                paddle.device.cuda.empty_cache()
            except Exception:
                pass
