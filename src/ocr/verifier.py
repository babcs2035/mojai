"""
文字認識検証モジュール

PaddleOCR を使用して切り出した文字画像が正しいかを検証する。
"""

import numpy as np

# PaddleOCR 3.x の新しいAPI
from paddleocr import PaddleOCR
from PIL import Image


class CharacterVerifier:
    """
    PaddleOCR ベースの文字認識検証

    PP-OCRv5 の高精度日本語手書き認識を活用し、
    切り出した文字画像をOCRで認識してアノテーションと比較する。
    """

    def __init__(self):
        print("🧠 Loading PaddleOCR (Japanese)...")
        # PaddleOCR 3.x の新しいシンプルなAPI
        self.ocr = PaddleOCR(lang="japan")
        print("✅ PaddleOCR loaded successfully")

    def recognize(self, char_image: Image.Image) -> str:
        """
        文字画像を認識し、認識結果を返す

        Args:
            char_image: 文字画像

        Returns:
            認識された文字列
        """
        img_np = np.array(char_image.convert("RGB"))

        try:
            result = self.ocr.predict(img_np)

            if result and len(result) > 0:
                # 新しいAPIの結果形式に対応
                texts = []
                for item in result:
                    if isinstance(item, dict) and "rec_texts" in item:
                        texts.extend(item["rec_texts"])
                    elif isinstance(item, list):
                        for subitem in item:
                            if isinstance(subitem, dict) and "text" in subitem:
                                texts.append(subitem["text"])
                            elif isinstance(subitem, (list, tuple)) and len(subitem) >= 2:
                                texts.append(
                                    str(subitem[1][0])
                                    if isinstance(subitem[1], (list, tuple))
                                    else str(subitem[1])
                                )
                return "".join(texts)
        except Exception as e:
            print(f"  ⚠️ OCR error: {e}")

        return ""

    def verify(self, char_image: Image.Image, expected: str) -> tuple[bool, str]:
        """
        文字画像がアノテーションと一致するか検証

        Args:
            char_image: 文字画像
            expected: 期待される文字

        Returns:
            (一致したか, 認識結果)
        """
        result = self.recognize(char_image)
        is_match = result == expected
        return is_match, result

    def verify_batch(
        self, char_images: list[Image.Image], expected_chars: list[str]
    ) -> list[tuple[bool, str]]:
        """
        複数の文字画像をバッチ検証

        Args:
            char_images: 文字画像のリスト
            expected_chars: 期待される文字のリスト

        Returns:
            [(一致したか, 認識結果), ...] のリスト
        """
        results = []
        for img, expected in zip(char_images, expected_chars, strict=False):
            is_match, recognized = self.verify(img, expected)
            results.append((is_match, recognized))
        return results
