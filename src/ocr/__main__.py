"""
OCR パイプラインエントリポイント
"""

from src.ocr import OCRPipeline

if __name__ == "__main__":
    pipeline = OCRPipeline()
    pipeline.process()
