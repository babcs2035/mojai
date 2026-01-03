"""
OCR パイプラインのエントリポイントモジュール．

このスクリプトを実行することで，手書きドキュメントの認識および文字分割の全工程を開始する．
"""

import sys

from src.ocr import OCRPipeline


def main():
    """OCR パイプラインの初期化と実行を行なうメイン関数．"""
    print("🚀 Starting mojai OCR pipeline...")

    try:
        pipeline = OCRPipeline()
        pipeline.process()
    except Exception as e:
        print(f"❌ Critical error during OCR process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
