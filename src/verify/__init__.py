"""
Core B: verify (TBIV - ターミナルベース検証インターフェース)

Textualを使用したCUIベースのOCR結果検証インターフェース。
Sixel/Kittyプロトコルによる画像表示に対応。
"""

from src.verify.app import VerificationApp

__all__ = ["VerificationApp"]
