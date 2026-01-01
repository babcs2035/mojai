"""
設定モジュールのテスト
"""

import os
from pathlib import Path

import pytest


def test_settings_import():
    """設定モジュールがインポートできることを確認"""
    from src.config import settings

    assert settings is not None


def test_settings_default_values():
    """デフォルト値が正しく設定されていることを確認"""
    from src.config import settings

    assert settings.ocr_lang == "japan"
    assert settings.diffuser_use_fp16 is True
    assert settings.diffuser_batch_size == 32


def test_settings_paths():
    """パス設定が正しく解決されることを確認"""
    from src.config import settings

    assert isinstance(settings.project_root, Path)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.models_dir, Path)


def test_settings_env_override(monkeypatch):
    """環境変数でのオーバーライドが機能することを確認"""
    monkeypatch.setenv("MOJAI_OCR_LANG", "en")
    monkeypatch.setenv("MOJAI_DIFFUSER_USE_FP16", "false")

    # 設定を再読み込み
    from src.config import Settings

    new_settings = Settings()

    assert new_settings.ocr_lang == "en"
    assert new_settings.diffuser_use_fp16 is False
