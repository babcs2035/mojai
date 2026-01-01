"""
共通設定モジュール

アプリケーション全体で使用する設定を管理する。
環境変数や設定ファイルから値を読み込む。
"""

from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定"""

    model_config = ConfigDict(
        env_prefix="MOJAI_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ===== パス設定 =====
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = data_dir / "models"
    input_dir: Path = data_dir / "input"
    output_dir: Path = data_dir / "output"

    # ===== OCR設定 (Core A: load) =====
    ocr_lang: str = "japan"  # PaddleOCR言語設定 (PaddleOCR v3.xでGPUは自動検出)
    ocr_det_model_dir: Path | None = None  # 検出モデルディレクトリ
    ocr_rec_model_dir: Path | None = None  # 認識モデルディレクトリ

    # ===== TUI設定 (Core B: verify) =====
    tui_image_protocol: str = "auto"  # auto, sixel, kitty, block

    # ===== 生成設定 (Core C: generate) =====
    diffuser_model_path: Path | None = None  # FontDiffuserモデルパス
    diffuser_use_fp16: bool = True  # FP16推論
    diffuser_batch_size: int = 32  # バッチサイズ
    diffuser_num_inference_steps: int = 25  # 推論ステップ数

    # ===== GPU設定 =====
    cuda_device: int = 0  # 使用するCUDAデバイス


# シングルトンインスタンス
settings = Settings()
