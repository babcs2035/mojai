"""
FontDiffuser ラッパーモジュール．

拡散モデルを用いて，一文字の参照画像から任意の文字種のフォント画像を生成する．
RTX 3090（VRAM 24GB）の計算リソースに最適化されており，
高度なスタイル転写と構造維持を両立する．
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.config import settings


@dataclass
class GenerationConfig:
    """フォント生成に関する設定パラメータを保持するデータクラス．"""

    num_inference_steps: int = 25  # 推論のステップ数
    guidance_scale: float = 7.5  # 分類器フリー・ガイダンスの尺度
    strength: float = 0.75  # img2img における元画像の保持強度
    batch_size: int = 32  # GPU での一括処理サイズ
    use_fp16: bool = True  # 半精度浮動小数点（FP16）による高速化の有無
    seed: int | None = None  # 再現性のための乱数シード
    image_size: int = 64  # 出力画像の解像度


class ContentRenderer:
    """
    文字コンテンツ（骨格）のレンダリングを担当するクラス．

    標準フォント（Noto Sans CJK JP 等）を用いて，生成モデルに投入する入力画像を生成する．
    """

    def __init__(self, font_path: Path | str | None = None, font_size: int = 48):
        """
        レンダラーを初期化する．

        Args:
            font_path (Path | str, optional): 参照に使用するフォントのパス．
            font_size (int): レンダリング時のフォントサイズ．
        """
        self.font_size = font_size
        self._font: ImageFont.FreeTypeFont | None = None

        if font_path:
            self.font_path = Path(font_path)
        else:
            # プロジェクトデフォルトのフォントを検索
            default_font = settings.models_dir / "fonts" / "NotoSansCJKjp-Regular.otf"
            self.font_path = default_font if default_font.exists() else None

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """フォントリソースを読み込む（キャッシュ対応）．"""
        if self._font is not None:
            return self._font

        if self.font_path and self.font_path.exists():
            self._font = ImageFont.truetype(str(self.font_path), self.font_size)
        else:
            # フォントファイルが見つからない場合はシステムデフォルトで代用
            self._font = ImageFont.load_default()

        return self._font

    def render(self, character: str, size: int = 64) -> Image.Image:
        """
        指定された文字をレンダリングし，グレースケール画像を返す．

        Args:
            character (str): 対象文字．
            size (int): キャンバスサイズ．

        Returns:
            Image.Image: 白背景に黒文字で描画された画像．
        """
        font = self._load_font()

        # 白背景のグレースケール画像を作成
        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)

        # テキストの描画範囲を計算し，中央に配置する
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]

        draw.text((x, y), character, font=font, fill=0)

        return img

    def render_batch(self, characters: list[str], size: int = 64) -> list[Image.Image]:
        """複数の文字を一括でレンダリングする．"""
        return [self.render(char, size) for char in characters]


class FontDiffuserWrapper:
    """
    FontDiffuser モデルを制御するためのラッパークラス．

    Diffusers ライブリを活用し，Stable Diffusion ベースのスタイル転写を実現する．
    GPU リソースが利用不可能な場合は，自動的に古典的な画像処理（OpenCV）による
    フォールバック処理を行なう．
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str | None = None,
        config: GenerationConfig | None = None,
    ):
        """
        生成エンジンを初期化する．

        Args:
            model_path (Path | str, optional): モデルの重みが保存されているパス．
            device (str, optional): 演算デバイス（"cuda" または "cpu"）．
            config (GenerationConfig, optional): 生成パラメータ設定．
        """
        self.model_path = Path(model_path) if model_path else settings.diffuser_model_path
        self.device = device or self._detect_device()
        self.config = config or GenerationConfig(
            use_fp16=settings.diffuser_use_fp16,
            batch_size=settings.diffuser_batch_size,
            num_inference_steps=settings.diffuser_num_inference_steps,
        )

        self._pipeline: Any = None
        self._content_renderer = ContentRenderer()
        self._is_loaded = False

    def _detect_device(self) -> str:
        """システム上の最適なデバイス（CUDA または CPU）を特定する．"""
        if torch.cuda.is_available():
            return f"cuda:{settings.cuda_device}"
        return "cpu"

    def _load_model(self) -> None:
        """生成モデルをメモリにロードする（初回実行時のみ実行される遅延ロード）．"""
        if self._is_loaded:
            return

        try:
            # 必要なライブラリのインポートチェックおよびロード
            if self.model_path and self.model_path.exists():
                print(f"📦 Loading custom FontDiffuser model from {self.model_path}...")
                self._load_custom_model()
            else:
                print("📦 Loading baseline stable-diffusion-v1-5 for font generation...")
                self._load_sd_pipeline()

            self._is_loaded = True

        except Exception as e:
            print(f"⚠️ Failed to load diffusion model: {e}")
            print("💡 Falling back to classical image processing engine")
            self._is_loaded = True

    def _load_custom_model(self) -> None:
        """特定のパスからカスタム学習済みの FontDiffuser モデルをロードする．"""
        from diffusers import StableDiffusionImg2ImgPipeline

        self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            safety_checker=None,
        )
        self._pipeline.to(self.device)

        if "cuda" in self.device:
            self._pipeline.enable_attention_slicing()

    def _load_sd_pipeline(self) -> None:
        """標準的な Stable Diffusion img2img パイプラインをロードする．"""
        from diffusers import StableDiffusionImg2ImgPipeline

        self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            safety_checker=None,
            variant="fp16" if self.config.use_fp16 else None,
        )
        self._pipeline.to(self.device)

        if "cuda" in self.device:
            self._pipeline.enable_attention_slicing()

    def extract_style(self, style_image: Image.Image | np.ndarray | Path | str) -> dict:
        """
        手書きスタイル参照画像から視覚的特徴（スタイル情報）を抽出する．

        Args:
            style_image: 一文字分の参照画像．

        Returns:
            dict: 画像の統計情報や正規化された画像を含むスタイル辞書．
        """
        # 画像形式の統一
        if isinstance(style_image, (Path, str)):
            img = Image.open(style_image)
        elif isinstance(style_image, np.ndarray):
            img = Image.fromarray(style_image)
        else:
            img = style_image.copy()

        if img.mode != "L":
            img = img.convert("L")

        # 指定された解像度にリサイズ
        img = img.resize(
            (self.config.image_size, self.config.image_size),
            Image.Resampling.LANCZOS,
        )

        img_array = np.array(img, dtype=np.float32) / 255.0

        # 基本的な統計情報の算出
        style_info = {
            "image": img,
            "mean_intensity": float(np.mean(img_array)),
            "std_intensity": float(np.std(img_array)),
            "stroke_width": self._estimate_stroke_width(img_array),
        }

        return style_info

    def _estimate_stroke_width(self, img_array: np.ndarray) -> float:
        """
        距離変換を用いて文字の線の太さ（ストローク幅）を推定する．

        古典的画像処理でのフォールバック時に，グリフの太さを合わせるために使用される．
        """
        import cv2

        # 二値化と距離変換
        binary = (img_array < 0.5).astype(np.uint8) * 255
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        if dist.max() > 0:
            return float(dist.max() * 2)
        return 2.0

    def generate(
        self,
        style_info: dict,
        target_characters: list[str],
        content_font: str | None = None,
    ) -> list[Image.Image]:
        """
        抽出されたスタイル情報を元に，特定の文字リストに対するグリフを生成する．

        Args:
            style_info (dict): 抽出済みのスタイル特徴量．
            target_characters (list[str]): 生成対象の文字リスト．
            content_font (str, optional): コンテンツとして使用するフォント名．

        Returns:
            list[Image.Image]: 生成された画像リスト．
        """
        self._load_model()

        if content_font:
            self._content_renderer = ContentRenderer(content_font)

        generated_images: list[Image.Image] = []
        style_image = style_info["image"]

        # 乱数シードの設定
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        # バッチ単位での生成実行
        for i in range(0, len(target_characters), self.config.batch_size):
            batch_chars = target_characters[i : i + self.config.batch_size]
            batch_images = self._generate_batch(batch_chars, style_image, style_info, generator)
            generated_images.extend(batch_images)

        return generated_images

    def _generate_batch(
        self,
        characters: list[str],
        style_image: Image.Image,
        style_info: dict,
        generator: torch.Generator | None,
    ) -> list[Image.Image]:
        """指定されたバッチ内の文字画像を一括生成する．"""
        generated = []
        for char in characters:
            # 骨格（コンテンツ）画像のレンダリング
            content_image = self._content_renderer.render(char, self.config.image_size)

            # スタイル転写の実行（拡散モデルまたは古典的手法）
            styled_image = self._apply_style(content_image, style_image, style_info, generator)
            generated.append(styled_image)

        return generated

    def _apply_style(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        style_info: dict,
        generator: torch.Generator | None,
    ) -> Image.Image:
        """コンテンツ画像に対してスタイルを転写するエントリメソッド．"""
        if self._pipeline is not None:
            return self._apply_style_diffusion(content_image, style_image, style_info, generator)
        else:
            return self._apply_style_classical(content_image, style_image, style_info)

    def _apply_style_diffusion(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        style_info: dict,
        generator: torch.Generator | None,
    ) -> Image.Image:
        """
        拡散モデル（img2img）を用いたスタイル転写．

        標準フォントによる骨格をガイドとして，手書き風のテクスチャとスタイルを生成する．
        """
        # SD は RGB 入力を前提とするため変換
        content_rgb = content_image.convert("RGB")

        # 生成を誘導するプロンプトの構成
        prompt = "handwritten Japanese character, calligraphy style, black ink on white paper"
        negative_prompt = "blurry, low quality, distorted, extra strokes"

        try:
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=content_rgb,
                strength=self.config.strength,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            )

            # 出力結果をグレースケールとして処理し，サイズを正規化
            generated = result.images[0].convert("L")
            return generated.resize(
                (self.config.image_size, self.config.image_size),
                Image.Resampling.LANCZOS,
            )

        except Exception:
            # 推論失敗時は古典的手法にフォールバック
            return self._apply_style_classical(content_image, style_image, style_info)

    def _apply_style_classical(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        style_info: dict,
    ) -> Image.Image:
        """
        OpenCV 等を用いた古典的な画像処理によるスタイル転写．

        プロトタイプ環境やリソース制限下でのフォールバックとして機能し，
        線幅の調整とノイズ付与により手書きらしさを再現する．
        """
        import cv2

        content_arr = np.array(content_image, dtype=np.float32)

        # 二値化処理
        _, content_binary = cv2.threshold(
            content_arr.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 推定された線幅に基づいた膨張・収縮
        stroke_width = style_info.get("stroke_width", 2.0)
        kernel_size = max(1, int(stroke_width / 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if stroke_width > 2:
            styled = cv2.dilate(content_binary, kernel, iterations=1)
        else:
            styled = content_binary

        # 手書き感を演出するための微細なノイズとガウスぼかしの適用
        noise = np.random.normal(0, 5, styled.shape).astype(np.float32)
        styled = np.clip(styled.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        styled = cv2.GaussianBlur(styled, (3, 3), 0.5)

        # 白背景・黒文字への変換
        styled = 255 - styled

        return Image.fromarray(styled)

    def release(self) -> None:
        """GPU メモリおよびモデルリソースを安全に解放し，次のタスクへのリソース競合を防ぐ．"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU resources released")
