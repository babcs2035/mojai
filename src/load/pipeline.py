"""
OCR認識パイプライン

画像の読み込みからOCR認識、文字セグメンテーションまでの
一連の処理を統合したパイプライン。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import settings
from src.load.ocr import OCREngine
from src.load.segmentation import CharacterSegmenter


@dataclass
class DocumentResult:
    """文書全体のOCR結果"""

    source_path: str
    lines: list[dict]
    metadata: dict


class OCRPipeline:
    """
    OCR認識パイプライン

    Core A (load) の統合エントリポイント。
    画像の読み込み → OCR認識 → 文字セグメンテーション → JSON出力
    """

    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        segmenter: CharacterSegmenter | None = None,
    ):
        """
        パイプラインを初期化

        Args:
            ocr_engine: OCRエンジン (Noneの場合はデフォルト設定で作成)
            segmenter: 文字セグメンター (Noneの場合はデフォルト設定で作成)
        """
        self.ocr_engine = ocr_engine or OCREngine()
        self.segmenter = segmenter or CharacterSegmenter()

    def process_image(
        self,
        image_path: Path | str,
        output_dir: Path | str | None = None,
        save_char_images: bool = True,
    ) -> DocumentResult:
        """
        単一画像を処理

        Args:
            image_path: 入力画像パス
            output_dir: 出力ディレクトリ (Noneの場合はsettings.output_dir)
            save_char_images: 文字画像を保存するか

        Returns:
            文書のOCR結果
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else settings.output_dir

        # 出力ディレクトリを作成
        doc_output_dir = output_dir / image_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # 画像を読み込み
        image = Image.open(image_path)
        image_array = np.array(image)

        # OCR認識
        line_results = self.ocr_engine.recognize(image_array)

        # 結果を構築
        lines_data: list[dict] = []

        for line_idx, line in enumerate(line_results):
            # 文字セグメンテーション
            char_boxes = self.segmenter.segment_from_bbox(
                image_array,
                line.bbox,
                orientation="horizontal",  # TODO: 縦書き検出を追加
            )

            # 文字画像を保存
            chars_data: list[dict] = []
            for char_idx, char_box in enumerate(char_boxes):
                char_data = {
                    "index": char_idx,
                    "bbox": {
                        "x": char_box.x,
                        "y": char_box.y,
                        "width": char_box.width,
                        "height": char_box.height,
                    },
                    "text": "",  # 検証フェーズで入力
                    "confidence": 0.0,
                    "is_style_anchor": False,  # スタイル参照フラグ
                }

                if save_char_images:
                    char_image_path = doc_output_dir / f"line{line_idx:03d}_char{char_idx:03d}.png"
                    Image.fromarray(char_box.image).save(char_image_path)
                    char_data["image_path"] = str(char_image_path.relative_to(output_dir))

                chars_data.append(char_data)

            # 認識テキストを文字に割り当て (簡易的な分配)
            if len(line.text) == len(chars_data):
                for i, char in enumerate(line.text):
                    chars_data[i]["text"] = char
                    chars_data[i]["confidence"] = line.confidence
            elif len(chars_data) > 0:
                # 文字数が一致しない場合は最初の文字にテキスト全体を設定
                chars_data[0]["text"] = line.text
                chars_data[0]["confidence"] = line.confidence

            line_data = {
                "index": line_idx,
                "text": line.text,
                "confidence": line.confidence,
                "bbox": line.bbox,
                "characters": chars_data,
            }
            lines_data.append(line_data)

        # 結果を構築
        result = DocumentResult(
            source_path=str(image_path),
            lines=lines_data,
            metadata={
                "image_width": image.width,
                "image_height": image.height,
                "total_lines": len(lines_data),
                "total_characters": sum(len(line["characters"]) for line in lines_data),
            },
        )

        # JSONを保存
        json_path = doc_output_dir / "result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        return result

    def process_directory(
        self,
        input_dir: Path | str,
        output_dir: Path | str | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ) -> list[DocumentResult]:
        """
        ディレクトリ内の全画像を処理

        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            extensions: 対象ファイル拡張子

        Returns:
            各画像のOCR結果リスト
        """
        input_dir = Path(input_dir)

        # 対象ファイルを収集
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        results: list[DocumentResult] = []
        for image_path in tqdm(image_files, desc="OCR処理中"):
            try:
                result = self.process_image(image_path, output_dir)
                results.append(result)
            except Exception as e:
                click.echo(f"エラー: {image_path} の処理に失敗: {e}", err=True)

        return results

    def release(self) -> None:
        """GPUメモリを解放"""
        self.ocr_engine.release()


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="出力ディレクトリ",
)
def main(input_path: str, output: str | None) -> None:
    """
    OCR処理を実行

    INPUT_PATH: 入力画像ファイルまたはディレクトリ
    """
    input_path = Path(input_path)
    output_dir = Path(output) if output else settings.output_dir

    pipeline = OCRPipeline()

    try:
        if input_path.is_file():
            click.echo(f"📄 処理中: {input_path}")
            result = pipeline.process_image(input_path, output_dir)
            click.echo(f"✅ 完了: {result.metadata['total_lines']} 行, "
                       f"{result.metadata['total_characters']} 文字を検出")
        else:
            click.echo(f"📁 ディレクトリを処理中: {input_path}")
            results = pipeline.process_directory(input_path, output_dir)
            click.echo(f"✅ 完了: {len(results)} ファイルを処理")

        click.echo(f"📂 出力先: {output_dir}")

    finally:
        pipeline.release()


if __name__ == "__main__":
    main()
