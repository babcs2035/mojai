"""
OCR 統合パイプライン

前処理、文字分割、Deep Learning検証、レポート生成を統合したメインパイプライン。
"""

import json
import webbrowser
from pathlib import Path

from PIL import Image

from src.config import settings
from src.ocr.adjuster import IterativeRefiner
from src.ocr.preprocessor import Preprocessor
from src.ocr.report import ReportGenerator
from src.ocr.segmenter import CharacterSegmenter
from src.ocr.verifier import CharacterVerifier


class OCRPipeline:
    """
    OCR統合パイプライン (Deep Learning検証付き)

    1. 谷点検出で初期分割
    2. manga-ocr で各文字を検証
    3. 不一致があれば境界を調整して再検証
    4. HTMLレポート生成
    """

    def __init__(self):
        self.output_dir = settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = Preprocessor()
        self.segmenter = CharacterSegmenter()
        self.verifier = CharacterVerifier()
        self.refiner = IterativeRefiner(max_iterations=3)
        self.report_generator = ReportGenerator(self.output_dir)

    def process(self) -> Path:
        """画像処理 → 文字分割 → 検証・調整 → HTMLレポート生成"""
        print("📝 OCR処理を開始...")

        # 入力ファイルの確認
        input_path = self._find_input_image()
        anno_path = settings.input_dir / "annotation.txt"

        if input_path is None:
            print(f"❌ Error: Input image not found in {settings.input_dir}")
            return None

        if not anno_path.exists():
            print(f"❌ Error: Annotation file not found: {anno_path}")
            return None

        # 画像とアノテーションの読み込み
        image = Image.open(input_path).convert("RGB")
        with open(anno_path, encoding="utf-8") as f:
            anno_lines = [line.strip() for line in f if line.strip()]

        target_text = "".join(anno_lines)
        print(f"  📊 Target: {len(target_text)} chars ({len(anno_lines)} lines)")

        # 1. 前処理
        _, binary = self.preprocessor.process(image)
        print("  ✅ Preprocessing complete")

        # 2. 行検出
        line_regions = self.preprocessor.detect_lines(binary)
        print(f"  📏 Detected {len(line_regions)} lines")

        if len(line_regions) != len(anno_lines):
            print(
                f"  ⚠️ Warning: Line count mismatch (detected={len(line_regions)}, annotation={len(anno_lines)})"
            )
            num_proc = min(len(line_regions), len(anno_lines))
        else:
            num_proc = len(line_regions)

        all_results = []
        char_idx = 0
        total_verified = 0
        total_matched = 0

        # 3. 行ごとの処理
        for line_no in range(num_proc):
            y1, y2 = line_regions[line_no]
            line_text = anno_lines[line_no]
            line_binary = binary[y1:y2, :]
            line_image = image.crop((0, y1, image.width, y2))

            print(f"  📝 Line {line_no + 1}: '{line_text[:15]}...' ({len(line_text)} chars)")

            # 初期分割
            initial_boundaries = self.segmenter.segment(line_binary, len(line_text))

            # 反復的検証・調整
            refined_boundaries = self.refiner.refine(
                initial_boundaries,
                line_binary,
                line_image,
                list(line_text),
                self.verifier,
            )

            # 最終検証
            char_images = []
            for x_start, x_end in refined_boundaries:
                if x_end > x_start:
                    char_images.append(line_image.crop((x_start, 0, x_end, line_image.height)))
                else:
                    char_images.append(Image.new("RGB", (10, 10), "white"))

            verification_results = self.verifier.verify_batch(char_images, list(line_text))
            line_matched = sum(1 for is_match, _ in verification_results if is_match)
            total_verified += len(line_text)
            total_matched += line_matched
            print(f"    ✅ Verified: {line_matched}/{len(line_text)} chars matched")

            # 文字画像の保存
            for i, (x_start, x_end) in enumerate(refined_boundaries):
                if i >= len(line_text):
                    break

                char = line_text[i]
                is_match, recognized = (
                    verification_results[i] if i < len(verification_results) else (False, "?")
                )

                # 絶対座標
                char_x, char_y = x_start, y1
                char_w, char_h = x_end - x_start, y2 - y1

                # マージンを追加
                margin = 3
                crop_x1 = max(0, char_x - margin)
                crop_y1 = max(0, char_y - margin)
                crop_x2 = min(image.width, char_x + char_w + margin)
                crop_y2 = min(image.height, char_y + char_h + margin)

                # 保存
                char_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                char_filename = f"char_{char_idx:03d}_{char}.png"
                char_img.save(self.output_dir / char_filename)

                all_results.append(
                    {
                        "index": char_idx,
                        "text": char,
                        "recognized": recognized,
                        "verified": is_match,
                        "bbox": [int(char_x), int(char_y), int(char_w), int(char_h)],
                        "image_path": char_filename,
                    }
                )
                char_idx += 1

        # 精度計算
        accuracy = total_matched / total_verified if total_verified > 0 else 0
        print(f"  🎯 Overall accuracy: {accuracy:.1%} ({total_matched}/{total_verified})")

        # 4. 結果JSONの保存
        final_output = {
            "source_path": str(input_path),
            "metadata": {
                "total_characters": len(all_results),
                "detector": "Valley Detection + manga-ocr Verification",
                "accuracy": accuracy,
                "verified": total_matched,
                "total": total_verified,
            },
            "characters": all_results,
        }

        result_json_path = self.output_dir / "result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"  💾 Result saved: {len(all_results)} characters")

        # 5. HTMLレポート生成
        report_path = self.report_generator.generate(final_output)
        print(f"  📄 Report generated: {report_path}")

        # 6. ブラウザで表示
        webbrowser.open(f"file://{report_path.resolve()}")
        print("✅ Complete! Opening report in browser...")

        return report_path

    def _find_input_image(self) -> Path | None:
        """入力画像を検索"""
        for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
            path = settings.input_dir / f"image.{ext}"
            if path.exists():
                return path
        return None
