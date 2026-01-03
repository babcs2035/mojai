"""
OCR 統合パイプラインモジュール．

manga-ocr による行単位の認識結果を利用し，画像解析に基づいた文字分割を行なう．
認識結果とアノテーションを照合し，各文字の座標範囲と確信度を特定する．
"""

import json
import webbrowser
from pathlib import Path

import numpy as np
from manga_ocr import MangaOcr
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.config import settings
from src.ocr.preprocessor import Preprocessor
from src.ocr.report import ReportGenerator


class OCRPipeline:
    """
    OCR 統合パイプラインクラス．

    以下のステップで処理を実行する：
    1. 前処理による行領域の検出．
    2. manga-ocr による行単位の文字列認識と確信度の取得．
    3. 認識結果とアノテーションの照合．
    4. 垂直投影プロファイルを用いた幾何学的な文字境界の推定．
    5. 解析結果をまとめた HTML レポートの生成．
    """

    def __init__(self):
        """パイプラインの初期化を行なう．"""
        self.output_dir = settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = Preprocessor()
        self.report_generator = ReportGenerator(self.output_dir)

        # モデルのロード
        print("🧠 Loading manga-ocr model...")
        self.mocr = MangaOcr()
        print("✅ Model loaded successfully")

    def process(self) -> Path:
        """
        画像処理からレポート生成までの一連の OCR フローを実行する．

        Returns:
            Path: 生成された HTML レポートのパス．
        """
        print("📝 Starting OCR process...")

        # 入力データのパスを特定
        input_path = self._find_input_image()
        anno_path = settings.input_dir / "annotation.txt"

        if input_path is None:
            print(f"❌ Error: Input image not found in {settings.input_dir}")
            return None

        if not anno_path.exists():
            print(f"❌ Error: Annotation file not found: {anno_path}")
            return None

        # 画像およびアノテーションの読み込み
        image = Image.open(input_path).convert("RGB")
        with open(anno_path, encoding="utf-8") as f:
            anno_lines = [line.strip() for line in f if line.strip()]

        target_text = "".join(anno_lines)
        print(f"  📊 Target: {len(target_text)} characters across {len(anno_lines)} lines")

        # 1. 画像の前処理（二値化など）
        _, binary = self.preprocessor.process(image)
        print("  ✅ Image preprocessing complete")

        # 2. 行領域の検出
        line_regions = self.preprocessor.detect_lines(binary)
        print(f"  📏 Detected {len(line_regions)} lines in the image")

        # 行数の不一致に関する警告
        if len(line_regions) != len(anno_lines):
            print(
                f"  ⚠️ Warning: Line count mismatch! (Detected: {len(line_regions)}, Annotation: {len(anno_lines)})"
            )
            num_proc = min(len(line_regions), len(anno_lines))
        else:
            num_proc = len(line_regions)

        all_results = []
        char_idx = 0
        total_chars = 0
        line_matches = 0

        # 3. 各行に対する処理の実行
        for line_no in range(num_proc):
            y1, y2 = line_regions[line_no]
            line_text = anno_lines[line_no]
            line_image = image.crop((0, y1, image.width, y2))
            line_binary = binary[y1:y2, :]

            print(f"  📝 Line {line_no + 1}: '{line_text[:15]}...' ({len(line_text)} chars)")

            # manga-ocr による認識および確信度の算出
            recognized_text, char_confidences = self._recognize_with_confidence(line_image)
            print(f"    🔍 Recognized: '{recognized_text[:20]}...'")

            # 行単位の認識精度チェック
            if recognized_text == line_text:
                line_matches += 1
                print("    ✅ Perfect line match")
            else:
                print(f"    ⚠️ Recognized {len(recognized_text)} chars (Target: {len(line_text)})")

            # 投影解析による文字境界の推定
            char_boundaries = self._find_character_boundaries(line_binary, len(line_text))

            # 各文字の抽出と保存
            for i in range(len(line_text)):
                char = line_text[i]
                total_chars += 1

                # 境界情報の取得（不足時は等分割で補完）
                if i < len(char_boundaries):
                    x_start, x_end = char_boundaries[i]
                else:
                    char_width = line_image.width / len(line_text)
                    x_start = int(i * char_width)
                    x_end = int((i + 1) * char_width)

                # 座標計算と保存
                char_x, char_y = x_start, y1
                char_w, char_h = x_end - x_start, y2 - y1

                # 余裕を持たせたクロップ範囲の設定
                margin = 3
                crop_x1 = max(0, char_x - margin)
                crop_y1 = max(0, char_y - margin)
                crop_x2 = min(image.width, char_x + char_w + margin)
                crop_y2 = min(image.height, char_y + char_h + margin)

                # 文字画像の保存
                char_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                char_filename = f"char_{char_idx:03d}_{char}.png"
                char_img.save(self.output_dir / char_filename)

                # 認識結果の紐付け（文字列長が異なる場合はアライメントがズレる可能性あり）
                if i < len(recognized_text):
                    recognized_char = recognized_text[i]
                    conf = char_confidences[i] if i < len(char_confidences) else 0.0
                else:
                    recognized_char = "?"
                    conf = 0.0

                is_match = recognized_char == char

                # 解析結果の蓄積
                all_results.append(
                    {
                        "index": char_idx,
                        "text": char,
                        "recognized": recognized_char,
                        "verified": is_match,
                        "confidence": float(conf),
                        "bbox": [int(char_x), int(char_y), int(char_w), int(char_h)],
                        "image_path": char_filename,
                    }
                )
                char_idx += 1

        # 統計情報の算出
        matched_chars = sum(1 for r in all_results if r["verified"])
        accuracy = matched_chars / total_chars if total_chars > 0 else 0
        print(f"  🎯 Overall accuracy: {accuracy:.1%} ({matched_chars}/{total_chars})")
        print(f"  📊 Line match record: {line_matches}/{num_proc}")

        # 4. 解析結果の JSON 出力
        final_output = {
            "source_path": str(input_path),
            "metadata": {
                "total_characters": len(all_results),
                "detector": "manga-ocr (Line-based) + Valley Detection",
                "accuracy": accuracy,
                "verified": matched_chars,
                "total": total_chars,
                "line_matches": line_matches,
            },
            "characters": all_results,
        }

        result_json_path = self.output_dir / "result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print("  💾 Results saved to JSON file")

        # 5. HTML レポートの生成と表示
        report_path = self.report_generator.generate(final_output)
        print(f"  📄 Report generated at: {report_path}")

        webbrowser.open(f"file://{report_path.resolve()}")
        print("✅ OCR process complete! Opening report...")

        return report_path

    def _recognize_with_confidence(self, image: Image.Image) -> tuple[str, list[float]]:
        """
        画像を OCR モデルに投入し，各生成トークンの確信度を取得する．

        Args:
            image (Image.Image): 認識対象の行画像．

        Returns:
            tuple[str, list[float]]: 認識された文字列と，各文字に対応する確信度のリスト．
        """
        import torch
        from transformers import ViTImageProcessor

        # モデル内部で使用されるプロセッサを取得
        processor = ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base")

        pixel_values = processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.mocr.model.device)

        # 生成オプション（スコア出力を有効化）
        with torch.no_grad():
            outputs = self.mocr.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=300,
            )

        # 1. 認識文字列の取得
        sequences = outputs.sequences
        decoded_text = self.mocr.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        # Transformers が挿入する余分な空白を除去
        decoded_text = decoded_text.replace(" ", "")

        # 2. 各生成ステップにおける確率（確信度）の算出
        scores = outputs.scores  # 各ステップのロジット（tuple）
        token_confidences = []

        for i, score_tensor in enumerate(scores):
            # 実際に出力されたトークンの確率値を計算
            token_id = sequences[0][i + 1]  # sequences[0][0] は [CLS] トークン

            # 終了トークンに到達した場合は停止
            if token_id == self.mocr.tokenizer.sep_token_id:
                break

            probs = torch.softmax(score_tensor, dim=-1)
            prob = probs[0, token_id].item()
            token_confidences.append(prob)

        # 3. 文字と確信度のアライメント調整
        # 簡易的な実装として，デコード結果の文字長に合わせてリストをリサイズする
        char_confidences = []
        if len(decoded_text) == len(token_confidences):
            char_confidences = token_confidences
        else:
            avg_conf = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
            char_confidences = token_confidences[: len(decoded_text)]
            while len(char_confidences) < len(decoded_text):
                char_confidences.append(avg_conf)

        return decoded_text, char_confidences

    def _find_character_boundaries(
        self, line_binary: np.ndarray, num_chars: int
    ) -> list[tuple[int, int]]:
        """
        行画像に対して垂直投影解析を行ない，文字の境界を推定する（谷点検出）．

        Args:
            line_binary (np.ndarray): 二値化された行画像．
            num_chars (int): その行に含まれるべき文字数．

        Returns:
            list[tuple[int, int]]: 各文字の左右の境界座標（x_start, x_end）のリスト．
        """
        height, width = line_binary.shape

        # 垂直方向への投影（インク量の積算）
        projection = np.sum(line_binary, axis=0).astype(float)
        # 信号の平滑化
        smoothed = gaussian_filter1d(projection, sigma=3)

        # インクが存在する有効領域を特定
        ink_threshold = np.max(smoothed) * 0.02
        ink_mask = smoothed > ink_threshold

        ink_start = 0
        ink_end = width
        for i in range(width):
            if ink_mask[i]:
                ink_start = max(0, i - 3)
                break
        for i in range(width - 1, -1, -1):
            if ink_mask[i]:
                ink_end = min(width, i + 3)
                break

        # インクが全くない場合のフォールバック（等幅分割）
        if ink_end <= ink_start:
            char_width = width / num_chars
            return [(int(i * char_width), int((i + 1) * char_width)) for i in range(num_chars)]

        ink_region = smoothed[ink_start:ink_end]
        estimated_char_width = len(ink_region) / num_chars
        min_distance = max(5, int(estimated_char_width * 0.3))

        # インク量が極小となる箇所（谷点）を抽出
        valleys, _ = find_peaks(
            -ink_region, distance=min_distance, prominence=np.max(ink_region) * 0.05
        )
        valleys = valleys + ink_start

        # 必要な区切り位置の数
        needed = num_chars - 1

        if len(valleys) >= needed:
            # 谷の深さ（インク量の少なさ）に基づいて上位を採用
            depths = smoothed[valleys]
            sorted_indices = np.argsort(depths)[:needed]
            boundaries = sorted(valleys[sorted_indices])
        elif len(valleys) > 0:
            # 谷が不足している場合，広い領域の中央を追加して補完
            boundaries = list(valleys)
            all_points = [ink_start] + list(boundaries) + [ink_end]
            while len(boundaries) < needed:
                gaps = [(all_points[i + 1] - all_points[i], i) for i in range(len(all_points) - 1)]
                gaps.sort(reverse=True)
                widest_idx = gaps[0][1]
                mid = (all_points[widest_idx] + all_points[widest_idx + 1]) // 2
                boundaries.append(mid)
                boundaries.sort()
                all_points = [ink_start] + list(boundaries) + [ink_end]
        else:
            # 谷が全く検出されない場合は単に均等分割
            ink_width = ink_end - ink_start
            boundaries = [ink_start + int(ink_width * (i + 1) / num_chars) for i in range(needed)]

        # 境界座標のリストを構成
        all_bounds = [ink_start] + list(boundaries) + [ink_end]
        return [(all_bounds[i], all_bounds[i + 1]) for i in range(num_chars)]

    def _find_input_image(self) -> Path | None:
        """データディレクトリ内の画像ファイルを検索する．"""
        for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
            path = settings.input_dir / f"image.{ext}"
            if path.exists():
                return path
        return None
