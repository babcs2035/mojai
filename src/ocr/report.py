"""
レポート生成モジュール

HTMLレポートの生成を担当する。
"""

from pathlib import Path


class ReportGenerator:
    """HTMLレポート生成"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate(self, data: dict) -> Path:
        """HTMLレポートを生成"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='ja'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "  <title>OCR Result Report</title>",
            "  <style>",
            "    * { box-sizing: border-box; margin: 0; padding: 0; }",
            "    body { font-family: 'Helvetica Neue', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }",
            "    h1 { text-align: center; margin-bottom: 20px; color: #00d4ff; }",
            "    .meta { background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }",
            "    .meta p { margin: 5px 0; }",
            "    .chars { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }",
            "    .char { background: #0f3460; border-radius: 8px; padding: 10px; text-align: center; min-width: 60px; }",
            "    .char img { max-height: 50px; border: 1px solid #333; background: #fff; display: block; margin: 0 auto; }",
            "    .char-label { margin-top: 5px; font-size: 18px; color: #00d4ff; }",
            "    .char-index { font-size: 10px; color: #666; }",
            "    .summary { text-align: center; margin-top: 20px; color: #888; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>📝 OCR Result Report</h1>",
            "  <div class='meta'>",
            f"    <p><strong>Source:</strong> {data['source_path']}</p>",
            f"    <p><strong>Characters:</strong> {data['metadata']['total_characters']}</p>",
            f"    <p><strong>Detector:</strong> {data['metadata'].get('detector', 'N/A')}</p>",
            "  </div>",
            "  <div class='chars'>",
        ]

        for char in data.get("characters", []):
            img_path = char.get("image_path", "")
            html_parts.append("    <div class='char'>")
            if img_path:
                html_parts.append(f"      <img src='{img_path}' alt='{char['text']}'>")
            html_parts.append(f"      <div class='char-label'>{char['text']}</div>")
            html_parts.append(f"      <div class='char-index'>#{char['index']}</div>")
            html_parts.append("    </div>")

        html_parts.extend(
            [
                "  </div>",
                "  <div class='summary'>Report generated successfully</div>",
                "</body>",
                "</html>",
            ]
        )

        report_path = self.output_dir / "report.html"
        report_path.write_text("\n".join(html_parts), encoding="utf-8")
        return report_path
