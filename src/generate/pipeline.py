"""
フォント生成パイプライン

スタイル参照画像からフォントファイルを生成する
一連の処理を統合したパイプライン。

処理フロー:
1. スタイル参照画像の読み込み
2. 生成対象文字リストの作成
3. FontDiffuserによる文字画像生成
4. フォントファイルの構築
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from PIL import Image
from tqdm import tqdm

from src.generate.diffuser import FontDiffuserWrapper
from src.generate.font_builder import FontBuilder, FontMetadata


# 文字セット定義
# 常用漢字 (一部) + ひらがな + カタカナ + 基本記号
DEFAULT_CHARSETS = {
    "hiragana": (
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
        "ぁぃぅぇぉっゃゅょ"
    ),
    "katakana": (
        "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
        "ァィゥェォッャュョー"
    ),
    "basic_kanji": "一二三四五六七八九十百千万円年月日時分秒",
    "common_kanji": (
        # 教育漢字（小学1年生）
        "一右雨円王音下火花貝学気九休玉金空月犬見五口校左三山子四糸字耳七車手十出女小上森人水正生青夕石赤千川先早"
        "草足村大男竹中虫町天田土二日入年白八百文木本名目立力林六"
        # 教育漢字（小学2年生の一部）
        "引羽雲園遠何科夏家歌画回会海絵外角楽活間丸岩顔汽記帰弓牛魚京強教近兄形計元言原戸古午後語工公広交光考行高黄合"
        "谷国黒今才細作算止市矢姉思紙寺自時室社弱首秋週春書少場色食心新親図数西声星晴切雪船線前組走多太体台地池知茶昼"
    ),
    "numbers": "0123456789",
    "alphabet_upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "alphabet_lower": "abcdefghijklmnopqrstuvwxyz",
    "punctuation": "。、！？「」『』（）・…ー〜",
    "symbols": "＠＃＄％＆＊＋－＝／＼｜：；",
}


class GenerationPipeline:
    """
    フォント生成パイプライン

    Core C (generate) の統合エントリポイント。
    スタイル参照画像 → 文字画像生成 → フォントファイル出力
    """

    def __init__(
        self,
        diffuser: FontDiffuserWrapper | None = None,
        font_builder: FontBuilder | None = None,
    ):
        """
        パイプラインを初期化

        Args:
            diffuser: FontDiffuserラッパー
            font_builder: フォントビルダー
        """
        self.diffuser = diffuser or FontDiffuserWrapper()
        self.font_builder = font_builder or FontBuilder()

    def get_charset(self, charset_name: str | None = None) -> str:
        """
        文字セットを取得

        Args:
            charset_name: 文字セット名 ("hiragana", "katakana", "basic_kanji", "all", None)

        Returns:
            対象文字の文字列
        """
        if charset_name is None or charset_name == "all":
            return "".join(DEFAULT_CHARSETS.values())

        if charset_name in DEFAULT_CHARSETS:
            return DEFAULT_CHARSETS[charset_name]

        # ファイルからの読み込み
        charset_path = Path(charset_name)
        if charset_path.exists():
            return charset_path.read_text(encoding="utf-8").strip()

        return charset_name  # 直接文字列として解釈

    def generate_font(
        self,
        style_image: Image.Image | Path | str,
        output_path: Path | str,
        charset: str | None = None,
        font_name: str = "MojaiFont",
        save_intermediates: bool = False,
    ) -> Path:
        """
        スタイル参照画像からフォントを生成

        Args:
            style_image: スタイル参照画像 (1文字の手書き画像)
            output_path: 出力フォントファイルパス (.ttf/.otf)
            charset: 生成する文字セット
            font_name: フォント名
            save_intermediates: 中間ファイル(文字画像)を保存するか

        Returns:
            生成されたフォントファイルのパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 文字セットを取得
        target_chars = list(self.get_charset(charset))
        click.echo(f"📝 生成対象: {len(target_chars)} 文字")

        # スタイル特徴量を抽出
        click.echo("🎨 スタイルを抽出中...")
        style_features = self.diffuser.extract_style(style_image)

        # 文字画像を生成
        click.echo("✨ 文字を生成中...")
        generated_images = []

        # プログレスバー付きで生成
        batch_size = self.diffuser.config.batch_size
        for i in tqdm(range(0, len(target_chars), batch_size), desc="生成中"):
            batch_chars = target_chars[i : i + batch_size]
            batch_images = self.diffuser.generate(style_features, batch_chars)
            generated_images.extend(batch_images)

        # 中間ファイルを保存
        if save_intermediates:
            intermediate_dir = output_path.parent / f"{output_path.stem}_chars"
            intermediate_dir.mkdir(exist_ok=True)

            for char, img in zip(target_chars, generated_images, strict=True):
                img_path = intermediate_dir / f"U+{ord(char):04X}_{char}.png"
                img.save(img_path)

            click.echo(f"📁 中間ファイル保存先: {intermediate_dir}")

        # フォントビルダーを設定
        self.font_builder.metadata = FontMetadata(
            family_name=font_name,
            style_name="Regular",
        )

        # グリフを追加
        click.echo("📦 フォントを構築中...")
        for char, img in zip(target_chars, generated_images, strict=True):
            self.font_builder.add_glyph(char, img)

        # フォントを生成
        font_path = self.font_builder.build(output_path)
        click.echo(f"✅ フォント生成完了: {font_path}")

        return font_path

    def generate_from_anchors(
        self,
        json_path: Path | str,
        output_path: Path | str,
        charset: str | None = None,
        font_name: str = "MojaiFont",
    ) -> Path:
        """
        検証結果JSONからアンカー画像を取得してフォントを生成

        Args:
            json_path: OCR結果のJSONファイル (is_style_anchor=Trueの文字を使用)
            output_path: 出力フォントファイルパス
            charset: 生成する文字セット
            font_name: フォント名

        Returns:
            生成されたフォントファイルのパス
        """
        json_path = Path(json_path)

        # JSONを読み込み
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # アンカー画像を探索
        anchor_images: list[Path] = []
        base_dir = json_path.parent

        for line in data.get("lines", []):
            for char in line.get("characters", []):
                if char.get("is_style_anchor", False):
                    img_path = base_dir / char.get("image_path", "")
                    if img_path.exists():
                        anchor_images.append(img_path)

        if not anchor_images:
            raise ValueError("アンカー画像が見つかりません。検証フェーズでスタイル参照を選択してください。")

        click.echo(f"🔍 アンカー画像: {len(anchor_images)} 個")

        # 最初のアンカー画像を使用 (将来的には複数アンカーの統合も検討)
        return self.generate_font(
            style_image=anchor_images[0],
            output_path=output_path,
            charset=charset,
            font_name=font_name,
        )

    def release(self) -> None:
        """GPUメモリを解放"""
        self.diffuser.release()


@click.command()
@click.argument("style_ref", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--charset",
    "-c",
    default="hiragana",
    help="文字セット (hiragana, katakana, basic_kanji, all, またはファイルパス)",
)
@click.option(
    "--name",
    "-n",
    default="MojaiFont",
    help="フォント名",
)
@click.option(
    "--save-images",
    is_flag=True,
    help="中間ファイル(文字画像)を保存",
)
def main(
    style_ref: str,
    output: str,
    charset: str,
    name: str,
    save_images: bool,
) -> None:
    """
    フォントを生成

    STYLE_REF: スタイル参照画像 (1文字の手書き画像) またはOCR結果JSON
    OUTPUT: 出力フォントファイルパス (.ttf/.otf)
    """
    pipeline = GenerationPipeline()

    try:
        style_path = Path(style_ref)

        if style_path.suffix == ".json":
            # JSONからアンカー画像を取得
            pipeline.generate_from_anchors(
                json_path=style_path,
                output_path=output,
                charset=charset,
                font_name=name,
            )
        else:
            # 画像を直接使用
            pipeline.generate_font(
                style_image=style_path,
                output_path=output,
                charset=charset,
                font_name=name,
                save_intermediates=save_images,
            )

    finally:
        pipeline.release()


if __name__ == "__main__":
    main()
