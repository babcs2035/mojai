"""
フォント生成パイプラインモジュール．

スタイル参照画像（手書き文字）から，拡散モデル（FontDiffuser）を用いて
一連のグリフ画像を生成し，最終的なフォントファイル（TrueType）として構築する．
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from PIL import Image
from tqdm import tqdm

from src.generate.diffuser import FontDiffuserWrapper
from src.generate.font_builder import FontBuilder, FontMetadata

# 標準的な文字セット定義（日本語常用漢字，仮名，アルファベット，記号類）
DEFAULT_CHARSETS = {
    "hiragana": (
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
        "ぁぃぅぇぉっゃゅょ"
    ),
    "katakana": (
        "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        "ガギグゲゴザジズゼゾタチヅデドバビブベボパピプペポ"
        "ァィゥェォッャュョー"
    ),
    "basic_kanji": "一二三四五六七八九十百千万円年月日時分秒",
    "common_kanji": (
        # 小学校第一学年で学習する漢字
        "一右雨円王音下火花貝学気九休玉金空月犬見五口校左三山子四糸字耳七車手十出女小上森人水正生青夕石赤千川先早"
        "草足村大男竹中虫町天田土二日入年白八百文木本名目立力林六"
        # 小学校第二学年で学習する漢字の一部
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
    フォント生成を一括制御するパイプラインクラス．

    拡散モデルによるスタイル転送と，フォントファイル構造へのパッキングを統合する．
    """

    def __init__(
        self,
        diffuser: FontDiffuserWrapper | None = None,
        font_builder: FontBuilder | None = None,
    ):
        """
        生成パイプラインを初期化する．

        Args:
            diffuser (FontDiffuserWrapper, optional): グリフ生成用エンジンのインスタンス．
            font_builder (FontBuilder, optional): フォント構築エンジンのインスタンス．
        """
        # 未指定時は各コンポーネントのデフォルト構成で初期化する
        self.diffuser = diffuser or FontDiffuserWrapper()
        self.font_builder = font_builder or FontBuilder()

    def get_charset(self, charset_name: str | None = None) -> str:
        """
        指定された名前あるいはファイルに基づいて，対象となる文字セットを取得する．

        Args:
            charset_name (str, optional): 文字セット名，パス，あるいは直接の文字列．

        Returns:
            str: 生成対象となる全文字が含まれる文字列．
        """
        # 全ての定義済み文字セットを統合
        if charset_name is None or charset_name == "all":
            return "".join(DEFAULT_CHARSETS.values())

        # 定義済みのセット名から検索
        if charset_name in DEFAULT_CHARSETS:
            return DEFAULT_CHARSETS[charset_name]

        # 外部ファイルからの読み込み試行
        charset_path = Path(charset_name)
        if charset_path.exists():
            return charset_path.read_text(encoding="utf-8").strip()

        # いずれにも当てはまらない場合は，引数自体を文字リストとして解釈
        return charset_name

    def generate_font(
        self,
        style_image: Image.Image | Path | str,
        output_path: Path | str,
        charset: str | None = None,
        font_name: str = "MojaiFont",
        save_intermediates: bool = False,
    ) -> Path:
        """
        手書きスタイル参照を元に，指定されたフォントファイルを生成する．

        Args:
            style_image: 参照となるスタイル画像（一文字分）．
            output_path: 出力先ファイルパス（.ttf または .otf）．
            charset: 生成対象とする文字セット．
            font_name: フォントの内部名称．
            save_intermediates: 生成された中間画像（PNG）を個別に保存するかどうか．

        Returns:
            Path: 生成されたフォントファイルのパス．
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 文字リストの確定
        target_chars = list(self.get_charset(charset))
        print(f"📝 Target: {len(target_chars)} characters to generate")

        # スタイル特徴量の抽出
        print("🎨 Extracting handwriting style from reference...")
        style_features = self.diffuser.extract_style(style_image)

        # 全文字のグリフ画像を連続生成
        print("✨ Generating glyph images using diffusion model...")
        generated_images = []

        # バッチ処理による効率的な生成
        batch_size = self.diffuser.config.batch_size
        for i in tqdm(range(0, len(target_chars), batch_size), desc="  Processing batches"):
            batch_chars = target_chars[i : i + batch_size]
            batch_images = self.diffuser.generate(style_features, batch_chars)
            generated_images.extend(batch_images)

        # 中間解析用としての画像保存
        if save_intermediates:
            intermediate_dir = output_path.parent / f"{output_path.stem}_intermediates"
            intermediate_dir.mkdir(exist_ok=True)

            for char, img in zip(target_chars, generated_images, strict=True):
                # ユニコードコードポイントに基づいたファイル名で保存
                img_path = intermediate_dir / f"U+{ord(char):04X}_{char}.png"
                img.save(img_path)

            print(f"📁 Intermediate glyph images saved to: {intermediate_dir}")

        # フォント構築エンジンの設定
        self.font_builder.metadata = FontMetadata(
            family_name=font_name,
            style_name="Regular",
        )

        # 各文字画像の TrueType グリフへの変換と登録
        print("📦 Assembling font file fragments...")
        for char, img in zip(target_chars, generated_images, strict=True):
            self.font_builder.add_glyph(char, img)

        # ファイル書き出し
        font_path = self.font_builder.build(output_path)
        print(f"✅ Success: Font file created at {font_path}")

        return font_path

    def generate_from_anchors(
        self,
        json_path: Path | str,
        output_path: Path | str,
        charset: str | None = None,
        font_name: str = "MojaiFont",
    ) -> Path:
        """
        OCR 解析後の JSON データから，スタイル参照として指定された文字を抽出してフォントを生成する．

        Args:
            json_path: OCR 解析結果を含む JSON ファイルへのパス．
            output_path: 生成フォントの保存先パス．
            charset: 生成対象文字セット．
            font_name: フォント名称．

        Returns:
            Path: 生成されたフォントファイルのパス．
        """
        json_path = Path(json_path)

        # 解析データの読み込み
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # 「スタイル参照（アンカー）」フラグが付与された文字画像を探索
        anchor_images: list[Path] = []
        base_dir = json_path.parent

        # 階層構造に従って走査し，存在する画像ファイルのみを対象とする
        for line in data.get("lines", []):
            for char in line.get("characters", []):
                if char.get("is_style_anchor", False):
                    img_name = char.get("image_path", "")
                    img_path = base_dir / img_name
                    if img_path.exists():
                        anchor_images.append(img_path)

        # 参照画像が一つも見つからない場合はエラー
        if not anchor_images:
            raise ValueError(
                "No style anchor images found. Please select reference characters in verify phase."
            )

        print(f"🔍 Found {len(anchor_images)} style reference(s)")

        # 最初の候補を参照元として採用する
        return self.generate_font(
            style_image=anchor_images[0],
            output_path=output_path,
            charset=charset,
            font_name=font_name,
        )

    def release(self) -> None:
        """GPU リソースを明示的に解放する．"""
        self.diffuser.release()


@click.command()
@click.argument("style_ref", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--charset",
    "-c",
    default="hiragana",
    help="Target characters (hiragana, katakana, all, or file path)",
)
@click.option(
    "--name",
    "-n",
    default="MojaiFont",
    help="Internal font name",
)
@click.option(
    "--save-images",
    is_flag=True,
    help="Save intermediate glyph PNG files",
)
def main(
    style_ref: str,
    output: str,
    charset: str,
    name: str,
    save_images: bool,
) -> None:
    """
    スタイル参照からフォントを生成する CLI ツール．

    STYLE_REF: スタイル参照画像（一文字）または OCR 解析結果の JSON．
    OUTPUT: 出力先フォントパス (.ttf)．
    """
    pipeline = GenerationPipeline()

    try:
        style_path = Path(style_ref)

        if style_path.suffix == ".json":
            # JSON 形式の場合はアンカー情報の抽出フローへ
            pipeline.generate_from_anchors(
                json_path=style_path,
                output_path=output,
                charset=charset,
                font_name=name,
            )
        else:
            # 画像ファイルの場合は直接生成フローへ
            pipeline.generate_font(
                style_image=style_path,
                output_path=output,
                charset=charset,
                font_name=name,
                save_intermediates=save_images,
            )

    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)
    finally:
        pipeline.release()


if __name__ == "__main__":
    main()
