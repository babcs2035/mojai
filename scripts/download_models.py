"""
モデルダウンロードスクリプト

PaddleOCRとFontDiffuserの学習済みモデルをダウンロードする。
"""

import subprocess
import urllib.request
from pathlib import Path

import click

from src.config import settings


def download_paddleocr_models() -> None:
    """PaddleOCRモデルをダウンロード"""
    click.echo("📥 PaddleOCR モデルをダウンロード中...")

    try:
        from paddleocr import PaddleOCR

        # モデルの初期化（自動ダウンロード）
        # 注: PaddleOCR v3.x では use_gpu, show_log パラメータは廃止
        # GPU/CPUは自動検出される
        _ = PaddleOCR(lang="japan")
        click.echo("✅ PaddleOCR モデルのダウンロード完了")

    except ImportError:
        click.echo("⚠️ PaddleOCRがインストールされていません。")
        click.echo("   'uv sync' を実行してください。")
    except Exception as e:
        click.echo(f"⚠️ PaddleOCR モデルのダウンロード中にエラー: {e}")


def download_fontdiffuser_models() -> None:
    """FontDiffuserモデルをダウンロード (HuggingFace Hub経由)"""
    click.echo("📥 FontDiffuser モデルをダウンロード中...")

    models_dir = settings.models_dir / "fontdiffuser"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        # FontDiffuserのHuggingFace Hubリポジトリ
        repo_id = "yeungchenwa/FontDiffuser"

        click.echo(f"   リポジトリ: {repo_id}")
        click.echo("   ダウンロード中...")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
            )
            click.echo(f"✅ FontDiffuser モデルのダウンロード完了: {models_dir}")
        except Exception as e:
            click.echo(f"⚠️ HuggingFace Hubからのダウンロード失敗: {e}")
            _create_model_info(models_dir, repo_id)

    except ImportError:
        click.echo("⚠️ huggingface_hubがインストールされていません。")
        _create_model_info(models_dir, "yeungchenwa/FontDiffuser")


def _create_model_info(models_dir: Path, repo_id: str) -> None:
    """モデル情報ファイルを作成（手動ダウンロード用）"""
    model_info_file = models_dir / "MODEL_INFO.txt"
    github_url = "https://github.com/yeungchenwa/FontDiffuser"

    model_info_file.write_text(f"""FontDiffuser Model Information
==============================

FontDiffuserのモデルは以下から取得できます:

1. HuggingFace Hub:
   huggingface-cli download {repo_id} --local-dir {models_dir}

2. GitHub:
   {github_url}

モデルファイルを {models_dir} に配置してください。

必要なファイル:
- unet/: UNetモデル
- content_encoder/: コンテンツエンコーダ
- style_encoder/: スタイルエンコーダ
- vae/: VAEモデル

詳細は上記リポジトリのREADMEを参照してください。
""")
    click.echo(f"📁 モデル情報ファイルを作成: {model_info_file}")
    click.echo("⚠️ モデルは手動でダウンロードしてください。")


def download_noto_fonts() -> None:
    """Notoフォントをダウンロード (コンテンツ画像生成用)"""
    click.echo("📥 Noto Sans CJK フォントをダウンロード中...")

    fonts_dir = settings.models_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    font_path = fonts_dir / "NotoSansCJKjp-Regular.otf"

    if font_path.exists():
        click.echo(f"✅ フォント既存: {font_path}")
        return

    # Noto Sans CJK JPのダウンロードURL（複数候補）
    urls = [
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
        "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
    ]

    for noto_url in urls:
        try:
            click.echo(f"   試行中: {noto_url[:60]}...")
            urllib.request.urlretrieve(noto_url, font_path)
            click.echo(f"✅ フォントダウンロード完了: {font_path}")
            return
        except Exception as e:
            click.echo(f"   ⚠️ 失敗: {e}")
            continue

    # 全て失敗
    click.echo("⚠️ フォントダウンロード失敗")
    click.echo("   手動でダウンロードしてください:")
    click.echo("   https://github.com/notofonts/noto-cjk/releases")
    click.echo(f"   保存先: {font_path}")


def download_potrace() -> None:
    """Potraceのインストール確認"""
    click.echo("🔍 Potrace のインストール確認中...")

    try:
        result = subprocess.run(
            ["potrace", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0] if result.stdout else "unknown"
            click.echo(f"✅ Potrace インストール済み: {version}")
        else:
            _show_potrace_install_instructions()
    except FileNotFoundError:
        _show_potrace_install_instructions()


def _show_potrace_install_instructions() -> None:
    """Potraceのインストール手順を表示"""
    click.echo("⚠️ Potrace がインストールされていません。")
    click.echo("   フォント生成には Potrace が必要です。")
    click.echo("")
    click.echo("   インストール方法:")
    click.echo("   - Ubuntu/Debian: sudo apt install potrace")
    click.echo("   - macOS: brew install potrace")
    click.echo("   - Windows: https://potrace.sourceforge.net/")


def download_sam_model() -> None:
    """SAM (vit_b) モデルを直接ダウンロード"""
    click.echo("📥 SAM (vit_b) モデルをダウンロード中...")

    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    sam_path = models_dir / "sam_vit_b.pth"

    if sam_path.exists():
        click.echo(f"✅ SAM モデル既存: {sam_path}")
        return

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    try:
        click.echo(f"   URL: {url}")
        urllib.request.urlretrieve(url, sam_path)
        click.echo(f"✅ SAM モデルダウンロード完了: {sam_path}")
    except Exception as e:
        click.echo(f"⚠️ SAM モデルダウンロード失敗: {e}")


@click.command()
@click.option("--paddleocr", is_flag=True, help="PaddleOCRモデルのみダウンロード")
@click.option("--fontdiffuser", is_flag=True, help="FontDiffuserモデルのみダウンロード")
@click.option("--sam", is_flag=True, help="SAMモデルのみダウンロード")
@click.option("--fonts", is_flag=True, help="フォントのみダウンロード")
@click.option("--check", is_flag=True, help="依存ツールのインストール確認のみ")
def main(paddleocr: bool, fontdiffuser: bool, sam: bool, fonts: bool, check: bool) -> None:
    """
    学習済みモデルをダウンロード
    """
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    if check:
        download_potrace()
        return

    download_all = not (paddleocr or fontdiffuser or sam or fonts)

    if download_all or sam:
        download_sam_model()

    if download_all or paddleocr:
        download_paddleocr_models()

    if download_all or fontdiffuser:
        download_fontdiffuser_models()

    if download_all or fonts:
        download_noto_fonts()

    if download_all:
        download_potrace()

    click.echo("")
    click.echo("🎉 モデルダウンロード処理が完了しました")
    click.echo(f"📁 モデル保存先: {settings.models_dir}")


if __name__ == "__main__":
    main()
