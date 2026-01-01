"""
ターミナル画像表示モジュール

Sixel/Kittyプロトコルを使用したターミナル上での画像表示を提供。
SSH経由でのリモート検証にも対応。
"""

import io
import os
import sys
from enum import Enum, auto
from pathlib import Path

import numpy as np
from PIL import Image


class ImageProtocol(Enum):
    """画像表示プロトコル"""

    SIXEL = auto()  # Sixelプロトコル (iTerm2, WezTerm, mlterm等)
    KITTY = auto()  # Kitty Graphics Protocol
    ITERM2 = auto()  # iTerm2 Inline Images
    BLOCK = auto()  # Unicode Block文字 (フォールバック)


class TerminalImageDisplay:
    """
    ターミナル画像表示クラス

    計画書の「5.1 技術選定」に基づき、実行環境のターミナル能力を
    自動判別して最適なプロトコルで画像を表示する。
    """

    def __init__(self, protocol: ImageProtocol | str = "auto"):
        """
        画像表示を初期化

        Args:
            protocol: 使用するプロトコル ("auto", "sixel", "kitty", "iterm2", "block")
        """
        if isinstance(protocol, str):
            if protocol == "auto":
                self.protocol = self._detect_protocol()
            else:
                self.protocol = ImageProtocol[protocol.upper()]
        else:
            self.protocol = protocol

    def _detect_protocol(self) -> ImageProtocol:
        """
        ターミナルの能力を検出して最適なプロトコルを選択
        """
        term = os.environ.get("TERM", "")
        term_program = os.environ.get("TERM_PROGRAM", "")
        kitty_window_id = os.environ.get("KITTY_WINDOW_ID", "")

        # Kitty の検出
        if kitty_window_id or "kitty" in term.lower():
            return ImageProtocol.KITTY

        # iTerm2 の検出
        if term_program == "iTerm.app":
            return ImageProtocol.ITERM2

        # Sixelサポートの検出 (WezTerm, mlterm等)
        if any(x in term.lower() for x in ["wezterm", "mlterm", "foot", "contour"]):
            return ImageProtocol.SIXEL

        # xterm-256colorでもSixelサポートしている場合がある
        if "xterm" in term.lower() or "256color" in term.lower():
            # DA1クエリで確認することも可能だが、簡易的にSixelを試みる
            return ImageProtocol.SIXEL

        # フォールバック
        return ImageProtocol.BLOCK

    def render(
        self,
        image: Image.Image | np.ndarray | Path | str,
        max_width: int = 80,
        max_height: int = 40,
    ) -> str:
        """
        画像をターミナル表示用の文字列にレンダリング

        Args:
            image: 入力画像
            max_width: 最大幅 (文字数)
            max_height: 最大高さ (行数)

        Returns:
            ターミナル表示用のエスケープシーケンス文字列
        """
        # 画像を読み込み
        if isinstance(image, (Path, str)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        # RGBに変換
        if img.mode != "RGB":
            img = img.convert("RGB")

        # プロトコルに応じてレンダリング
        if self.protocol == ImageProtocol.SIXEL:
            return self._render_sixel(img, max_width, max_height)
        elif self.protocol == ImageProtocol.KITTY:
            return self._render_kitty(img, max_width, max_height)
        elif self.protocol == ImageProtocol.ITERM2:
            return self._render_iterm2(img, max_width, max_height)
        else:
            return self._render_block(img, max_width, max_height)

    def _render_sixel(self, img: Image.Image, max_width: int, max_height: int) -> str:
        """Sixelプロトコルでレンダリング"""
        # リサイズ (Sixelは6ピクセル = 1行)
        char_pixel_width = 8  # 1文字 ≈ 8ピクセル
        char_pixel_height = 12  # 1行 ≈ 12ピクセル (フォント依存)

        target_width = max_width * char_pixel_width
        target_height = max_height * char_pixel_height

        img = self._resize_to_fit(img, target_width, target_height)

        try:
            from term_image.image import SixelImage

            sixel_img = SixelImage(img)
            return str(sixel_img)
        except ImportError:
            # term-imageがない場合はブロック文字にフォールバック
            return self._render_block(img, max_width, max_height)

    def _render_kitty(self, img: Image.Image, max_width: int, max_height: int) -> str:
        """Kitty Graphics Protocolでレンダリング"""
        import base64

        char_pixel_width = 8
        char_pixel_height = 12

        target_width = max_width * char_pixel_width
        target_height = max_height * char_pixel_height

        img = self._resize_to_fit(img, target_width, target_height)

        # PNGにエンコード
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")

        # Kittyエスケープシーケンスを構築
        # _Ga=T: 一時画像, f=100: PNG, a=T: 直接転送
        chunks = []
        while data:
            chunk = data[:4096]
            data = data[4096:]
            m = 1 if data else 0  # more data flag
            if not chunks:
                chunks.append(f"\x1b_Gf=100,a=T,m={m};{chunk}\x1b\\")
            else:
                chunks.append(f"\x1b_Gm={m};{chunk}\x1b\\")

        return "".join(chunks)

    def _render_iterm2(self, img: Image.Image, max_width: int, max_height: int) -> str:
        """iTerm2 Inline Imagesでレンダリング"""
        import base64

        char_pixel_width = 8
        char_pixel_height = 12

        target_width = max_width * char_pixel_width
        target_height = max_height * char_pixel_height

        img = self._resize_to_fit(img, target_width, target_height)

        # PNGにエンコード
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")

        # iTerm2エスケープシーケンス
        return f"\x1b]1337;File=inline=1;preserveAspectRatio=1:{data}\a"

    def _render_block(self, img: Image.Image, max_width: int, max_height: int) -> str:
        """Unicode Block文字でレンダリング (フォールバック)"""
        # 各文字は2行分のピクセルを表現 (▀ = 上半分)
        img = self._resize_to_fit(img, max_width, max_height * 2)

        pixels = np.array(img)
        result = []

        for y in range(0, img.height, 2):
            row = []
            for x in range(img.width):
                # 上のピクセル
                r1, g1, b1 = pixels[y, x] if y < img.height else (0, 0, 0)

                # 下のピクセル
                if y + 1 < img.height:
                    r2, g2, b2 = pixels[y + 1, x]
                else:
                    r2, g2, b2 = 0, 0, 0

                # ANSIエスケープシーケンス
                # 前景色 (上半分): \x1b[38;2;R;G;Bm
                # 背景色 (下半分): \x1b[48;2;R;G;Bm
                row.append(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▀")

            result.append("".join(row) + "\x1b[0m")  # リセット

        return "\n".join(result)

    def _resize_to_fit(
        self,
        img: Image.Image,
        max_width: int,
        max_height: int,
    ) -> Image.Image:
        """アスペクト比を維持してリサイズ"""
        ratio = min(max_width / img.width, max_height / img.height)
        if ratio < 1:
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    def display(
        self,
        image: Image.Image | np.ndarray | Path | str,
        max_width: int = 80,
        max_height: int = 40,
    ) -> None:
        """
        画像をターミナルに表示

        Args:
            image: 入力画像
            max_width: 最大幅
            max_height: 最大高さ
        """
        output = self.render(image, max_width, max_height)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
