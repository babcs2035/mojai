"""
Textual用カスタムウィジェット

OCR検証インターフェースで使用するカスタムウィジェット群。
Sixel/Kitty画像表示プロトコルに対応した画像ウィジェットを含む。
"""

from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Static

from src.verify.image_display import ImageProtocol, TerminalImageDisplay


class ImageRenderable:
    """
    Richでレンダリング可能な画像オブジェクト

    TerminalImageDisplayを使用してターミナルプロトコルに応じた
    画像出力を生成する。
    """

    def __init__(
        self,
        image_path: str | Path | None = None,
        max_width: int = 40,
        max_height: int = 20,
    ):
        self.image_path = Path(image_path) if image_path else None
        self.max_width = max_width
        self.max_height = max_height
        self._display = TerminalImageDisplay()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.image_path and self.image_path.exists():
            try:
                # ブロック文字でレンダリング（最も互換性が高い）
                rendered = self._display.render(
                    self.image_path,
                    max_width=min(self.max_width, options.max_width),
                    max_height=self.max_height,
                )
                yield Text.from_ansi(rendered)
            except Exception as e:
                yield Text(f"[画像読み込みエラー: {e}]", style="red")
        else:
            yield Text("[画像なし]", style="dim")

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        return Measurement(self.max_width, self.max_width)


class CharacterCard(Widget):
    """
    文字カードウィジェット

    1文字分の情報（画像パス、認識テキスト、確信度）を表示し、
    編集可能なインターフェースを提供する。
    """

    DEFAULT_CSS = """
    CharacterCard {
        height: auto;
        width: 100%;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    CharacterCard.low-confidence {
        border: solid $error;
    }

    CharacterCard.high-confidence {
        border: solid $success;
    }

    CharacterCard.anchor {
        border: double $warning;
        background: $warning 20%;
    }

    CharacterCard.selected {
        border: solid $accent;
        background: $accent 10%;
    }

    CharacterCard .char-text {
        text-style: bold;
        width: 100%;
    }

    CharacterCard .char-confidence {
        color: $text-muted;
    }
    """

    # リアクティブプロパティ
    text = reactive("")
    confidence = reactive(0.0)
    is_anchor = reactive(False)
    is_selected = reactive(False)
    image_path = reactive("")

    def __init__(
        self,
        char_index: int,
        line_index: int,
        text: str = "",
        confidence: float = 0.0,
        image_path: str = "",
        is_anchor: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.char_index = char_index
        self.line_index = line_index
        self.text = text
        self.confidence = confidence
        self.image_path = image_path
        self.is_anchor = is_anchor

    def compose(self) -> ComposeResult:
        yield Static(f"[{self.line_index}:{self.char_index}]", classes="char-index")
        yield Input(value=self.text, placeholder="?", classes="char-input")
        yield Static(f"{self.confidence:.1%}", classes="char-confidence")

    def watch_confidence(self, confidence: float) -> None:
        """確信度に応じてスタイルを変更"""
        self.remove_class("low-confidence", "high-confidence")
        if confidence < 0.7:
            self.add_class("low-confidence")
        elif confidence > 0.9:
            self.add_class("high-confidence")

    def watch_is_anchor(self, is_anchor: bool) -> None:
        """アンカー状態に応じてスタイルを変更"""
        if is_anchor:
            self.add_class("anchor")
        else:
            self.remove_class("anchor")

    def watch_is_selected(self, is_selected: bool) -> None:
        """選択状態に応じてスタイルを変更"""
        if is_selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")


class ContextView(Static):
    """
    コンテキストビューウィジェット

    文書全体または現在行周辺の画像を表示する。
    前後の文脈を確認するためのビュー。
    """

    DEFAULT_CSS = """
    ContextView {
        height: 12;
        border: solid $primary;
        padding: 1;
    }
    """

    current_path = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._display = TerminalImageDisplay()

    def render(self):
        if self.current_path:
            path = Path(self.current_path)
            if path.exists():
                try:
                    # 画像をレンダリング
                    image_renderable = ImageRenderable(
                        path,
                        max_width=80,
                        max_height=8,
                    )
                    return Panel(
                        image_renderable,
                        title=f"コンテキスト: {path.name}",
                    )
                except Exception:
                    pass

            return Panel(
                Text(f"📄 {path.name}"),
                title="コンテキスト",
            )
        return Panel(Text("画像未読み込み", style="dim"), title="コンテキスト")


class FocusView(Static):
    """
    フォーカスビューウィジェット

    現在選択中の1文字の高解像度クロップ画像を表示する。
    Sixel/Kittyプロトコルを使用。
    """

    DEFAULT_CSS = """
    FocusView {
        height: 18;
        border: solid $primary;
        padding: 1;
    }
    """

    current_char_image = reactive("")
    current_text = reactive("")

    def __init__(self, output_dir: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir or Path(".")
        self._display = TerminalImageDisplay()

    def render(self):
        if self.current_char_image:
            # 相対パスを絶対パスに変換
            image_path = Path(self.current_char_image)
            if not image_path.is_absolute():
                image_path = self.output_dir / self.current_char_image

            if image_path.exists():
                try:
                    image_renderable = ImageRenderable(
                        image_path,
                        max_width=30,
                        max_height=12,
                    )
                    return Panel(
                        image_renderable,
                        title=f"フォーカス: {self.current_text or '?'}",
                        subtitle=image_path.name,
                    )
                except Exception as e:
                    return Panel(
                        Text(f"画像エラー: {e}", style="red"),
                        title="フォーカス",
                    )

            return Panel(
                Text(f"🔍 {image_path.name}", style="yellow"),
                title="フォーカス(ファイルなし)",
            )
        return Panel(Text("文字未選択", style="dim"), title="フォーカス")


class StatusBar(Static):
    """
    ステータスバーウィジェット

    現在の位置、操作ヒント、統計情報を表示する。
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    """

    position = reactive("0/0")
    message = reactive("")
    anchor_count = reactive(0)

    def render(self):
        hints = "[Tab] 次へ  [Shift+Tab] 前へ  [F5] アンカー設定  [Ctrl+S] 保存  [Q] 終了"

        if self.message:
            status_text = self.message
        else:
            status_text = hints

        anchor_info = f"⭐ アンカー: {self.anchor_count}" if self.anchor_count > 0 else ""

        return Text.assemble(
            ("📍 ", "bold"),
            (self.position, "cyan"),
            ("  │  ", "dim"),
            (anchor_info, "yellow") if anchor_info else ("", ""),
            ("  " if anchor_info else "", ""),
            (status_text, ""),
        )


class ConfidenceBar(Static):
    """
    確信度バーウィジェット

    全体の確信度分布を視覚的に表示する。
    """

    DEFAULT_CSS = """
    ConfidenceBar {
        height: 1;
        width: 100%;
        padding: 0 1;
    }
    """

    high_count = reactive(0)
    medium_count = reactive(0)
    low_count = reactive(0)

    def render(self):
        total = self.high_count + self.medium_count + self.low_count
        if total == 0:
            return Text("確信度: データなし", style="dim")

        high_pct = self.high_count / total
        medium_pct = self.medium_count / total
        low_pct = self.low_count / total

        bar_width = 30
        high_bar = "█" * int(bar_width * high_pct)
        medium_bar = "▓" * int(bar_width * medium_pct)
        low_bar = "░" * int(bar_width * low_pct)

        return Text.assemble(
            ("確信度: ", ""),
            (high_bar, "green"),
            (medium_bar, "yellow"),
            (low_bar, "red"),
            (f" 高:{self.high_count} 中:{self.medium_count} 低:{self.low_count}", "dim"),
        )
