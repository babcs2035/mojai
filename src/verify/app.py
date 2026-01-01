"""
TUI検証アプリケーション

Textualフレームワークを使用したターミナルベースの
OCR結果検証インターフェース。

計画書の「5.2 アプリケーション設計とUX」に基づく実装:
- 3ペイン構成 (コンテキスト/フォーカス/エディタ)
- キーボードナビゲーション
- 確信度の可視化
- スタイル参照(アンカー)の選定

使用方法:
    mise run verify <json_path>
    uv run python -m src.verify.app <json_path>
"""

import json
from pathlib import Path

import click
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import Footer, Header, Input, Static

from src.config import settings
from src.verify.widgets import (
    CharacterCard,
    ConfidenceBar,
    ContextView,
    FocusView,
    StatusBar,
)


class VerificationApp(App):
    """
    OCR結果検証アプリケーション

    CUI上で画像とテキストを同時に扱い、
    キーボードショートカットによる高速検証を実現する。
    """

    TITLE = "mojai - OCR検証インターフェース"

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 4;
        grid-columns: 1fr 2fr;
        grid-rows: auto auto 1fr auto;
    }

    #context-pane {
        column-span: 2;
        height: 12;
    }

    #confidence-bar {
        column-span: 2;
        height: 1;
    }

    #focus-pane {
        height: 100%;
        min-height: 15;
    }

    #editor-pane {
        height: 100%;
    }

    #status-bar {
        column-span: 2;
        height: 3;
        dock: bottom;
    }

    .line-container {
        height: auto;
        border: solid $primary-darken-2;
        margin: 1;
        padding: 1;
    }

    .line-container.current-line {
        border: solid $accent;
        background: $accent 5%;
    }

    .line-header {
        text-style: bold;
        margin-bottom: 1;
    }

    .char-grid {
        layout: horizontal;
        height: auto;
        overflow-x: auto;
    }

    CharacterCard {
        width: 12;
        height: 6;
        margin: 0 1;
    }

    CharacterCard.selected {
        border: double $accent;
        background: $accent 20%;
    }
    """

    BINDINGS = [
        Binding("tab", "next_char", "次の文字", show=True),
        Binding("shift+tab", "prev_char", "前の文字", show=True),
        Binding("down", "next_line", "次の行", show=False),
        Binding("up", "prev_line", "前の行", show=False),
        Binding("f5", "toggle_anchor", "アンカー設定", show=True),
        Binding("ctrl+s", "save", "保存", show=True),
        Binding("q", "quit", "終了", show=True),
        Binding("escape", "clear_focus", "フォーカス解除", show=False),
    ]

    def __init__(self, json_path: Path | str, **kwargs):
        super().__init__(**kwargs)
        self.json_path = Path(json_path)
        self.output_dir = self.json_path.parent
        self.data: dict = {}
        self.current_line_idx = 0
        self.current_char_idx = 0
        self.modified = False

    def compose(self) -> ComposeResult:
        yield Header()

        # コンテキストペイン (上段)
        yield ContextView(id="context-pane")

        # 確信度バー
        yield ConfidenceBar(id="confidence-bar")

        # フォーカスペイン (中段・左)
        yield FocusView(output_dir=self.output_dir, id="focus-pane")

        # エディタペイン (中段・右)
        yield ScrollableContainer(id="editor-pane")

        # ステータスバー (下段)
        yield StatusBar(id="status-bar")

        yield Footer()

    def on_mount(self) -> None:
        """アプリケーション起動時の初期化"""
        self._load_data()
        self._build_editor()
        self._update_confidence_bar()
        self._update_status()
        self._update_focus()
        self._highlight_current()

    def _load_data(self) -> None:
        """JSONデータを読み込み"""
        try:
            with open(self.json_path, encoding="utf-8") as f:
                self.data = json.load(f)

            # コンテキストビューを更新
            context = self.query_one("#context-pane", ContextView)
            context.current_path = self.data.get("source_path", "")

            self.notify(f"読み込み完了: {self.json_path.name}", severity="information")

        except Exception as e:
            self.notify(f"読み込みエラー: {e}", severity="error")

    def _build_editor(self) -> None:
        """エディタペインを構築"""
        editor = self.query_one("#editor-pane", ScrollableContainer)

        for line in self.data.get("lines", []):
            line_idx = line["index"]

            # 行コンテナ
            line_container = Container(classes="line-container", id=f"line-{line_idx}")

            # 行ヘッダー
            confidence = line.get("confidence", 0.0)
            confidence_style = "green" if confidence > 0.9 else "yellow" if confidence > 0.7 else "red"
            line_header = Static(
                f"行 {line_idx + 1}: 「{line['text']}」 [{confidence:.1%}]",
                classes="line-header",
            )

            # 文字グリッド
            char_grid = Horizontal(classes="char-grid")

            for char in line.get("characters", []):
                char_card = CharacterCard(
                    char_index=char["index"],
                    line_index=line_idx,
                    text=char.get("text", ""),
                    confidence=char.get("confidence", 0.0),
                    image_path=char.get("image_path", ""),
                    is_anchor=char.get("is_style_anchor", False),
                    id=f"char-{line_idx}-{char['index']}",
                )
                char_grid.mount(char_card)

            line_container.mount(line_header)
            line_container.mount(char_grid)
            editor.mount(line_container)

    def _update_confidence_bar(self) -> None:
        """確信度バーを更新"""
        try:
            bar = self.query_one("#confidence-bar", ConfidenceBar)
            high, medium, low = 0, 0, 0

            for line in self.data.get("lines", []):
                for char in line.get("characters", []):
                    conf = char.get("confidence", 0.0)
                    if conf > 0.9:
                        high += 1
                    elif conf > 0.7:
                        medium += 1
                    else:
                        low += 1

            bar.high_count = high
            bar.medium_count = medium
            bar.low_count = low
        except Exception:
            pass

    def _update_status(self) -> None:
        """ステータスバーを更新"""
        status = self.query_one("#status-bar", StatusBar)

        total_lines = len(self.data.get("lines", []))
        current_line = self.data.get("lines", [])[self.current_line_idx] if total_lines > 0 else {}
        total_chars = len(current_line.get("characters", []))

        status.position = f"行 {self.current_line_idx + 1}/{total_lines}, 文字 {self.current_char_idx + 1}/{total_chars}"

        # アンカー数をカウント
        anchor_count = sum(
            1
            for line in self.data.get("lines", [])
            for char in line.get("characters", [])
            if char.get("is_style_anchor", False)
        )
        status.anchor_count = anchor_count

        if self.modified:
            status.message = "⚠️ 未保存の変更があります"
        else:
            status.message = ""

    def _update_focus(self) -> None:
        """フォーカスビューを更新"""
        focus = self.query_one("#focus-pane", FocusView)

        try:
            line = self.data["lines"][self.current_line_idx]
            char = line["characters"][self.current_char_idx]
            focus.current_char_image = char.get("image_path", "")
            focus.current_text = char.get("text", "?")
        except (IndexError, KeyError):
            focus.current_char_image = ""
            focus.current_text = ""

    def _highlight_current(self) -> None:
        """現在の行と文字をハイライト"""
        # 全ての行からハイライトを削除
        for container in self.query(".line-container"):
            container.remove_class("current-line")

        # 全ての文字カードから選択状態を削除
        for card in self.query(CharacterCard):
            card.is_selected = False

        # 現在の行をハイライト
        try:
            current_line = self.query_one(f"#line-{self.current_line_idx}")
            current_line.add_class("current-line")

            # 現在の文字カードを選択
            card_id = f"char-{self.current_line_idx}-{self.current_char_idx}"
            current_card = self.query_one(f"#{card_id}", CharacterCard)
            current_card.is_selected = True

            # スクロールして表示
            current_card.scroll_visible()
        except Exception:
            pass

    def action_next_char(self) -> None:
        """次の文字に移動"""
        lines = self.data.get("lines", [])
        if not lines:
            return

        current_line = lines[self.current_line_idx]
        chars = current_line.get("characters", [])

        if self.current_char_idx < len(chars) - 1:
            self.current_char_idx += 1
        elif self.current_line_idx < len(lines) - 1:
            self.current_line_idx += 1
            self.current_char_idx = 0

        self._update_status()
        self._update_focus()
        self._highlight_current()

    def action_prev_char(self) -> None:
        """前の文字に移動"""
        lines = self.data.get("lines", [])
        if not lines:
            return

        if self.current_char_idx > 0:
            self.current_char_idx -= 1
        elif self.current_line_idx > 0:
            self.current_line_idx -= 1
            prev_line = lines[self.current_line_idx]
            self.current_char_idx = max(0, len(prev_line.get("characters", [])) - 1)

        self._update_status()
        self._update_focus()
        self._highlight_current()

    def action_next_line(self) -> None:
        """次の行に移動"""
        lines = self.data.get("lines", [])
        if self.current_line_idx < len(lines) - 1:
            self.current_line_idx += 1
            next_line = lines[self.current_line_idx]
            chars = next_line.get("characters", [])
            self.current_char_idx = min(self.current_char_idx, len(chars) - 1)
            self.current_char_idx = max(0, self.current_char_idx)

            self._update_status()
            self._update_focus()
            self._highlight_current()

    def action_prev_line(self) -> None:
        """前の行に移動"""
        if self.current_line_idx > 0:
            self.current_line_idx -= 1
            lines = self.data.get("lines", [])
            prev_line = lines[self.current_line_idx]
            chars = prev_line.get("characters", [])
            self.current_char_idx = min(self.current_char_idx, len(chars) - 1)
            self.current_char_idx = max(0, self.current_char_idx)

            self._update_status()
            self._update_focus()
            self._highlight_current()

    def action_toggle_anchor(self) -> None:
        """現在の文字のアンカー状態を切り替え"""
        try:
            char = self.data["lines"][self.current_line_idx]["characters"][self.current_char_idx]
            char["is_style_anchor"] = not char.get("is_style_anchor", False)
            self.modified = True

            # UIを更新
            card_id = f"char-{self.current_line_idx}-{self.current_char_idx}"
            card = self.query_one(f"#{card_id}", CharacterCard)
            card.is_anchor = char["is_style_anchor"]

            status = "設定 ⭐" if char["is_style_anchor"] else "解除"
            char_text = char.get("text", "?")
            self.notify(f"アンカー{status}: 「{char_text}」", severity="information")

            self._update_status()
        except (IndexError, KeyError):
            self.notify("アンカー設定エラー", severity="error")

    def action_save(self) -> None:
        """変更を保存"""
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            self.modified = False
            self.notify("保存しました ✅", severity="information")
            self._update_status()
        except Exception as e:
            self.notify(f"保存エラー: {e}", severity="error")

    def action_clear_focus(self) -> None:
        """フォーカスを解除"""
        self.set_focus(None)

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """入力変更時の処理"""
        # 親のCharacterCardを取得
        card = event.input.parent
        if isinstance(card, CharacterCard):
            try:
                char = self.data["lines"][card.line_index]["characters"][card.char_index]
                char["text"] = event.value
                self.modified = True
                self._update_status()
            except (IndexError, KeyError):
                pass


@click.command()
@click.argument("json_path", type=click.Path(exists=True))
def main(json_path: str) -> None:
    """
    TUI検証インターフェースを起動

    JSON_PATH: OCR結果のJSONファイル
    """
    app = VerificationApp(json_path)
    app.run()


if __name__ == "__main__":
    main()
