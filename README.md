# 手書き文書デジタル化・復元プラットフォーム (mojai)

次世代の手書き文書デジタル化・復元プラットフォームです。CUIベースの高速検証フローと拡散モデルによるワンショットフォント生成を統合しています。

## 特徴

- **Core A: load (IARE)** - PaddleOCRを使用した高精度なOCR認識
- **Core B: verify (TBIV)** - ターミナルベースの高速検証インターフェース
- **Core C: generate (GSTR)** - 拡散モデルによるワンショットフォント生成

## 必要要件

- Python 3.11+
- [mise](https://mise.jdx.dev/) (タスクランナー)
- [uv](https://github.com/astral-sh/uv) (Python依存関係管理)
- NVIDIA GPU (RTX 3090推奨、VRAM 24GB)

## セットアップ

```bash
# miseを信頼
mise trust

# 環境構築
mise run setup

# 学習済みモデルのダウンロード
mise run download-models
```

## 使用方法

### OCR処理

```bash
# 画像ファイルを処理
mise run load data/input/sample.jpg

# ディレクトリ内の全画像を処理
mise run load data/input/
```

### 検証インターフェース

```bash
# OCR結果の確認と修正
mise run verify data/output/result.json
```

### フォント生成

```bash
# スタイル参照画像からフォントを生成
mise run generate data/input/style_ref.png output/myfont.ttf
```

## 開発

```bash
# テスト実行
mise run test

# コード品質チェック
mise run lint

# コードフォーマット
mise run format
```
