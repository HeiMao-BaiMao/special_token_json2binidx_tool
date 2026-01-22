# jsonl to binidx 変換ツール（特殊トークン対応）

RWKV モデルのデータセット準備のため、JSONL ファイルをバイナリ (.bin) とインデックス (.idx) 形式に変換する簡易ツールです。

このリポジトリは [gpt-neox](https://github.com/EleutherAI/gpt-neox) から大幅に簡略化し、[RWKV-LM](https://github.com/BlinkDL/RWKV-LM) の事前学習用に特殊トークンをサポートした JSONL → binidx 変換機能のみを提供します。

**ワークフロー**: データセット → special_token_json2binidx_tool → RWKV-LM(-infctx) 事前学習 → ベースLLM → [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros) オンライン学習

[English README is here](README.md)

## 特徴

- JSONL を効率的な RWKV 学習用バイナリ形式に変換
- 特殊トークンのサポート（conversation、system、search、think、env、common、text）
- SSG プロトコル形式と標準 JSONL 形式を同一データセット内で混在可能
- データフォルダごとのフィールドマッピング設定
- 複数のトークナイザー対応（RWKV、HuggingFace、SentencePiece）
- **SentencePiece トークナイザー学習** - 独自コーパスからカスタムトークナイザーを学習可能

## インストール

```bash
pip install -r requirements.txt
```

**必要な環境:**
- Python 3.7以上
- ftfy==6.1.*
- lm-dataformat==0.0.20
- numpy==1.24.*
- tokenizers==0.13.*
- torch==2.0.*
- tqdm==4.65.*
- sentencepiece>=0.1.99

## クイックスタート

### 基本的な使い方（SSG プロトコル + 特殊トークン）

```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./your_folder1,./your_folder2 \
  --sp_token_config ./tools/sp_token_config.json \
  --output-prefix ./data/sample \
  --vocab ./rwkv_vocab_v20230424.txt \
  --dataset-impl mmap \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

### 標準 JSONL 形式（特殊トークンなし）

```bash
python tools/preprocess_data.py \
  --input ./your_data.jsonl \
  --output-prefix ./data/output \
  --vocab ./rwkv_vocab_v20230424.txt \
  --dataset-impl mmap \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

## データフォルダ構造

```
your_folder/
├── a.jsonl
├── b.jsonl
└── your_folder.json  (オプション：マッピング設定)
```

オプションの `your_folder.json` で、そのフォルダ専用のフィールドマッピングをカスタマイズできます。

## JSONL フォーマット

### SSG プロトコル形式（特殊トークン付き）

1行に1ドキュメント：

```json
{"sample": [{"text":"導入テキスト"},{"conversation":"質問: RWKVとは何ですか？"},{"conversation":"回答: RWKVは言語モデルです。"},{"system":"これはシステム情報"}]}
```

### 標準形式（プレーンテキスト）

```json
{"text": "特殊トークンなしのシンプルなテキストドキュメントです。"}
```

### 混在形式（新機能！）

同じ JSONL ファイル内で両方の形式を混在できるようになりました：

```json
{"sample": [{"text":"これはSSGプロトコル形式"}]}
{"text": "これは標準形式"}
{"sample": [{"conversation":"Q: 質問?"},{"conversation":"A: 回答"}]}
{"text": "また別の標準形式行"}
```

## 特殊トークンの種類

`tools/sp_token_config.json` で定義されています：

| タイプ | プレフィックストークン | ポストフィックストークン | 用途 |
|------|---------------------|----------------------|------|
| **search** | [65530, 65529] | [65535, 11] | 検索クエリ/結果 |
| **system** | [65530, 65531] | [65535, 11] | システムプロンプト |
| **conversation** | [65530, 65532] | [65535, 11] | Q&A、対話 |
| **think** | [65530, 65533] | [65535, 11] | 内部推論 |
| **env** | [65530, 65534] | [65535, 11] | 環境コンテキスト |
| **common** | [65530] | [65535, 11] | 汎用 |
| **text** | [] | [] | プレーンテキスト（トークンなし） |

## フィールドマッピング設定

データフォルダ内に `{フォルダ名}.json` を作成してフィールドマッピングをカスタマイズできます：

```json
{
  "data": "sample",
  "text": "aaa",
  "conversation": "conversation",
  "system": "system"
}
```

**デフォルトマッピング**（設定ファイルがない場合）：
```python
{
  "data": "sample",
  "text": "text",
  "conversation": ["conversation", "query", "answer"],
  "system": ["system", "instruct"],
  "search": "search",
  "env": "env",
  "common": "common"
}
```

## コマンドライン引数

### 共通引数

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--datafolder` | カンマ区切りのデータフォルダパス | 必須 |
| `--output-prefix` | 出力ファイルのプレフィックス（.bin/.idx なし） | 必須 |
| `--vocab` | 語彙ファイルのパス | 必須 |
| `--tokenizer-type` | トークナイザータイプ（RWKVTokenizer、HFTokenizer、SentencePieceTokenizer） | 必須 |
| `--sp_token_config` | 特殊トークン設定 JSON | `./tools/sp_token_config.json` |
| `--dataset-impl` | データセット形式（mmap、lazy、cached） | `mmap` |
| `--append-eod` | 文書終了トークンを追加 | False |
| `--workers` | ワーカープロセス数 | 1 |
| `--log-interval` | 進捗更新間隔 | 100 |

## 利用可能な語彙ファイル

このリポジトリには3つの RWKV 語彙ファイルが含まれています：

- `rwkv_vocab_v20230424.txt` - RWKV-4-World 標準語彙
- `rwkv_vocab_ssg_train.txt` - SSG 学習用語彙
- `rwkv_vocab_ssg_eval.txt` - SSG 評価用語彙

## 出力ファイル

ツールは2つのファイルを生成します：

- `{output-prefix}_text_document.bin` - バイナリトークンデータ
- `{output-prefix}_text_document.idx` - インデックスメタデータ

これらのファイルは RWKV-LM 学習スクリプトで直接使用できます。

## 使用例

### 例1: カスタムマッピングを使った単一フォルダ

**フォルダ構造:**
```
my_data/
├── conversations.jsonl
└── my_data.json
```

**my_data.json:**
```json
{
  "data": "messages",
  "conversation": "dialog"
}
```

**conversations.jsonl:**
```json
{"messages": [{"dialog":"Q: こんにちは"},{"dialog":"A: やあ！"}]}
```

**コマンド:**
```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./my_data \
  --output-prefix ./output/my_dataset \
  --vocab ./rwkv_vocab_v20230424.txt \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

### 例2: 複数フォルダの処理

```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./folder1,./folder2,./folder3 \
  --output-prefix ./output/combined \
  --vocab ./rwkv_vocab_v20230424.txt \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

### 例3: 1ファイル内での形式混在

**data.jsonl:**
```json
{"sample": [{"text":"指示: 英語に翻訳してください"},{"conversation":"入力: こんにちは"},{"conversation":"出力: Hello"}]}
{"text": "特殊フォーマットなしのプレーン学習テキスト。"}
{"sample": [{"system":"あなたは親切なアシスタントです"},{"conversation":"ユーザー: 助けて"},{"conversation":"アシスタント: もちろん！"}]}
```

すべての行が正しく処理され、1つのデータセットに統合されます。

## SentencePiece トークナイザーサポート

このツールは SentencePiece トークナイザーの学習と使用をサポートしており、独自のコーパスに最適化されたカスタム語彙を作成できます。

### SentencePiece を使う理由

- **カスタム語彙**: ドメイン固有のトークナイザーを学習可能（コード、医療、法律など）
- **多言語サポート**: 文字カバレッジ設定による多言語テキストの適切な処理
- **バイトフォールバック**: バイトレベルフォールバックであらゆる Unicode 文字に対応
- **デフォルトで65536**: デフォルト語彙サイズ（65529）+ リポジトリ特殊トークン（7）= 合計65536

### SentencePiece モデルの学習

```bash
python tools/train_sentencepiece.py \
  --input ./training_data \
  --model-prefix ./models/my_tokenizer \
  --model-type bpe \
  --character-coverage 0.9995 \
  --byte-fallback
```

デフォルトでは `--vocab-size` は 65529 で、7つのリポジトリ特殊トークンと合わせて合計 65536 になります。

#### 学習用引数

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--input` | 入力ファイル/ディレクトリ（txt、jsonl 対応） | 必須 |
| `--model-prefix` | 出力モデルのプレフィックス（.model と .vocab を作成） | 必須 |
| `--vocab-size` | 語彙サイズ | 65529 |
| `--model-type` | モデルタイプ: unigram、bpe、char、word | bpe |
| `--character-coverage` | 学習時の文字カバレッジ | 0.9995 |
| `--byte-fallback` | 未知文字のバイトフォールバックを有効化 | 有効 |
| `--user-special-tokens` | カスタム特殊トークンの JSON ファイル | None |
| `--input-format` | 入力形式: auto、txt、jsonl | auto |
| `--jsonl-key` | JSONL からテキストを抽出するキー | text |
| `--num-threads` | 学習スレッド数 | 16 |

#### 語彙サイズの制限

SentencePiece の最大語彙サイズは **65529** です。これはリポジトリ特殊トークン（ID 65529-65535）用の領域を確保するためです。ユーザー定義の特殊トークンを使用する場合、制限はさらに減少します。

```
ID 0-65528: SentencePiece 語彙
ID 65529-65535: リポジトリ特殊トークン（sp_token_config.json で定義）
```

#### カスタム特殊トークン

カスタム特殊トークンを定義する JSON ファイルを作成できます：

```json
{
  "user_defined_symbols": ["<custom1>", "<custom2>"],
  "control_symbols": ["<ctrl1>"]
}
```

学習時に使用（カスタムトークン分の容量を確保するため vocab-size を減らす）：

```bash
python tools/train_sentencepiece.py \
  --input ./data \
  --model-prefix ./models/custom \
  --vocab-size 65526 \
  --user-special-tokens ./my_special_tokens.json
```

この例: 65526 + ユーザートークン3個 + リポジトリトークン7個 = 合計65536

### 前処理での SentencePiece の使用

学習後、SentencePiece モデルを前処理に使用できます：

```bash
# 標準前処理
python tools/preprocess_data.py \
  --input ./your_data.jsonl \
  --output-prefix ./output/data \
  --vocab ./models/my_tokenizer.model \
  --tokenizer-type SentencePieceTokenizer \
  --dataset-impl mmap \
  --append-eod

# SSG プロトコル前処理
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./your_folder \
  --sp_token_config ./tools/sp_token_config.json \
  --output-prefix ./output/data \
  --vocab ./models/my_tokenizer.model \
  --tokenizer-type SentencePieceTokenizer \
  --dataset-impl mmap \
  --append-eod
```

### 例: エンドツーエンドのワークフロー

1. **学習コーパスの準備**（テキストファイルまたは JSONL）：
```bash
# 複数ディレクトリから
ls ./corpus/
  wiki/
  books/
  conversations/
```

2. **SentencePiece モデルの学習**：
```bash
python tools/train_sentencepiece.py \
  --input ./corpus \
  --model-prefix ./models/my_rwkv_tokenizer \
  --model-type bpe \
  --character-coverage 0.9995
```

これにより 65529 トークン（デフォルト）のトークナイザーが作成され、リポジトリ特殊トークンと合わせて合計 65536 になります。

3. **RWKV 学習用データの前処理**：
```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./training_data \
  --output-prefix ./output/dataset \
  --vocab ./models/my_rwkv_tokenizer.model \
  --tokenizer-type SentencePieceTokenizer \
  --append-eod
```

4. **RWKV-LM** で事前学習を実行。

## トラブルシューティング

### エラー: "Index file doesn't match expected format"

`--dataset-impl` が既存ファイルの形式と一致しているか確認してください。形式を変更する場合は、古い `.bin`/`.idx` ファイルを削除してください。

### 大規模データセットでのメモリ不足

- `--dataset-impl mmap`（デフォルト）を使用してメモリマップファイルを利用
- メモリ不足の場合は `--workers` を減らす
- データを小さいバッチに分けて処理

### JSON パースエラー

- 各行が有効な JSON であることを確認
- UTF-8 エンコーディングを確認
- テキストにエンコーディング問題がある場合は `--ftfy` フラグを使用

## ライセンス

[EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)（Apache 2.0）と Facebook の fairseq（MIT）のコードをベースにしています。

## 関連プロジェクト

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) - RWKV 言語モデル
- [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros) - RWKV のオンライン学習
- [gpt-neox](https://github.com/EleutherAI/gpt-neox) - オリジナルの前処理ツールキット
