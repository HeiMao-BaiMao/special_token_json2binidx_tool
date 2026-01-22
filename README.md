# jsonl to binidx tool (using special token)

A simplified tool for converting JSONL files into binary (.bin) and index (.idx) format for RWKV model dataset preparation.

This repository is greatly simplified from [gpt-neox](https://github.com/EleutherAI/gpt-neox) to ONLY handle JSONL to binidx conversion with special token support for [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) pretraining.

**Workflow**: datasets → special_token_json2binidx_tool → RWKV-LM(-infctx) pretraining → base LLM → [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros) online training

[日本語版 README はこちら](README.ja.md)

## Features

- Convert JSONL to binary format for efficient RWKV training
- Support for special tokens (conversation, system, search, think, env, common, text)
- Mix SSG protocol format and standard JSONL format in the same dataset
- Configurable field mapping per data folder
- Multiple tokenizer support (RWKV, HuggingFace, SentencePiece)
- **SentencePiece tokenizer training** - Train custom tokenizers from your own corpus

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- ftfy==6.1.*
- lm-dataformat==0.0.20
- numpy==1.24.*
- tokenizers==0.13.*
- torch==2.0.*
- tqdm==4.65.*
- sentencepiece>=0.1.99

## Quick Start

### Basic Usage (SSG Protocol with Special Tokens)

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

### Standard JSONL Format (No Special Tokens)

```bash
python tools/preprocess_data.py \
  --input ./your_data.jsonl \
  --output-prefix ./data/output \
  --vocab ./rwkv_vocab_v20230424.txt \
  --dataset-impl mmap \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

## Data Folder Structure

```
your_folder/
├── a.jsonl
├── b.jsonl
└── your_folder.json  (optional mapping config)
```

The optional `your_folder.json` allows you to customize field mappings for that specific folder.

## JSONL Format

### SSG Protocol Format (with special tokens)

One line per document:

```json
{"sample": [{"text":"Introduction text"},{"conversation":"Question: What is RWKV?"},{"conversation":"Answer: RWKV is a language model."},{"system":"This is system information"}]}
```

### Standard Format (plain text)

```json
{"text": "This is a simple text document without special tokens."}
```

### Mixed Format (NEW!)

You can now mix both formats in the same JSONL file:

```json
{"sample": [{"text":"This uses SSG protocol"}]}
{"text": "This is standard format"}
{"sample": [{"conversation":"Q: Question?"},{"conversation":"A: Answer"}]}
{"text": "Another standard format line"}
```

## Special Token Types

Defined in `tools/sp_token_config.json`:

| Type | Prefix Tokens | Postfix Tokens | Use Case |
|------|--------------|----------------|----------|
| **search** | [65530, 65529] | [65535, 11] | Search queries/results |
| **system** | [65530, 65531] | [65535, 11] | System prompts |
| **conversation** | [65530, 65532] | [65535, 11] | Q&A, dialogue |
| **think** | [65530, 65533] | [65535, 11] | Internal reasoning |
| **env** | [65530, 65534] | [65535, 11] | Environment context |
| **common** | [65530] | [65535, 11] | General purpose |
| **text** | [] | [] | Plain text (no tokens) |

## Field Mapping Configuration

Create a `{folder_name}.json` in your data folder to customize field mappings:

```json
{
  "data": "sample",
  "text": "aaa",
  "conversation": "conversation",
  "system": "system"
}
```

**Default mapping** (if no config file):
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

## Command Line Arguments

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--datafolder` | Comma-separated data folders | Required |
| `--output-prefix` | Output file prefix (without .bin/.idx) | Required |
| `--vocab` | Path to vocabulary file | Required |
| `--tokenizer-type` | Tokenizer type (RWKVTokenizer, HFTokenizer, SentencePieceTokenizer) | Required |
| `--sp_token_config` | Special token config JSON | `./tools/sp_token_config.json` |
| `--dataset-impl` | Dataset format (mmap, lazy, cached) | `mmap` |
| `--append-eod` | Append end-of-document token | False |
| `--workers` | Number of worker processes | 1 |
| `--log-interval` | Progress update interval | 100 |

## Available Vocabularies

This repository includes three RWKV vocabularies:

- `rwkv_vocab_v20230424.txt` - RWKV-4-World standard vocabulary
- `rwkv_vocab_ssg_train.txt` - SSG training vocabulary
- `rwkv_vocab_ssg_eval.txt` - SSG evaluation vocabulary

## Output Files

The tool generates two files:

- `{output-prefix}_text_document.bin` - Binary token data
- `{output-prefix}_text_document.idx` - Index metadata

These files can be directly used with RWKV-LM training scripts.

## Examples

### Example 1: Single folder with custom mapping

**Folder structure:**
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
{"messages": [{"dialog":"Q: Hello?"},{"dialog":"A: Hi there!"}]}
```

**Command:**
```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./my_data \
  --output-prefix ./output/my_dataset \
  --vocab ./rwkv_vocab_v20230424.txt \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

### Example 2: Multiple folders

```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./folder1,./folder2,./folder3 \
  --output-prefix ./output/combined \
  --vocab ./rwkv_vocab_v20230424.txt \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

### Example 3: Mixed format in one file

**data.jsonl:**
```json
{"sample": [{"text":"Instruction: Translate to English"},{"conversation":"Input: こんにちは"},{"conversation":"Output: Hello"}]}
{"text": "Plain training text without special formatting."}
{"sample": [{"system":"You are a helpful assistant"},{"conversation":"User: Help me"},{"conversation":"Assistant: Sure!"}]}
```

All lines will be processed correctly and merged into one dataset.

## SentencePiece Tokenizer Support

This tool supports training and using SentencePiece tokenizers, allowing you to create custom vocabularies optimized for your specific corpus.

### Why Use SentencePiece?

- **Custom Vocabulary**: Train a tokenizer optimized for your domain (e.g., code, medical, legal)
- **Language Support**: Better handling of multilingual text with character coverage settings
- **Byte Fallback**: Handle any Unicode character with byte-level fallback
- **Flexible Size**: Choose vocabulary size from small (1K) to large (65K+)

### Training a SentencePiece Model

```bash
python tools/train_sentencepiece.py \
  --input ./training_data \
  --model-prefix ./models/my_tokenizer \
  --vocab-size 32000 \
  --model-type bpe \
  --character-coverage 0.9995 \
  --byte-fallback
```

#### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input file(s) or directory (txt, jsonl supported) | Required |
| `--model-prefix` | Output model prefix (creates .model and .vocab) | Required |
| `--vocab-size` | Vocabulary size (max: 65529) | 32000 |
| `--model-type` | Model type: unigram, bpe, char, word | bpe |
| `--character-coverage` | Character coverage for training | 0.9995 |
| `--byte-fallback` | Enable byte fallback for unknown chars | Enabled |
| `--user-special-tokens` | JSON file with custom special tokens | None |
| `--input-format` | Input format: auto, txt, jsonl | auto |
| `--jsonl-key` | Key to extract text from JSONL | text |
| `--num-threads` | Number of training threads | 16 |

#### Vocabulary Size Limit

The maximum SentencePiece vocabulary size is **65529** to reserve space for repository special tokens (ID 65529-65535). If using user-defined special tokens, the limit is further reduced.

```
ID 0-65528: SentencePiece vocabulary
ID 65529-65535: Repository special tokens (defined in sp_token_config.json)
```

#### Custom Special Tokens

Create a JSON file to define custom special tokens:

```json
{
  "user_defined_symbols": ["<custom1>", "<custom2>"],
  "control_symbols": ["<ctrl1>"]
}
```

Then use it during training:

```bash
python tools/train_sentencepiece.py \
  --input ./data \
  --model-prefix ./models/custom \
  --vocab-size 30000 \
  --user-special-tokens ./my_special_tokens.json
```

### Using SentencePiece with Preprocessing

After training, use your SentencePiece model for preprocessing:

```bash
# Standard preprocessing
python tools/preprocess_data.py \
  --input ./your_data.jsonl \
  --output-prefix ./output/data \
  --vocab ./models/my_tokenizer.model \
  --tokenizer-type SentencePieceTokenizer \
  --dataset-impl mmap \
  --append-eod

# SSG protocol preprocessing
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./your_folder \
  --sp_token_config ./tools/sp_token_config.json \
  --output-prefix ./output/data \
  --vocab ./models/my_tokenizer.model \
  --tokenizer-type SentencePieceTokenizer \
  --dataset-impl mmap \
  --append-eod
```

### Example: End-to-End Workflow

1. **Prepare training corpus** (text files or JSONL):
```bash
# From multiple directories
ls ./corpus/
  wiki/
  books/
  conversations/
```

2. **Train SentencePiece model**:
```bash
python tools/train_sentencepiece.py \
  --input ./corpus \
  --model-prefix ./models/my_rwkv_tokenizer \
  --vocab-size 50000 \
  --model-type bpe \
  --character-coverage 0.9995
```

3. **Preprocess data for RWKV training**:
```bash
python tools/preprocess_ssg_protocol_data.py \
  --datafolder ./training_data \
  --output-prefix ./output/dataset \
  --vocab ./models/my_rwkv_tokenizer.model \
  --tokenizer-type SentencePieceTokenizer \
  --append-eod
```

4. **Use with RWKV-LM** for pretraining.

## Troubleshooting

### Error: "Index file doesn't match expected format"

Make sure `--dataset-impl` matches the format of existing files. Delete old `.bin`/`.idx` files if changing format.

### Memory issues with large datasets

- Use `--dataset-impl mmap` (default) for memory-mapped files
- Reduce `--workers` if running out of memory
- Process data in smaller batches

### JSON parsing errors

- Ensure each line is valid JSON
- Check for proper UTF-8 encoding
- Use `ftfy` flag if text has encoding issues: `--ftfy`

## License

Based on code from [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) (Apache 2.0) and Facebook's fairseq (MIT).

## Related Projects

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) - RWKV Language Model
- [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros) - Online training for RWKV
- [gpt-neox](https://github.com/EleutherAI/gpt-neox) - Original preprocessing toolkit
