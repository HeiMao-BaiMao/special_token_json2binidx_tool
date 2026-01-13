# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This tool converts JSONL files to binary (.bin) and index (.idx) format for RWKV model dataset preparation. It's a simplified version of gpt-neox preprocessing, focused specifically on handling special tokens for RWKV training.

The tool is designed for the workflow: datasets → special_token_json2binidx_tool → RWKV-LM pretraining → base LLM.

## Core Commands

### Basic Preprocessing (SSG Protocol)
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

### Standard Preprocessing (without special tokens)
```bash
python tools/preprocess_data.py \
  --input ./your_data \
  --output-prefix ./data/output \
  --vocab ./rwkv_vocab_v20230424.txt \
  --dataset-impl mmap \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

## Architecture

### Entry Points
- **tools/preprocess_ssg_protocol_data.py**: Main preprocessing script for SSG protocol data with special token support
- **tools/preprocess_data.py**: Basic preprocessing for standard JSONL data without special tokens

### Core Components

**Tokenization Pipeline** (tools/tokenizer.py):
- `build_tokenizer()`: Factory function that initializes tokenizer based on type
- `RWKVTokenizer`: Wraps TRIE_TOKENIZER for RWKV vocab (default EOD=0)
- `HFTokenizer`: HuggingFace tokenizer integration
- Token vocabulary padding handled by `_vocab_size_with_padding()`

**Special Token System** (tools/sp_token_config.json):
- Defines prefix/postfix token IDs for different content types: search, system, conversation, think, env, common, text
- Each type has configurable prefix and postfix token arrays (e.g., conversation uses [65530, 65532] prefix, [65535, 11] postfix)
- Empty prefix/postfix for "text" type allows plain text without special markers

**Dataset Builder** (tools/indexed_dataset.py):
- `MMapIndexedDatasetBuilder`: Memory-mapped builder for efficient large dataset handling
- `IndexedDatasetBuilder`: Standard in-memory builder
- `make_builder()`: Factory that selects builder based on impl type (mmap/lazy/cached)
- Dataset format: binary data (.bin) + index metadata (.idx)

### Data Processing Flow (SSG Protocol)

1. **Folder Processing**: Reads comma-separated datafolders, each containing .jsonl files
2. **Config Mapping**: Each datafolder can have a {folder_name}.json config that maps data fields
   - Default mapping_dict in preprocess_ssg_protocol_data.py:157-210
   - Config file overrides defaults via `mapping.update(config)`
3. **JSONL Parsing**: Each line contains a document with "data" field (mapped from config)
4. **Field Remapping**: Converts custom field names to standard keys using `find_key_by_value()`
5. **Token Assembly**: For each entity in flow:
   - Extract role key and content value
   - Construct: `sp_token_config[role]["prefix"] + tokenize(content) + sp_token_config[role]["postfix"]`
6. **Dataset Building**: Accumulated tokens written to .bin/.idx via indexed_dataset builders

### Key Data Structures

**Expected JSONL Format**:
```json
{"sample": [
  {"aaa": "以下是一段对话"},
  {"conversation": "Question: 你是谁？"},
  {"conversation": "Answer: 我是AI"}
]}
```

**Mapping Config Format** ({folder_name}.json):
```json
{
  "data": "sample",
  "text": "aaa",
  "conversation": "conversation",
  "system": "system"
}
```

## Available Tokenizers

Set via `--tokenizer-type`:
- **RWKVTokenizer**: For RWKV models (uses rwkv_vocab_v20230424.txt)
- **HFTokenizer**: HuggingFace tokenizers
- GPT2BPETokenizer, CharLevelTokenizer, TiktokenTokenizer (defined in choices but not implemented)

## Dataset Implementation Types

Set via `--dataset-impl`:
- **mmap** (default): Memory-mapped for large datasets
- **cached**: Prefetch to memory cache
- **lazy**: Load on demand

## Dependencies

See requirements.txt:
- ftfy==6.1.* (text cleaning)
- lm-dataformat==0.0.20 (data format handling)
- numpy==1.24.*
- tokenizers==0.13.*
- torch==2.0.*
- tqdm==4.65.*
