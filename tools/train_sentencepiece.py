# Copyright (c) 2024
# SentencePiece tokenizer training script for RWKV preprocessing tool.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a SentencePiece vocabulary for RWKV preprocessing."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

import sentencepiece as spm


# Default total vocabulary size
DEFAULT_VOCAB_SIZE = 65536


def get_args():
    parser = argparse.ArgumentParser(
        description="Build a SentencePiece vocabulary for RWKV preprocessing"
    )

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file(s) or directory. "
        "Supports .txt and .jsonl files. "
        "For directories, all .txt and .jsonl files will be used. "
        "Multiple paths can be comma-separated.",
    )
    group.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "txt", "jsonl"],
        help="Input format: auto (detect from extension), txt, or jsonl. Default: auto",
    )
    group.add_argument(
        "--jsonl-key",
        type=str,
        default="text",
        help="Key to extract text from JSONL files. Default: text",
    )

    group = parser.add_argument_group(title="output")
    group.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="Output model prefix. Will create {prefix}.model and {prefix}.vocab files.",
    )

    group = parser.add_argument_group(title="tokenizer settings")
    group.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Total vocabulary size target. Default: {DEFAULT_VOCAB_SIZE}. "
        f"SentencePiece vocab = vocab-size - special tokens count.",
    )
    group.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["unigram", "bpe", "char", "word"],
        help="Model type: unigram, bpe, char, word. Default: bpe",
    )
    group.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Character coverage for vocabulary building. Default: 1.0",
    )
    group.add_argument(
        "--byte-fallback",
        action="store_true",
        default=True,
        help="Enable byte fallback for unknown characters. Default: enabled",
    )
    group.add_argument(
        "--no-byte-fallback",
        action="store_true",
        help="Disable byte fallback for unknown characters.",
    )

    group = parser.add_argument_group(title="special tokens")
    group.add_argument(
        "--special-tokens",
        type=str,
        default=None,
        help="Path to text file defining special tokens (one token per line). "
        "These are added as SentencePiece user_defined_symbols.",
    )

    group = parser.add_argument_group(title="build options")
    group.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Maximum number of sentences to use. "
        "0 means use all sentences. Default: 0",
    )
    group.add_argument(
        "--shuffle-input-sentence",
        action="store_true",
        help="Randomly shuffle input sentences.",
    )
    group.add_argument(
        "--seed-sentencepiece-size",
        type=int,
        default=1000000,
        help="Seed sentencepiece size. Default: 1000000",
    )
    group.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of threads. Default: 16",
    )
    group.add_argument(
        "--train-extremely-large-corpus",
        action="store_true",
        help="Enable for extremely large corpus (may use more memory).",
    )

    args = parser.parse_args()

    # Handle byte_fallback logic
    if args.no_byte_fallback:
        args.byte_fallback = False

    return args


def read_txt_file(filepath: str) -> Iterator[str]:
    """Read lines from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def read_jsonl_file(filepath: str, key: str) -> Iterator[str]:
    """Read text from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if key in data:
                        text = data[key]
                        if isinstance(text, str) and text.strip():
                            yield text.strip()
                        elif isinstance(text, list):
                            # Handle SSG protocol format
                            for item in text:
                                if isinstance(item, dict):
                                    for v in item.values():
                                        if isinstance(v, str) and v.strip():
                                            yield v.strip()
                except json.JSONDecodeError:
                    continue


def get_input_files(input_paths: str) -> List[str]:
    """Get all input files from comma-separated paths."""
    files = []
    for path in input_paths.split(","):
        path = path.strip()
        if not path:
            continue

        p = Path(path)
        if p.is_file():
            files.append(str(p))
        elif p.is_dir():
            # Recursively find all .txt and .jsonl files
            files.extend(str(f) for f in p.rglob("*.txt"))
            files.extend(str(f) for f in p.rglob("*.jsonl"))
        else:
            print(f"Warning: Path not found: {path}", file=sys.stderr)

    return sorted(set(files))


def detect_format(filepath: str, default_format: str) -> str:
    """Detect file format from extension or use default."""
    if default_format != "auto":
        return default_format

    ext = Path(filepath).suffix.lower()
    if ext == ".jsonl":
        return "jsonl"
    else:
        return "txt"


def collect_text_to_file(
    input_files: List[str],
    input_format: str,
    jsonl_key: str,
    output_file: str,
) -> int:
    """Collect all text from input files into a single file for training."""
    total_lines = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for filepath in input_files:
            fmt = detect_format(filepath, input_format)
            print(f"Processing: {filepath} (format: {fmt})")

            if fmt == "jsonl":
                iterator = read_jsonl_file(filepath, jsonl_key)
            else:
                iterator = read_txt_file(filepath)

            for line in iterator:
                out.write(line + "\n")
                total_lines += 1

    return total_lines


def load_special_tokens(filepath: Optional[str]) -> List[str]:
    """Load special tokens from text file (one token per line)."""
    if filepath is None:
        return []

    tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    return tokens


def main():
    args = get_args()

    # Load special tokens
    special_tokens = load_special_tokens(args.special_tokens)
    special_token_count = len(special_tokens)

    # Calculate actual SentencePiece vocab size from total target
    # total = sp_vocab + special_tokens
    # sp_vocab = total - special_tokens
    sp_vocab_size = args.vocab_size - special_token_count

    if sp_vocab_size <= 0:
        print(
            f"Error: vocab_size ({args.vocab_size}) is too small.\n"
            f"  - Special tokens: {special_token_count}\n"
            f"  Resulting SentencePiece vocab would be: {sp_vocab_size}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Target total vocab size: {args.vocab_size}")
    print(f"  - Special tokens: {special_token_count}")
    print(f"  = SentencePiece vocab size: {sp_vocab_size}")
    if special_tokens:
        print(f"  Special tokens: {special_tokens}")

    # Get input files
    input_files = get_input_files(args.input)
    if not input_files:
        print("Error: No input files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} input file(s)")

    # Create output directory if needed
    output_dir = Path(args.model_prefix).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Collect all text into a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name

    try:
        total_lines = collect_text_to_file(
            input_files, args.input_format, args.jsonl_key, tmp_path
        )
        print(f"Collected {total_lines} lines of text")

        if total_lines == 0:
            print("Error: No text found in input files.", file=sys.stderr)
            sys.exit(1)

        # Build SentencePiece training arguments
        train_args = {
            "input": tmp_path,
            "model_prefix": args.model_prefix,
            "vocab_size": sp_vocab_size,
            "model_type": args.model_type,
            "character_coverage": args.character_coverage,
            "byte_fallback": args.byte_fallback,
            "num_threads": args.num_threads,
            "seed_sentencepiece_size": args.seed_sentencepiece_size,
        }

        # Add optional arguments
        if args.input_sentence_size > 0:
            train_args["input_sentence_size"] = args.input_sentence_size

        if args.shuffle_input_sentence:
            train_args["shuffle_input_sentence"] = True

        if args.train_extremely_large_corpus:
            train_args["train_extremely_large_corpus"] = True

        # Add special tokens
        if special_tokens:
            train_args["user_defined_symbols"] = ",".join(special_tokens)

        # Build the vocabulary
        print("\nBuilding SentencePiece vocabulary...")
        print(f"  Model type: {args.model_type}")
        print(f"  SentencePiece vocab size: {sp_vocab_size}")
        print(f"  Character coverage: {args.character_coverage}")
        print(f"  Byte fallback: {args.byte_fallback}")

        spm.SentencePieceTrainer.train(**train_args)

        print(f"\nModel saved to:")
        print(f"  {args.model_prefix}.model")
        print(f"  {args.model_prefix}.vocab")

        # Verify the model
        sp = spm.SentencePieceProcessor()
        sp.load(f"{args.model_prefix}.model")
        actual_vocab_size = sp.get_piece_size()

        print(f"\nModel verification:")
        print(f"  Total vocab size: {actual_vocab_size}")
        print(f"  (includes {special_token_count} special tokens)")
        print(f"\nBuilt-in special token IDs:")
        print(f"  UNK ID: {sp.unk_id()}")
        print(f"  BOS ID: {sp.bos_id()}")
        print(f"  EOS ID: {sp.eos_id()}")
        print(f"  PAD ID: {sp.pad_id()}")

        # Show user-defined special token IDs
        if special_tokens:
            print(f"\nUser-defined special token IDs:")
            for token in special_tokens:
                token_id = sp.piece_to_id(token)
                print(f"  {token}: {token_id}")

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
