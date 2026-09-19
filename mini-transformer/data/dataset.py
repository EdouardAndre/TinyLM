from __future__ import annotations

import csv
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import re
from typing import Iterable, Protocol

import torch
from torch.utils.data import DataLoader, Dataset


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...


class CharacterTokenizer:
    """Tiny character-level tokenizer for first-principles LM experiments."""

    def __init__(self, vocabulary: Iterable[str]) -> None:
        special_tokens = ["<pad>", "<unk>"]
        chars = sorted(set(vocabulary) - set(special_tokens))
        self.itos = special_tokens + chars
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_token_id = self.stoi["<pad>"]
        self.unk_token_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.unk_token_id) for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


class BytePairTokenizer:
    """Small byte-pair-encoding tokenizer trained from text.

    This keeps tokenization transparent for the project: we start with characters,
    then learn frequent adjacent merges such as ("t", "h") -> "th".
    """

    def __init__(self, vocabulary: Iterable[str], merges: Iterable[tuple[str, str]]) -> None:
        special_tokens = ["<pad>", "<unk>"]
        tokens = special_tokens + sorted(set(vocabulary) - set(special_tokens))
        self.itos = tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.merges = list(merges)
        self.pad_token_id = self.stoi["<pad>"]
        self.unk_token_id = self.stoi["<unk>"]

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        min_pair_frequency: int = 2,
    ) -> "BytePairTokenizer":
        if vocab_size < 3:
            raise ValueError("vocab_size must leave room for special tokens and text tokens")
        if min_pair_frequency < 2:
            raise ValueError("min_pair_frequency must be at least 2")

        chunk_counts: Counter[tuple[str, ...]] = Counter()
        vocabulary: set[str] = set()
        for text in texts:
            for chunk in _split_text_chunks(text):
                symbols = tuple(chunk)
                if symbols:
                    chunk_counts[symbols] += 1
                    vocabulary.update(symbols)

        merges: list[tuple[str, str]] = []
        max_text_tokens = vocab_size - 2
        while len(vocabulary) < max_text_tokens:
            pair_counts = _count_adjacent_pairs(chunk_counts)
            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]
            if frequency < min_pair_frequency:
                break

            merged_token = "".join(best_pair)
            if merged_token in vocabulary:
                break

            merges.append(best_pair)
            vocabulary.add(merged_token)
            chunk_counts = _merge_pair_in_counts(chunk_counts, best_pair, merged_token)

        return cls(vocabulary, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for chunk in _split_text_chunks(text):
            symbols = list(chunk)
            for pair in self.merges:
                symbols = _merge_pair_in_symbols(symbols, pair, "".join(pair))
            token_ids.extend(self.stoi.get(symbol, self.unk_token_id) for symbol in symbols)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.itos[token_id]
            if token in {"<pad>", "<unk>"}:
                continue
            pieces.append(token)
        return "".join(pieces)


def _iter_csv_text(
    csv_path: str | Path,
    text_column: str = "text",
    *,
    max_chars: int | None = None,
) -> Iterable[str]:
    chars_seen = 0
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {text_column!r} was not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            text = row[text_column]
            if max_chars is not None:
                remaining = max_chars - chars_seen
                if remaining <= 0:
                    break
                text = text[:remaining]
            chars_seen += len(text)
            yield text


def _split_text_chunks(text: str) -> list[str]:
    return re.findall(r"\s+|[^\s]+", text)


def _count_adjacent_pairs(
    chunk_counts: Counter[tuple[str, ...]],
) -> Counter[tuple[str, str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for symbols, count in chunk_counts.items():
        for left, right in zip(symbols, symbols[1:]):
            pair_counts[(left, right)] += count
    return pair_counts


def _merge_pair_in_symbols(
    symbols: list[str],
    pair: tuple[str, str],
    merged_token: str,
) -> list[str]:
    merged_symbols: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged_symbols.append(merged_token)
            i += 2
        else:
            merged_symbols.append(symbols[i])
            i += 1
    return merged_symbols


def _merge_pair_in_counts(
    chunk_counts: Counter[tuple[str, ...]],
    pair: tuple[str, str],
    merged_token: str,
) -> Counter[tuple[str, ...]]:
    merged_counts: Counter[tuple[str, ...]] = Counter()
    for symbols, count in chunk_counts.items():
        merged_symbols = _merge_pair_in_symbols(list(symbols), pair, merged_token)
        merged_counts[tuple(merged_symbols)] += count
    return merged_counts


def build_character_tokenizer_from_csv(
    csv_paths: Iterable[str | Path],
    text_column: str = "text",
    *,
    max_chars_per_file: int | None = None,
) -> CharacterTokenizer:
    vocabulary: set[str] = set()
    for csv_path in csv_paths:
        for text in _iter_csv_text(
            csv_path,
            text_column=text_column,
            max_chars=max_chars_per_file,
        ):
            vocabulary.update(text)
    return CharacterTokenizer(vocabulary)


def build_bpe_tokenizer_from_csv(
    csv_paths: Iterable[str | Path],
    text_column: str = "text",
    *,
    vocab_size: int,
    max_chars_per_file: int | None = None,
    min_pair_frequency: int = 2,
) -> BytePairTokenizer:
    texts = (
        text
        for csv_path in csv_paths
        for text in _iter_csv_text(
            csv_path,
            text_column=text_column,
            max_chars=max_chars_per_file,
        )
    )
    return BytePairTokenizer.train(
        texts,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
    )


def build_tokenizer_from_csv(
    csv_paths: Iterable[str | Path],
    text_column: str = "text",
    *,
    tokenizer_type: str = "bpe",
    vocab_size: int = 512,
    max_chars_per_file: int | None = None,
    min_pair_frequency: int = 2,
) -> Tokenizer:
    if tokenizer_type == "char":
        return build_character_tokenizer_from_csv(
            csv_paths,
            text_column=text_column,
            max_chars_per_file=max_chars_per_file,
        )
    if tokenizer_type == "bpe":
        return build_bpe_tokenizer_from_csv(
            csv_paths,
            text_column=text_column,
            vocab_size=vocab_size,
            max_chars_per_file=max_chars_per_file,
            min_pair_frequency=min_pair_frequency,
        )
    raise ValueError("tokenizer_type must be either 'char' or 'bpe'")


class LanguageModelingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Creates overlapping next-token prediction examples.

    If `context_length` is T, each item returns:
      input_ids: [T]
      targets:   [T]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        if len(token_ids) <= context_length:
            raise ValueError(
                "Need more tokens than context_length to create next-token examples"
            )

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        tokenizer: Tokenizer,
        context_length: int,
        text_column: str = "text",
        *,
        max_chars: int | None = None,
        document_separator: str = "\n",
    ) -> "LanguageModelingDataset":
        token_ids: list[int] = []
        for text in _iter_csv_text(csv_path, text_column=text_column, max_chars=max_chars):
            token_ids.extend(tokenizer.encode(text))
            token_ids.extend(tokenizer.encode(document_separator))
        return cls(token_ids, context_length)

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.context_length + 1]
        input_ids = chunk[:-1]
        targets = chunk[1:]
        return input_ids, targets


@dataclass
class TinyStoriesDataModule:
    train_path: str | Path
    validation_path: str | Path
    context_length: int
    batch_size: int
    text_column: str = "text"
    tokenizer_type: str = "bpe"
    vocab_size: int = 512
    min_pair_frequency: int = 2
    max_train_chars: int | None = None
    max_validation_chars: int | None = None

    def setup(self) -> None:
        self.tokenizer = build_tokenizer_from_csv(
            [self.train_path, self.validation_path],
            text_column=self.text_column,
            tokenizer_type=self.tokenizer_type,
            vocab_size=self.vocab_size,
            max_chars_per_file=self.max_train_chars,
            min_pair_frequency=self.min_pair_frequency,
        )
        self.train_dataset = LanguageModelingDataset.from_csv(
            self.train_path,
            self.tokenizer,
            self.context_length,
            self.text_column,
            max_chars=self.max_train_chars,
        )
        self.validation_dataset = LanguageModelingDataset.from_csv(
            self.validation_path,
            self.tokenizer,
            self.context_length,
            self.text_column,
            max_chars=self.max_validation_chars,
        )

    def train_dataloader(self, *, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    def validation_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
