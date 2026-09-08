from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


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


def build_tokenizer_from_csv(
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
        tokenizer: CharacterTokenizer,
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
    max_train_chars: int | None = None
    max_validation_chars: int | None = None

    def setup(self) -> None:
        self.tokenizer = build_tokenizer_from_csv(
            [self.train_path, self.validation_path],
            text_column=self.text_column,
            max_chars_per_file=self.max_train_chars,
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
