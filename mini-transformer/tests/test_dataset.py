import csv

import torch

from data.dataset import (
    BytePairTokenizer,
    CharacterTokenizer,
    LanguageModelingDataset,
    TinyStoriesDataModule,
    build_tokenizer_from_csv,
    tokenizer_from_state_dict,
)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text"])
        writer.writeheader()
        for text in rows:
            writer.writerow({"text": text})


def test_character_tokenizer_round_trips_known_text():
    tokenizer = CharacterTokenizer("abc")

    token_ids = tokenizer.encode("cab")

    assert tokenizer.decode(token_ids) == "cab"
    assert tokenizer.vocab_size == 5


def test_bpe_tokenizer_learns_frequent_pairs_and_round_trips_text():
    tokenizer = BytePairTokenizer.train(["low lower lowest"], vocab_size=20)

    token_ids = tokenizer.encode("lower")

    assert len(token_ids) < len("lower")
    assert tokenizer.decode(token_ids) == "lower"


def test_bpe_tokenizer_state_round_trips_without_retraining():
    tokenizer = BytePairTokenizer.train(["low lower lowest"], vocab_size=20)

    restored = tokenizer_from_state_dict(tokenizer.state_dict())

    assert restored.encode("lower") == tokenizer.encode("lower")
    assert restored.decode(restored.encode("lowest")) == "lowest"


def test_language_modeling_dataset_returns_next_token_pairs():
    tokenizer = CharacterTokenizer("ABCDE")
    dataset = LanguageModelingDataset(tokenizer.encode("ABCDE"), context_length=4)

    input_ids, targets = dataset[0]

    assert input_ids.tolist() == tokenizer.encode("ABCD")
    assert targets.tolist() == tokenizer.encode("BCDE")
    assert input_ids.shape == torch.Size([4])
    assert targets.shape == torch.Size([4])


def test_csv_data_module_batches_are_batch_by_time(tmp_path):
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    _write_csv(train_csv, ["abcd efgh", "ijkl mnop"])
    _write_csv(validation_csv, ["qrst uvwx"])

    data = TinyStoriesDataModule(
        train_path=train_csv,
        validation_path=validation_csv,
        context_length=4,
        batch_size=2,
        vocab_size=32,
    )
    data.setup()

    input_ids, targets = next(iter(data.train_dataloader(shuffle=False)))

    assert input_ids.shape == torch.Size([2, 4])
    assert targets.shape == torch.Size([2, 4])
    assert targets[0].tolist() == input_ids[0, 1:].tolist() + [targets[0, -1].item()]


def test_build_tokenizer_from_csv_uses_selected_text_column(tmp_path):
    csv_path = tmp_path / "stories.csv"
    _write_csv(csv_path, ["tiny story"])

    tokenizer = build_tokenizer_from_csv([csv_path], vocab_size=32)

    assert tokenizer.decode(tokenizer.encode("tiny story")) == "tiny story"


def test_build_tokenizer_from_csv_can_still_build_character_tokenizer(tmp_path):
    csv_path = tmp_path / "stories.csv"
    _write_csv(csv_path, ["tiny story"])

    tokenizer = build_tokenizer_from_csv([csv_path], tokenizer_type="char")

    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.decode(tokenizer.encode("tiny story")) == "tiny story"
