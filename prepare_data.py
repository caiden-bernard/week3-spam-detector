"""Split the bundled SMS examples into deterministic training and test files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "messages.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
LABELS = ("ham", "spam")


def load_messages(path: Path) -> list[dict[str, str]]:
    """Load and validate rows with ``message`` and ``label`` columns."""
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"message", "label"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Input CSV must contain message and label columns.")

        messages: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            message = (row.get("message") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not message:
                raise ValueError(f"Line {line_number} has an empty message.")
            if label not in LABELS:
                raise ValueError(
                    f"Line {line_number} has invalid label {label!r}; "
                    "expected 'ham' or 'spam'."
                )
            messages.append({"message": message, "label": label})

    if not messages:
        raise ValueError("Input CSV does not contain any messages.")
    return messages


def split_messages(
    messages: list[dict[str, str]], test_size: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return stratified train and test splits using a reproducible seed."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    grouped = {label: [] for label in LABELS}
    for row in messages:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for label in LABELS:
        rows = grouped[label]
        if len(rows) < 2:
            raise ValueError(
                f"At least two {label!r} examples are required for a split."
            )
        rng.shuffle(rows)
        test_count = min(len(rows) - 1, max(1, round(len(rows) * test_size)))
        test.extend(rows[:test_count])
        train.extend(rows[test_count:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_messages(path: Path, messages: list[dict[str, str]]) -> None:
    """Write messages as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["message", "label"])
        writer.writeheader()
        writer.writerows(messages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    messages = load_messages(args.input)
    train, test = split_messages(messages, args.test_size, args.seed)
    write_messages(args.output_dir / "train.csv", train)
    write_messages(args.output_dir / "test.csv", test)
    print(
        f"Prepared {len(messages)} messages: {len(train)} train, "
        f"{len(test)} test (seed={args.seed})."
    )


if __name__ == "__main__":
    main()
