"""Train and evaluate a small multinomial Naive Bayes spam classifier."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "data" / "processed" / "train.csv"
DEFAULT_TEST = ROOT / "data" / "processed" / "test.csv"
LABELS = ("ham", "spam")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
URL_PATTERN = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"']+""")
KNOWN_TLDS = {"com", "org", "net", "edu", "gov"}


def tokenize(message: str) -> list[str]:
    """Normalize a message into lowercase alphanumeric word tokens."""
    return TOKEN_PATTERN.findall(message.lower())


def extract_urls(message: str) -> list[str]:
    """Find URLs with an http(s) scheme or a www prefix."""
    urls: list[str] = []
    for match in URL_PATTERN.findall(message):
        candidate = match.rstrip(".,!?;:)]}")
        parsed = urlsplit(candidate if "://" in candidate else f"http://{candidate}")
        if parsed.hostname:
            urls.append(parsed.hostname.lower())
    return urls


def extract_features(message: str) -> list[str]:
    """Combine word tokens with link-count and URL top-level-domain features."""
    hosts = extract_urls(message)
    link_count_feature = f"__feature_link_count_{len(hosts)}__"
    domain_features = []
    for host in hosts:
        tld = host.rsplit(".", 1)[-1]
        domain_type = tld if tld in KNOWN_TLDS else "other"
        domain_features.append(f"__feature_domain_tld_{domain_type}__")

    if not domain_features:
        domain_features.append("__feature_domain_tld_none__")
    return tokenize(message) + [link_count_feature] + domain_features


def load_messages(path: Path) -> list[dict[str, str]]:
    """Read messages and labels from a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"message", "label"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path} must contain message and label columns.")

        messages: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            message = (row.get("message") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not message or label not in LABELS:
                raise ValueError(f"{path}, line {line_number}: invalid message row.")
            messages.append({"message": message, "label": label})

    if not messages:
        raise ValueError(f"{path} does not contain any messages.")
    return messages


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace-smoothed word likelihoods."""

    def __init__(self) -> None:
        self.document_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_words: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.is_fitted = False

    def fit(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise ValueError("Training data must contain at least one message.")

        self.document_counts.clear()
        self.word_counts.clear()
        self.total_words.clear()
        self.vocabulary.clear()

        for row in messages:
            label = row["label"]
            if label not in LABELS:
                raise ValueError(f"Unsupported label: {label!r}.")
            words = extract_features(row["message"])
            self.document_counts[label] += 1
            self.word_counts[label].update(words)
            self.total_words[label] += len(words)
            self.vocabulary.update(words)

        self.is_fitted = True

    def predict(self, message: str) -> str:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        total_documents = sum(self.document_counts.values())
        vocabulary_size = max(1, len(self.vocabulary))
        scores: dict[str, float] = {}
        for label in LABELS:
            class_documents = self.document_counts[label]
            if class_documents == 0:
                scores[label] = float("-inf")
                continue
            score = math.log(class_documents / total_documents)
            denominator = self.total_words[label] + vocabulary_size
            for word, count in Counter(extract_features(message)).items():
                likelihood = (self.word_counts[label][word] + 1) / denominator
                score += count * math.log(likelihood)
            scores[label] = score

        return max(LABELS, key=lambda label: scores[label])

    def feature_weights(self) -> list[tuple[str, float]]:
        """Return each feature's spam-vs-ham log-likelihood ratio."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_weights().")

        vocabulary_size = max(1, len(self.vocabulary))
        spam_denominator = self.total_words["spam"] + vocabulary_size
        ham_denominator = self.total_words["ham"] + vocabulary_size
        weights = []
        for feature in self.vocabulary:
            spam_likelihood = (
                self.word_counts["spam"][feature] + 1
            ) / spam_denominator
            ham_likelihood = (
                self.word_counts["ham"][feature] + 1
            ) / ham_denominator
            weights.append(
                (feature, math.log(spam_likelihood / ham_likelihood))
            )
        return sorted(weights, key=lambda item: (-abs(item[1]), item[0]))


def evaluate(
    classifier: NaiveBayesClassifier, messages: list[dict[str, str]]
) -> dict[str, float | int]:
    """Calculate accuracy and spam precision/recall/F1 on labeled examples."""
    true_positive = true_negative = false_positive = false_negative = correct = 0
    for row in messages:
        prediction = classifier.predict(row["message"])
        actual = row["label"]
        correct += prediction == actual
        true_positive += prediction == "spam" and actual == "spam"
        true_negative += prediction == "ham" and actual == "ham"
        false_positive += prediction == "spam" and actual == "ham"
        false_negative += prediction == "ham" and actual == "spam"

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": len(messages),
        "correct": correct,
        "accuracy": correct / len(messages) if messages else 0.0,
        "spam_precision": precision,
        "spam_recall": recall,
        "spam_f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--show-data",
        action="store_true",
        help="print every labeled message used for training and testing",
    )
    args = parser.parse_args()

    training_messages = load_messages(args.train)
    test_messages = load_messages(args.test)
    if args.show_data:
        for heading, messages in (
            ("TRAINING DATA", training_messages),
            ("TEST DATA", test_messages),
        ):
            print(f"\n{heading} ({len(messages)} messages)")
            for index, row in enumerate(messages, start=1):
                print(f"{index:>2}. [{row['label']}] {row['message']}")

    classifier = NaiveBayesClassifier()
    classifier.fit(training_messages)
    metrics = evaluate(classifier, test_messages)

    print(f"Training messages: {len(training_messages)}")
    print(f"Test messages: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Spam precision: {metrics['spam_precision']:.1%}")
    print(f"Spam recall: {metrics['spam_recall']:.1%}")
    print(f"Spam F1: {metrics['spam_f1']:.1%}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted spam  Predicted legitimate")
    print(
        "Actual spam      "
        f"{metrics['true_positive']:>14}  "
        f"{metrics['false_negative']:>20}"
    )
    print(
        "Actual legitimate"
        f"{metrics['false_positive']:>14}  "
        f"{metrics['true_negative']:>20}"
    )
    print(
        "\nSpam correctly caught: "
        f"{metrics['true_positive']}; legitimate correctly passed: "
        f"{metrics['true_negative']}; legitimate wrongly marked spam: "
        f"{metrics['false_positive']}; spam wrongly let through: "
        f"{metrics['false_negative']}."
    )

    print("\nMost influential learned features (log P(feature|spam) / P(feature|ham)):")
    print("Positive weights point toward spam; negative weights point toward legitimate.")
    for feature, weight in classifier.feature_weights()[:10]:
        direction = "spam" if weight > 0 else "legitimate"
        print(f"  {feature:<32} {weight:>7.3f} ({direction})")

    print("\nExample predictions:")
    for message in (
        "Congratulations, you won a free prize. Claim it now!",
        "Hi, are we still meeting for lunch today?",
    ):
        print(f"[{classifier.predict(message)}] {message}")


if __name__ == "__main__":
    main()
