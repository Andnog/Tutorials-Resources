"""Metricas deterministas para respuestas estructuradas y texto."""

import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def normalize_text(value: str) -> str:
    """Normalize superficial differences before exact comparison."""
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(character for character in value if unicodedata.category(character) != "Mn")
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def is_valid_json(value: str) -> bool:
    try:
        json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False
    return True


def json_exact_match(value: str, reference: dict[str, Any]) -> float:
    if not is_valid_json(value):
        return 0.0
    return float(json.loads(value) == reference)


def classification_scores(expected: Iterable[str], predicted: Iterable[str]) -> dict[str, Any]:
    expected_list, predicted_list = list(expected), list(predicted)
    labels = sorted(set(expected_list) | set(predicted_list))
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected_list, predicted_list, average="weighted", zero_division=0
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "labels": labels,
        "confusion_matrix": confusion_matrix(expected_list, predicted_list, labels=labels),
    }


def tool_sequence_match(expected: list[str], actual: list[str]) -> float:
    return float(expected == actual)


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    return float(expected == actual)


def bleu_score(prediction: str, reference: str) -> float:
    return float(BLEU(effective_order=True).sentence_score(prediction, [reference]).score / 100)


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False).score(reference, prediction)
    return {name: float(score.fmeasure) for name, score in scores.items()}
