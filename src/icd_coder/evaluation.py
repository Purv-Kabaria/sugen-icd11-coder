"""Metrics for gold-labelled notes: the official score plus diagnostics.

`top1`, `top3` and `points` reproduce `score.py`. The rest explain the misses:
`family_top1` counts a top-1 code in the right family (same depth-one stem), which
separates wrong-specificity errors from wrong-disease errors, and `chapter_top1` does
the same at chapter level.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from icd_coder.catalogue import Catalogue, normalise_code

SLICE_POINTS = {"A": 75.0, "B": 25.0}
TOP1_SHARE = 2 / 3


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    gold: frozenset[str]
    slice: str
    style: str | None


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    count: int
    top1: float
    top3: float
    mrr: float
    family_top1: float
    chapter_top1: float
    invalid_rate: float
    missing_rate: float

    def points(self, weight: float) -> float:
        return weight * (TOP1_SHARE * self.top1 + (1 - TOP1_SHARE) * self.top3)


def load_examples(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [
        Example(
            id=str(row["id"]),
            text=row["text"],
            gold=frozenset(code for code in map(normalise_code, row["gold"]) if code),
            slice=row.get("slice", "A"),
            style=row.get("style"),
        )
        for row in rows
    ]


def evaluate(
    examples: Iterable[Example], predictions: Mapping[str, Sequence[str]], catalogue: Catalogue
) -> dict[str, SliceMetrics]:
    """Metrics per slice, plus an `all` entry when there is more than one slice."""
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.slice, []).append(example)
    metrics = {
        name: _slice_metrics(members, predictions, catalogue) for name, members in groups.items()
    }
    if len(groups) > 1:
        everything = [example for members in groups.values() for example in members]
        metrics["all"] = _slice_metrics(everything, predictions, catalogue)
    return dict(sorted(metrics.items()))


def official_points(metrics: Mapping[str, SliceMetrics]) -> tuple[float, float]:
    """Points earned and points available over the slices present, as `score.py` reports."""
    earned = available = 0.0
    for name, weight in SLICE_POINTS.items():
        if name in metrics:
            earned += metrics[name].points(weight)
            available += weight
    return earned, available


def _slice_metrics(
    examples: Sequence[Example], predictions: Mapping[str, Sequence[str]], catalogue: Catalogue
) -> SliceMetrics:
    top1 = top3 = reciprocal = family = chapter = invalid = missing = 0.0
    for example in examples:
        raw = predictions.get(example.id)
        if raw is None:
            missing += 1
            continue
        codes = [catalogue.validate(code) for code in list(raw)[:3]]
        invalid += sum(code is None for code in codes)
        ranks = [rank for rank, code in enumerate(codes, 1) if code in example.gold]
        top1 += bool(ranks) and ranks[0] == 1
        top3 += bool(ranks)
        reciprocal += 1 / ranks[0] if ranks else 0.0
        first = codes[0] if codes else None
        if first is not None:
            family += any(catalogue.stem(first) == catalogue.stem(gold) for gold in example.gold)
            chapter += any(
                catalogue[first].chapter == catalogue[gold].chapter for gold in example.gold
            )
    count = len(examples)
    return SliceMetrics(
        count=count,
        top1=top1 / count,
        top3=top3 / count,
        mrr=reciprocal / count,
        family_top1=family / count,
        chapter_top1=chapter / count,
        invalid_rate=invalid / (3 * count),
        missing_rate=missing / count,
    )


def format_report(metrics: Mapping[str, SliceMetrics]) -> str:
    lines = [
        f"{'slice':<6}{'n':>5}{'top1':>8}{'top3':>8}{'mrr':>8}{'fam@1':>8}{'chap@1':>8}"
        f"{'invalid':>9}{'missing':>9}"
    ]
    for name, item in metrics.items():
        lines.append(
            f"{name:<6}{item.count:>5}{item.top1:>8.3f}{item.top3:>8.3f}{item.mrr:>8.3f}"
            f"{item.family_top1:>8.3f}{item.chapter_top1:>8.3f}{item.invalid_rate:>9.3f}"
            f"{item.missing_rate:>9.3f}"
        )
    earned, available = official_points(metrics)
    lines.append(f"points {earned:.1f} / {available:.0f}")
    return "\n".join(lines)
