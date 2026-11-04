"""Train the LightGBM reranker and compare it with the rule-based ranking."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm
import numpy as np
import numpy.typing as npt

from icd_coder import clinical_text, features, language, lexical, offline, text
from icd_coder.catalogue import Catalogue
from icd_coder.cli import load_knowledge
from icd_coder.config import Settings
from icd_coder.evaluation import Example, evaluate, format_report, load_examples
from icd_coder.features import CATEGORICAL_FEATURES, FEATURES, FeatureExtractor
from icd_coder.offline import MAX_CODES, OfflineCoder
from icd_coder.reranker import MODEL_GLOB, ensemble_scores

LOG = logging.getLogger("train_reranker")
DEV_NOTES = Path("data/dev_notes.jsonl")
FEATURE_CACHE = Path("cache/features")
MODEL_DIR = Path("data/models")
FEATURE_MODULES = (clinical_text, features, language, lexical, offline, text)
FOLDS = 5
MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 100
ENSEMBLE_SEEDS = (11, 23, 37, 51, 73)
BOOTSTRAP_SAMPLES = 2000
PARAMETERS: dict[str, object] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [1],
    "first_metric_only": True,
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

FloatMatrix = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class NoteCandidates:
    example: Example
    codes: tuple[str, ...]
    features: FloatMatrix
    rules: tuple[str, ...]

    @property
    def labels(self) -> npt.NDArray[np.int32]:
        return np.array([code in self.example.gold for code in self.codes], dtype=np.int32)

    def ranked(self, scores: npt.NDArray[np.float64]) -> tuple[str, ...]:
        """The top codes by score, padded from the rules exactly as `OfflineCoder` does."""
        order = np.argsort(-scores, kind="stable")[:MAX_CODES]
        learned = [self.codes[index] for index in order]
        return tuple(dict.fromkeys([*learned, *self.rules]))[:MAX_CODES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--compare", type=Path, nargs="*", default=[])
    parser.add_argument("--no-save", action="store_true", help="report without writing models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(path.resolve() == DEV_NOTES.resolve() for path in [*args.train, *args.compare]):
        parser.error("dev notes are a regression check, not training or selection data")

    catalogue, index = load_knowledge(Settings.from_environment(os.environ))
    extractor = FeatureExtractor(catalogue, index)
    rules = OfflineCoder(catalogue, index)
    training = [note for path in args.train for note in load_candidates(path, extractor, rules)]
    LOG.info("training on %d notes, %d candidates", len(training), _candidate_count(training))

    rounds = cross_validate(training, catalogue)
    models = [fit(training, rounds, seed) for seed in ENSEMBLE_SEEDS]
    for path in args.compare:
        compare(path.name, load_candidates(path, extractor, rules), models, catalogue)
    if not args.no_save:
        save(models)


def load_candidates(
    path: Path, extractor: FeatureExtractor, rules: OfflineCoder
) -> list[NoteCandidates]:
    """Candidates, features and the rule-based prediction for every note in a file."""
    examples = load_examples(path)
    cache = FEATURE_CACHE / f"{path.parent.name}-{path.stem}-{_cache_key(path)}.npz"
    if cache.exists():
        return _read_cache(cache, examples)

    started = time.monotonic()
    notes = []
    for example in examples:
        candidates = extractor.extract(example.text)
        notes.append(
            NoteCandidates(
                example, candidates.codes, candidates.features, rules.predict(example.text)
            )
        )
    LOG.info("extracted %s in %.1f s", path, time.monotonic() - started)
    _write_cache(cache, notes)
    return notes


def cross_validate(notes: Sequence[NoteCandidates], catalogue: Catalogue) -> int:
    """Out-of-fold comparison with the rules; returns the boosting rounds to train with."""
    groups = [catalogue.stem(min(note.example.gold)) for note in notes]
    predictions: dict[str, tuple[str, ...]] = {}
    best_rounds = []
    for fold, (train_rows, valid_rows) in enumerate(grouped_folds(groups, FOLDS), 1):
        train_notes = [notes[row] for row in train_rows]
        valid_notes = [notes[row] for row in valid_rows]
        model = lightgbm.train(
            {**PARAMETERS, "seed": fold},
            _dataset(train_notes),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[_dataset(valid_notes)],
            callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_rounds.append(model.best_iteration)
        for note in valid_notes:
            predictions[note.example.id] = note.ranked(ensemble_scores([model], note.features))
    rounds = int(np.median(best_rounds))
    LOG.info("best rounds per fold %s; training with %d", best_rounds, rounds)
    _report("cross-validation (out of fold)", notes, predictions, catalogue)
    return rounds


def grouped_folds(groups: Sequence[str], folds: int) -> list[tuple[list[int], list[int]]]:
    """Train and validation rows per fold, with every group wholly inside one fold."""
    rows_by_group: dict[str, list[int]] = {}
    for row, group in enumerate(groups):
        rows_by_group.setdefault(group, []).append(row)
    assigned: list[list[int]] = [[] for _ in range(folds)]
    for rows in sorted(rows_by_group.values(), key=len, reverse=True):
        min(assigned, key=len).extend(rows)
    everything = set(range(len(groups)))
    return [(sorted(everything - set(valid)), sorted(valid)) for valid in assigned]


def fit(notes: Sequence[NoteCandidates], rounds: int, seed: int) -> lightgbm.Booster:
    return lightgbm.train({**PARAMETERS, "seed": seed}, _dataset(notes), num_boost_round=rounds)


def compare(
    name: str,
    notes: Sequence[NoteCandidates],
    models: Sequence[lightgbm.Booster],
    catalogue: Catalogue,
) -> None:
    predictions = {
        note.example.id: note.ranked(ensemble_scores(models, note.features)) for note in notes
    }
    _report(f"held-out {name}", notes, predictions, catalogue)


def save(models: Sequence[lightgbm.Booster]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in MODEL_DIR.glob(MODEL_GLOB):
        stale.unlink()
    for number, model in enumerate(models, 1):
        model.save_model(str(MODEL_DIR / f"reranker-{number}.txt"))
    LOG.info("saved %d models to %s", len(models), MODEL_DIR)


def _report(
    title: str,
    notes: Sequence[NoteCandidates],
    reranked: Mapping[str, Sequence[str]],
    catalogue: Catalogue,
) -> None:
    examples = [note.example for note in notes]
    rules = {note.example.id: note.rules for note in notes}
    recall = np.mean([bool(note.example.gold & set(note.codes)) for note in notes])
    low, high = _top1_gain_interval(notes, reranked)
    print(f"\n== {title}: {len(notes)} notes, candidate recall {recall:.3f}")
    print("rules\n" + format_report(evaluate(examples, rules, catalogue)))
    print("reranker\n" + format_report(evaluate(examples, reranked, catalogue)))
    print(f"top-1 gain over rules, 95% bootstrap interval: [{low:+.3f}, {high:+.3f}]")


def _top1_gain_interval(
    notes: Sequence[NoteCandidates], reranked: Mapping[str, Sequence[str]]
) -> tuple[float, float]:
    """Paired bootstrap of the reranker's top-1 accuracy minus the rules'."""

    def hit(codes: Sequence[str], gold: frozenset[str]) -> float:
        return float(bool(codes) and codes[0] in gold)

    gains = np.array(
        [
            hit(reranked.get(note.example.id, ()), note.example.gold)
            - hit(note.rules, note.example.gold)
            for note in notes
        ]
    )
    rng = np.random.default_rng(0)
    samples = rng.choice(gains, size=(BOOTSTRAP_SAMPLES, len(gains))).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _dataset(notes: Sequence[NoteCandidates]) -> lightgbm.Dataset:
    """One query group per note; notes without candidates carry no ranking signal."""
    usable = [note for note in notes if note.codes]
    return lightgbm.Dataset(
        np.vstack([note.features for note in usable]),
        label=np.concatenate([note.labels for note in usable]),
        group=[len(note.codes) for note in usable],
        feature_name=list(FEATURES),
        categorical_feature=list(CATEGORICAL_FEATURES),
        free_raw_data=False,
    )


def _candidate_count(notes: Sequence[NoteCandidates]) -> int:
    return sum(len(note.codes) for note in notes)


def _cache_key(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    for module in FEATURE_MODULES:
        digest.update(Path(module.__file__ or "").read_bytes())
    for knowledge in sorted(Path("data/kb").glob("*.gz")):
        digest.update(knowledge.read_bytes())
    return digest.hexdigest()[:16]


def _write_cache(path: Path, notes: Sequence[NoteCandidates]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        ids=np.array([note.example.id for note in notes]),
        sizes=np.array([len(note.codes) for note in notes], dtype=np.int64),
        codes=np.array([code for note in notes for code in note.codes], dtype=str),
        features=np.vstack([note.features for note in notes]).astype(np.float32),
        rules=np.array([[*note.rules, *[""] * (MAX_CODES - len(note.rules))] for note in notes]),
    )
    partial.replace(path)


def _read_cache(path: Path, examples: Sequence[Example]) -> list[NoteCandidates]:
    with np.load(path, allow_pickle=False) as stored:
        if stored["ids"].tolist() != [example.id for example in examples]:
            msg = f"feature cache {path} does not match its notes"
            raise ValueError(msg)
        sizes, codes = stored["sizes"], stored["codes"].tolist()
        matrix, rules = stored["features"], stored["rules"].tolist()
    starts = np.concatenate([[0], np.cumsum(sizes)])
    return [
        NoteCandidates(
            example,
            tuple(codes[starts[row] : starts[row + 1]]),
            matrix[starts[row] : starts[row + 1]],
            tuple(code for code in rules[row] if code),
        )
        for row, example in enumerate(examples)
    ]


if __name__ == "__main__":
    main()
