"""Run the full pipeline on a labelled set and report quality, cost and latency."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from icd_coder import index_store
from icd_coder.catalogue import Catalogue
from icd_coder.cli import offline_coder
from icd_coder.config import Settings, load_dotenv
from icd_coder.evaluation import Example, evaluate, format_report, load_examples
from icd_coder.guards import Budget, Deadline
from icd_coder.lexical import ScoredCode, catalogue_labels
from icd_coder.llm.factory import build_model_coder
from icd_coder.notes import Note, PredictionWriter
from icd_coder.pipeline import NoteUpgrader, run

LOG = logging.getLogger("evaluate")
MMS_PATH = Path("data/mms-2026-01.json")


class TimedUpgrader:
    """Records how long each note's upgrade took."""

    def __init__(self, inner: NoteUpgrader) -> None:
        self._inner = inner
        self.seconds: dict[object, float] = {}
        self._lock = threading.Lock()

    def code(self, note: Note, ranked: Sequence[ScoredCode]) -> Sequence[str]:
        started = time.monotonic()
        try:
            return self._inner.code(note, ranked)
        finally:
            with self._lock:
                self.seconds[note.id] = time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N notes")
    parser.add_argument("--no-reply-cache", action="store_true")
    parser.add_argument("--offline", action="store_true", help="skip the model")
    parser.add_argument("--rules", action="store_true", help="rank offline by rules only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv(Path(".env"))
    if not args.no_reply_cache:
        os.environ.setdefault("ICD_CODER_RESPONSE_CACHE", "cache/replies")
    if args.offline:
        os.environ["ICD_CODER_PROVIDER"] = "none"
    if args.rules:
        os.environ["ICD_CODER_MODEL_DIR"] = "none"
    settings = Settings.from_environment(os.environ)

    examples = load_examples(args.gold)
    if args.limit:
        examples = examples[: args.limit]
    notes = [Note(example.id, example.text) for example in examples]

    catalogue = Catalogue.load(settings.kb_dir, settings.languages)
    index = index_store.load_or_build(
        catalogue_labels(catalogue, settings.languages), settings.cache_dir
    )
    budget = Budget(settings.budget_usd)
    coder = build_model_coder(settings, catalogue, index, budget)
    timed = TimedUpgrader(coder) if coder is not None else None

    out_dir = Path("out") / args.name
    predictions_path = out_dir / "predictions.jsonl"
    with PredictionWriter(predictions_path) as writer:
        report = run(
            notes,
            offline_coder(settings, catalogue, index),
            catalogue,
            writer,
            upgrader=timed,
            deadline=Deadline(settings.deadline_seconds),
            concurrency=settings.concurrency,
            stop=threading.Event(),
        )

    predictions = _read_predictions(predictions_path)
    print(format_report(evaluate(examples, predictions, catalogue)))
    print(
        f"model upgrades {report.upgraded}/{report.notes}  spend ${budget.spent:.4f}"
        f"  (${budget.spent / max(len(notes), 1):.5f} per note)  wall {report.seconds:.1f} s"
    )
    if timed is not None and timed.seconds:
        latencies = np.array(list(timed.seconds.values()))
        print(
            f"latency per note p50 {np.percentile(latencies, 50):.1f} s"
            f"  p95 {np.percentile(latencies, 95):.1f} s"
        )
    _write_per_note(out_dir / "per-note.jsonl", examples, predictions, catalogue)
    _official_score(args.gold, predictions_path, args.limit)


def _read_predictions(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return {str(row["id"]): row["codes"] for row in rows}


def _write_per_note(
    path: Path, examples: Sequence[Example], predictions: dict[str, list[str]], catalogue: Catalogue
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            codes = predictions.get(example.id, [])
            record = {
                "id": example.id,
                "slice": example.slice,
                "style": example.style,
                "gold": sorted(example.gold),
                "gold_titles": [
                    catalogue[code].title for code in sorted(example.gold) if code in catalogue
                ],
                "codes": codes,
                "titles": [catalogue[code].title for code in codes if code in catalogue],
                "top1": bool(codes) and codes[0] in example.gold,
                "top3": bool(example.gold & set(codes[:3])),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _official_score(gold: Path, predictions: Path, limit: int) -> None:
    if limit:
        return
    result = subprocess.run(
        [
            sys.executable,
            "score.py",
            "--gold",
            str(gold),
            "--pred",
            str(predictions),
            "--mms",
            str(MMS_PATH),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print("official:", result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr)


if __name__ == "__main__":
    main()
