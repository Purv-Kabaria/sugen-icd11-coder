"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

from icd_coder import index_store
from icd_coder.catalogue import Catalogue
from icd_coder.config import Settings
from icd_coder.features import FeatureExtractor
from icd_coder.guards import Budget, Deadline
from icd_coder.lexical import LabelIndex, catalogue_labels
from icd_coder.llm.factory import build_model_coder
from icd_coder.notes import PredictionWriter, read_notes
from icd_coder.offline import OfflineCoder
from icd_coder.pipeline import run
from icd_coder.reranker import Reranker

LOG = logging.getLogger("icd_coder")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="icd-coder", description="Code consult notes in ICD-11.")
    parser.add_argument("notes", type=Path, help="input JSONL with id and text")
    parser.add_argument("predictions", type=Path, help="output JSONL with id and codes")
    args = parser.parse_args(argv)

    _configure_logging()
    settings = Settings.from_environment(os.environ)
    deadline = Deadline(settings.deadline_seconds)
    budget = Budget(settings.budget_usd)
    stop = _stop_on_termination()

    notes = read_notes(args.notes)
    LOG.info("read %d notes", len(notes))
    with PredictionWriter(args.predictions) as writer:
        try:
            catalogue, index = load_knowledge(settings)
            report = run(
                notes,
                offline_coder(settings, catalogue, index),
                catalogue,
                writer,
                upgrader=build_model_coder(settings, catalogue, index, budget),
                deadline=deadline,
                concurrency=settings.concurrency,
                stop=stop,
            )
        except Exception:
            LOG.exception("run failed; writing empty predictions for unanswered notes")
            for note in notes:
                writer.write(note.id, [])
            return 1
    LOG.info(
        "done: %d notes, %d upgraded by the model, $%.4f spent, %.1f s",
        report.notes,
        report.upgraded,
        budget.spent,
        report.seconds,
    )
    return 0


def prepare() -> int:
    _configure_logging()
    load_knowledge(Settings.from_environment(os.environ))
    return 0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def offline_coder(settings: Settings, catalogue: Catalogue, index: LabelIndex) -> OfflineCoder:
    """The offline coder, with the trained reranker when one ships in the model directory."""
    reranker = (
        Reranker.load(settings.model_dir, FeatureExtractor(catalogue, index))
        if settings.model_dir is not None
        else None
    )
    return OfflineCoder(catalogue, index, reranker)


def load_knowledge(settings: Settings) -> tuple[Catalogue, LabelIndex]:
    catalogue = Catalogue.load(settings.kb_dir, settings.languages)
    labels = catalogue_labels(catalogue, settings.languages)
    return catalogue, index_store.load_or_build(labels, settings.cache_dir)


def _stop_on_termination() -> threading.Event:
    stop = threading.Event()

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        LOG.warning("received signal %d; finishing with offline predictions", signum)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    return stop


if __name__ == "__main__":
    raise SystemExit(main())
