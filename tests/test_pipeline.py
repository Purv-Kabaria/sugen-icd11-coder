from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from icd_coder.catalogue import Catalogue
from icd_coder.guards import Deadline
from icd_coder.lexical import LabelIndex, ScoredCode, catalogue_labels
from icd_coder.notes import Note, PredictionWriter
from icd_coder.offline import OfflineCoder
from icd_coder.pipeline import finalise, run

NOTES = [
    Note("a", "A: falciparum malaria."),
    Note("b", "Dx: mumps."),
    Note(3, "Imp: type 2 diabetes mellitus."),
]


class ScriptedUpgrader:
    def __init__(
        self, answers: dict[object, Sequence[str] | Exception], delay: float = 0.0
    ) -> None:
        self._answers = answers
        self._delay = delay

    def code(self, note: Note, ranked: Sequence[ScoredCode]) -> Sequence[str]:
        time.sleep(self._delay)
        answer = self._answers.get(note.id, ())
        if isinstance(answer, Exception):
            raise answer
        return answer


def run_pipeline(
    catalogue: Catalogue,
    path: Path,
    upgrader: ScriptedUpgrader | None,
    deadline_seconds: float = 5.0,
    stop: threading.Event | None = None,
) -> dict[object, list[str]]:
    offline = OfflineCoder(catalogue, LabelIndex.build(catalogue_labels(catalogue)))
    with PredictionWriter(path) as writer:
        run(
            NOTES,
            offline,
            catalogue,
            writer,
            upgrader=upgrader,
            deadline=Deadline(deadline_seconds),
            concurrency=2,
            stop=stop or threading.Event(),
        )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len({line["id"] for line in lines}) == len(NOTES)
    return {line["id"]: line["codes"] for line in lines}


def test_offline_only_answers_every_note(catalogue: Catalogue, tmp_path: Path) -> None:
    result = run_pipeline(catalogue, tmp_path / "p.jsonl", None)
    assert result["a"][0] == "1F40.Z"
    assert result["b"][0] == "1D80.0"
    assert result[3][0] == "5A11"


def test_model_answers_are_validated_resolved_and_padded(
    catalogue: Catalogue, tmp_path: Path
) -> None:
    upgrader = ScriptedUpgrader({"a": ["XX99", "1f40.0&XS25", "1F40"], "b": ["1D80"]})
    result = run_pipeline(catalogue, tmp_path / "p.jsonl", upgrader)
    assert result["a"][:2] == ["1F40.0", "1F40.Z"]
    assert result["b"][0] == "1D80.0"


def test_a_failing_note_falls_back_without_stopping_the_run(
    catalogue: Catalogue, tmp_path: Path
) -> None:
    upgrader = ScriptedUpgrader({"a": RuntimeError("provider down"), "b": ["1D80.1"]})
    result = run_pipeline(catalogue, tmp_path / "p.jsonl", upgrader)
    assert result["a"][0] == "1F40.Z"
    assert result["b"][0] == "1D80.1"


def test_deadline_writes_offline_predictions_for_unfinished_notes(
    catalogue: Catalogue, tmp_path: Path
) -> None:
    upgrader = ScriptedUpgrader({"a": ["1F40.0"], "b": ["1D80.1"], 3: ["5A11"]}, delay=2.0)
    started = time.monotonic()
    result = run_pipeline(catalogue, tmp_path / "p.jsonl", upgrader, deadline_seconds=0.3)
    assert time.monotonic() - started < 1.5
    assert result["a"][0] == "1F40.Z"


def test_stop_request_writes_offline_predictions(catalogue: Catalogue, tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    upgrader = ScriptedUpgrader({"a": ["1F40.0"]}, delay=1.0)
    result = run_pipeline(catalogue, tmp_path / "p.jsonl", upgrader, stop=stop)
    assert result["a"][0] == "1F40.Z"


@pytest.mark.parametrize(
    ("proposed", "fallback", "expected"),
    [
        (["1F40.0", "1F40.0"], ["5A11"], ("1F40.0", "5A11")),
        ([], ["1D80"], ("1D80.0",)),
        (["bad", "", "5A11", "1F40.Y", "1F40.Z"], ["1D80.0"], ("5A11", "1F40.Y", "1F40.Z")),
    ],
)
def test_finalise(
    catalogue: Catalogue, proposed: list[str], fallback: list[str], expected: tuple[str, ...]
) -> None:
    assert finalise(proposed, fallback, catalogue) == expected
