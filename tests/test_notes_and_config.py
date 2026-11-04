from __future__ import annotations

import json
from pathlib import Path

import pytest

from icd_coder.config import Settings
from icd_coder.guards import Budget, Deadline, FailureStreak
from icd_coder.notes import Note, PredictionWriter, read_notes


def test_read_notes_tolerates_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "notes.jsonl"
    path.write_bytes(
        "﻿".encode()
        + b'{"id": "n1", "text": "fever"}\n'
        + b"\n"
        + b"not json\n"
        + b'{"text": "no id"}\n'
        + b'{"id": true, "text": "boolean id"}\n'
        + b'{"id": 7, "text": null}\n'
        + b'{"id": "n1", "text": "duplicate"}\n'
        + b'{"id": "n2", "text": "homa kali"}\n'
    )
    assert read_notes(path) == [Note("n1", "fever"), Note(7, ""), Note("n2", "homa kali")]


def test_writer_keeps_the_first_line_per_id(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    with PredictionWriter(path) as writer:
        assert writer.write("n1", ["1F40.Z"])
        assert not writer.write("n1", ["5A11"])
        assert writer.write(2, [])
        assert writer.has_written("n1")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"id": "n1", "codes": ["1F40.Z"]}, {"id": 2, "codes": []}]


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, None),
        ({"ANTHROPIC_API_KEY": "k"}, "anthropic"),
        ({"OPENAI_API_KEY": "k"}, "openai"),
        ({"ANTHROPIC_API_KEY": " ", "OPENAI_API_KEY": "k"}, "openai"),
        ({"ANTHROPIC_API_KEY": "k", "ICD_CODER_PROVIDER": "none"}, None),
        ({"ANTHROPIC_API_KEY": "k", "ICD_CODER_PROVIDER": "openai"}, None),
        (
            {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k", "ICD_CODER_PROVIDER": "OpenAI"},
            "openai",
        ),
    ],
)
def test_provider_choice(environ: dict[str, str], expected: str | None) -> None:
    assert Settings.from_environment(environ).chosen_provider() == expected


def test_guards() -> None:
    now = [100.0]
    deadline = Deadline(10, clock=lambda: now[0])
    assert deadline.remaining() == 10
    now[0] = 111
    assert deadline.expired()

    budget = Budget(1.0)
    assert budget.can_afford(0.6)
    budget.charge(0.6)
    assert not budget.can_afford(0.5)
    assert budget.spent == pytest.approx(0.6)

    streak = FailureStreak(2)
    streak.record(succeeded=False)
    streak.record(succeeded=True)
    streak.record(succeeded=False)
    assert not streak.tripped()
    streak.record(succeeded=False)
    assert streak.tripped()


def test_model_directory_defaults_and_can_be_disabled() -> None:
    assert Settings.from_environment({}).model_dir == Path("data/models")
    assert Settings.from_environment({"ICD_CODER_MODEL_DIR": "none"}).model_dir is None
