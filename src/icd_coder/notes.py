"""Reading consult notes and writing predictions as JSON lines."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Self

LOG = logging.getLogger(__name__)

type NoteId = str | int


@dataclass(frozen=True, slots=True)
class Note:
    id: NoteId
    text: str


def read_notes(path: Path) -> list[Note]:
    """Every note with a usable id, in file order.

    A line that is not JSON or has no string or integer id cannot be answered and is
    skipped with a warning. A repeated id keeps its first line, as the scorer does.
    """
    notes: list[Note] = []
    seen: set[NoteId] = set()
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("input line %d is not JSON; skipped", number)
                continue
            note_id = row.get("id") if isinstance(row, dict) else None
            if isinstance(note_id, bool) or not isinstance(note_id, str | int):
                LOG.warning("input line %d has no usable id; skipped", number)
                continue
            if note_id in seen:
                LOG.warning("input line %d repeats id %r; skipped", number, note_id)
                continue
            seen.add(note_id)
            text = row.get("text")
            notes.append(Note(note_id, text if isinstance(text, str) else ""))
    return notes


class PredictionWriter:
    """Writes one line per note and flushes it at once, so a killed run keeps finished work.

    The scorer keeps the first line for an id, so a second write for the same id is
    refused rather than appended.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = path.open("w", encoding="utf-8", newline="\n")
        self._written: set[NoteId] = set()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def has_written(self, note_id: NoteId) -> bool:
        return note_id in self._written

    def write(self, note_id: NoteId, codes: Sequence[str]) -> bool:
        if note_id in self._written:
            return False
        line = json.dumps({"id": note_id, "codes": list(codes)}, ensure_ascii=False)
        self._handle.write(line + "\n")
        self._handle.flush()
        self._written.add(note_id)
        return True

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
