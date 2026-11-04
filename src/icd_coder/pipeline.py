"""Code a batch of notes: offline first, then upgrade with a model while time and money last.

The order guarantees an answer for every note. Offline predictions exist before the
first model call, every note is written exactly once, and whatever the model has not
finished when the deadline passes, a stop is requested or the provider gives up is
written from the offline prediction.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Protocol

from icd_coder.catalogue import Catalogue
from icd_coder.guards import Deadline
from icd_coder.lexical import ScoredCode
from icd_coder.notes import Note, NoteId, PredictionWriter
from icd_coder.offline import MAX_CODES, OfflineCoder, specific_codes

LOG = logging.getLogger(__name__)
POLL_SECONDS = 1.0


class NoteUpgrader(Protocol):
    def code(self, note: Note, ranked: Sequence[ScoredCode]) -> Sequence[str]:
        """Codes for the note, best first. May raise; the caller falls back."""
        ...


@dataclass(frozen=True, slots=True)
class RunReport:
    notes: int
    upgraded: int
    seconds: float


def finalise(
    proposed: Iterable[str], fallback: Iterable[str], catalogue: Catalogue
) -> tuple[str, ...]:
    """Validated, specificity-resolved, distinct codes, topped up from the fallback."""
    codes: list[str] = []
    for raw in (*proposed, *fallback):
        code = catalogue.validate(raw)
        if code is None:
            continue
        code = catalogue.most_specific_default(code)
        if code not in codes:
            codes.append(code)
        if len(codes) == MAX_CODES:
            break
    return tuple(codes)


def run(
    notes: Sequence[Note],
    offline: OfflineCoder,
    catalogue: Catalogue,
    writer: PredictionWriter,
    *,
    upgrader: NoteUpgrader | None,
    deadline: Deadline,
    concurrency: int,
    stop: threading.Event,
) -> RunReport:
    started = time.monotonic()
    rankings = {note.id: offline.rank(note.text) for note in notes}
    fallbacks = {note_id: specific_codes(ranked, catalogue) for note_id, ranked in rankings.items()}
    LOG.info("offline predictions ready for %d notes", len(notes))

    upgraded = 0
    if upgrader is not None and notes:
        upgraded = _upgrade(
            notes,
            rankings,
            fallbacks,
            catalogue,
            writer=writer,
            upgrader=upgrader,
            deadline=deadline,
            concurrency=concurrency,
            stop=stop,
        )
    for note in notes:
        writer.write(note.id, fallbacks[note.id])
    return RunReport(len(notes), upgraded, time.monotonic() - started)


def _upgrade(
    notes: Sequence[Note],
    rankings: Mapping[NoteId, Sequence[ScoredCode]],
    fallbacks: Mapping[NoteId, Sequence[str]],
    catalogue: Catalogue,
    *,
    writer: PredictionWriter,
    upgrader: NoteUpgrader,
    deadline: Deadline,
    concurrency: int,
    stop: threading.Event,
) -> int:
    upgraded = 0
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="upgrade")
    pending: dict[Future[Sequence[str]], Note] = {
        pool.submit(upgrader.code, note, rankings[note.id]): note for note in notes
    }
    try:
        while pending and not deadline.expired() and not stop.is_set():
            timeout = min(POLL_SECONDS, deadline.remaining())
            done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            for future in done:
                note = pending.pop(future)
                proposed = _result_or_nothing(future, note.id)
                upgraded += bool(proposed)
                writer.write(note.id, finalise(proposed, fallbacks[note.id], catalogue))
        if pending:
            reason = "stop requested" if stop.is_set() else "deadline reached"
            LOG.warning(
                "%s with %d notes unfinished; using offline predictions", reason, len(pending)
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return upgraded


def _result_or_nothing(future: Future[Sequence[str]], note_id: NoteId) -> Sequence[str]:
    try:
        return future.result()
    except Exception as error:  # noqa: BLE001 - one failed note must never stop the run
        LOG.warning(
            "note %r: upgrade failed (%s); using offline prediction", note_id, type(error).__name__
        )
        return ()
