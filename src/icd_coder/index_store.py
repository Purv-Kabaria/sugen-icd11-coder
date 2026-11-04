"""Build the label index once and reuse it.

Tokenising 700k labels takes a minute; loading the stored index takes about a second.
The cache key covers the labels and the source of every module that shapes their
tokens, so editing the tokenizer or the knowledge base invalidates it automatically.
A missing, stale or unreadable cache is rebuilt, never trusted.
"""

from __future__ import annotations

import hashlib
import logging
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

from icd_coder import abbreviations, lexical, text
from icd_coder.lexical import Label, LabelIndex

LOG = logging.getLogger(__name__)
TOKENISER_MODULES = (text, lexical, abbreviations)


def fingerprint(labels: Sequence[Label]) -> str:
    digest = hashlib.sha256()
    for module in TOKENISER_MODULES:
        digest.update(Path(module.__file__ or "").read_bytes())
    for label in labels:
        digest.update(f"{label.code}\t{label.text}\n".encode())
    return digest.hexdigest()[:20]


def load_or_build(labels: Sequence[Label], cache_dir: Path | None) -> LabelIndex:
    path = cache_dir / f"labels-{fingerprint(labels)}.npz" if cache_dir else None
    if path is not None and path.exists():
        try:
            return LabelIndex.load(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            LOG.warning("index cache %s unreadable (%s); rebuilding", path, error)

    started = time.monotonic()
    index = LabelIndex.build(labels)
    LOG.info("built label index over %d labels in %.1f s", len(labels), time.monotonic() - started)
    if path is not None:
        try:
            index.save(path)
        except OSError as error:
            LOG.warning("could not store index cache at %s (%s)", path, error)
    return index
