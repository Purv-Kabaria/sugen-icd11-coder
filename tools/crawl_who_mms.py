"""Crawl the ICD-11 MMS linearization from a local WHO ICD-API container."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LOG = logging.getLogger("crawl_who_mms")
MAX_ATTEMPTS = 3


def fetch_entity(url: str, language: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"API-Version": "v2", "Accept-Language": language, "Accept": "application/json"},
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload: dict[str, Any] = json.load(response)
                return payload
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == MAX_ATTEMPTS:
                raise
            LOG.warning("retrying %s after %s", url, error)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def entity_id(uri: str, release_prefix: str) -> str:
    """Path of a linearization uri relative to the MMS root, for example `1435254666/other`."""
    return uri.removeprefix(release_prefix).lstrip("/")


def text_of(value: dict[str, Any] | None) -> str | None:
    return value["@value"] if value else None


def labels_of(items: Iterable[dict[str, Any]] | None) -> list[str]:
    return [item["label"]["@value"] for item in items or []]


def references_of(
    items: Iterable[dict[str, Any]] | None, release_prefix: str
) -> list[dict[str, Any]]:
    return [
        {
            "label": item["label"]["@value"],
            "ref": entity_id(item["linearizationReference"], release_prefix)
            if "linearizationReference" in item
            else None,
        }
        for item in items or []
    ]


def to_record(entity: dict[str, Any], release_prefix: str) -> dict[str, Any]:
    return {
        "id": entity_id(entity["@id"], release_prefix),
        "code": entity.get("code") or None,
        "class_kind": entity.get("classKind"),
        "title": text_of(entity.get("title")),
        "parents": [entity_id(uri, release_prefix) for uri in entity.get("parent", [])],
        "children": [entity_id(uri, release_prefix) for uri in entity.get("child", [])],
        "definition": text_of(entity.get("definition")),
        "long_definition": text_of(entity.get("longDefinition")),
        "fully_specified_name": text_of(entity.get("fullySpecifiedName")),
        "coding_note": text_of(entity.get("codingNote")),
        "index_terms": labels_of(entity.get("indexTerm")),
        "inclusions": labels_of(entity.get("inclusion")),
        "exclusions": references_of(entity.get("exclusion"), release_prefix),
        "child_elsewhere": references_of(entity.get("foundationChildElsewhere"), release_prefix),
    }


def crawl(
    base_url: str, release: str, language: str, workers: int, excluded_chapters: set[str]
) -> list[dict[str, Any]]:
    root = f"{base_url}/icd/release/11/{release}/mms"
    release_prefix = f"http://id.who.int/icd/release/11/{release}/mms"
    root_entity = fetch_entity(root, language)
    frontier = [entity_id(uri, release_prefix) for uri in root_entity["child"]]
    seen = set(frontier)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while frontier:
            entities = pool.map(lambda node: fetch_entity(f"{root}/{node}", language), frontier)
            next_frontier: list[str] = []
            for entity in entities:
                record = to_record(entity, release_prefix)
                if record["class_kind"] == "chapter" and record["code"] in excluded_chapters:
                    continue
                records.append(record)
                for child in record["children"]:
                    if child not in seen:
                        seen.add(child)
                        next_frontier.append(child)
            LOG.info("level done: %d entities so far", len(records))
            frontier = next_frontier
    return sorted(records, key=lambda record: record["id"])


def write_jsonl_gz(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
    ):
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            compressed.write(line.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8382")
    parser.add_argument("--release", default="2026-01")
    parser.add_argument("--language", default="en")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--exclude-chapter",
        action="append",
        default=None,
        help="chapter code to skip; defaults to V and X, which hold no scoreable codes",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    excluded = set(args.exclude_chapter or ["V", "X"])
    records = crawl(args.base_url, args.release, args.language, args.workers, excluded)
    write_jsonl_gz(records, args.out)
    LOG.info("wrote %d entities to %s", len(records), args.out)


if __name__ == "__main__":
    main()
