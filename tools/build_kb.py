"""Build the coder's knowledge base from the MMS code list and the WHO crawls.

    uv run python tools/build_kb.py --language fr --language es

Reads `data/mms-2026-01.json` (the scoreable codes) and the crawls written by
`tools/crawl_who_mms.py` under `cache/who/`. Writes one line per scoreable category
to `data/kb/mms-2026-01.en.jsonl.gz`, and for each extra language a file of titles,
index terms and inclusions to `data/kb/mms-2026-01.<language>.jsonl.gz`. Languages
live in separate files so that rebuilding the English core does not grow the git
history by the size of every translation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger("build_kb")
RELEASE = "2026-01"
MMS_URI_PREFIX = "http://id.who.int/icd/release/11/mms/"


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl_gz(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            compressed.write(line.encode("utf-8"))


def ancestor_titles(entity_id: str, entities: dict[str, dict[str, Any]]) -> list[str]:
    """Titles of the enclosing chapter and blocks, outermost first."""
    titles: list[str] = []
    parents = entities[entity_id]["parents"]
    while parents and parents[0] in entities:
        parent = entities[parents[0]]
        if parent["class_kind"] in {"chapter", "block"}:
            titles.append(parent["title"])
        parents = parent["parents"]
    return list(reversed(titles))


def parent_code(entity_id: str, entities: dict[str, dict[str, Any]]) -> str | None:
    parents = entities[entity_id]["parents"]
    if not parents or parents[0] not in entities:
        return None
    parent = entities[parents[0]]
    return parent["code"] if parent["class_kind"] == "category" else None


def resolve_references(
    references: list[dict[str, Any]], entities: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    resolved = []
    for reference in references:
        target = entities.get(reference["ref"] or "")
        resolved.append({"label": reference["label"], "code": target["code"] if target else None})
    return resolved


def load_crawl(crawl_dir: Path, language: str) -> dict[str, dict[str, Any]]:
    path = crawl_dir / f"mms-{RELEASE}.{language}.jsonl.gz"
    return {entity["id"]: entity for entity in read_jsonl_gz(path)}


def load_mms(mms_path: Path) -> list[dict[str, Any]]:
    with mms_path.open(encoding="utf-8") as handle:
        categories: list[dict[str, Any]] = json.load(handle)
    return categories


def build_core(
    mms: list[dict[str, Any]], english: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    for category in mms:
        entity_id = category["uri"].removeprefix(MMS_URI_PREFIX)
        entity = english[entity_id]
        if entity["code"] != category["code"]:
            raise ValueError(
                f"code mismatch for {entity_id}: {entity['code']} != {category['code']}"
            )
        records.append(
            {
                "code": category["code"],
                "title": category["title"],
                "chapter": category["chapter"],
                "leaf": category["leaf"],
                "residual": category["residual"],
                "depth": category["depth"],
                "parent": parent_code(entity_id, english),
                "ancestors": ancestor_titles(entity_id, english),
                "definition": entity["definition"],
                "coding_note": entity["coding_note"],
                "index_terms": entity["index_terms"],
                "inclusions": entity["inclusions"],
                "exclusions": resolve_references(entity["exclusions"], english),
            }
        )
    return sorted(records, key=lambda record: record["code"])


def build_translation(
    mms: list[dict[str, Any]], crawl: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    for category in mms:
        entity = crawl.get(category["uri"].removeprefix(MMS_URI_PREFIX))
        if entity and entity["title"]:
            terms = list(dict.fromkeys(entity["index_terms"] + entity["inclusions"]))
            records.append({"code": category["code"], "title": entity["title"], "terms": terms})
    return sorted(records, key=lambda record: record["code"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mms", type=Path, default=Path("data/mms-2026-01.json"))
    parser.add_argument("--crawl-dir", type=Path, default=Path("cache/who"))
    parser.add_argument("--language", action="append", default=[], help="extra WHO language")
    parser.add_argument("--out-dir", type=Path, default=Path("data/kb"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mms = load_mms(args.mms)
    outputs = {"en": build_core(mms, load_crawl(args.crawl_dir, "en"))}
    for language in args.language:
        outputs[language] = build_translation(mms, load_crawl(args.crawl_dir, language))
    for language, records in outputs.items():
        path = args.out_dir / f"mms-{RELEASE}.{language}.jsonl.gz"
        write_jsonl_gz(records, path)
        LOG.info("wrote %d records to %s (%d bytes)", len(records), path, path.stat().st_size)


if __name__ == "__main__":
    main()
