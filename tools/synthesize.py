"""Generate labelled synthetic consult notes for evaluation beyond the 60 dev notes.

    uv run python tools/synthesize.py --profile primary-care --count 300 --name synth-a
    uv run python tools/synthesize.py --profile all-chapters --count 300 --name synth-b

For each target code a generator model writes a note that supports exactly that code:
the distinguishing details of a specific code are documented, and for an unspecified
`.Z` code they are left out, as a real clinician would leave them out. A judge model
then checks that the target is the most specific code the note supports; notes that
fail are dropped. Dev notes are never shown to either model, so no dev content leaks
into the synthetic sets.

Targets come from two profiles:
- primary-care: the generator first lists conditions common in East African
  outpatient clinics, each is mapped to its ICD-11 family by the label index, and a
  leaf is drawn from that family (the dev-like distribution, slice A).
- all-chapters: chapters are drawn uniformly and a leaf uniformly within the chapter,
  across several languages and care settings (the out-of-distribution slice B).

Output: `eval/<name>/notes.jsonl` (all), `tune.jsonl` and `test.jsonl` (a fixed
50/50 split by id hash) and `MANIFEST.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from icd_coder import index_store
from icd_coder.catalogue import Catalogue, Category
from icd_coder.config import Settings, load_dotenv
from icd_coder.guards import Budget
from icd_coder.language import detect
from icd_coder.lexical import LabelIndex, catalogue_labels, query_from_text
from icd_coder.llm.providers import (
    AnthropicProvider,
    CachingProvider,
    ChatCompletionsProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
)

LOG = logging.getLogger("synthesize")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
SLICE_CHAPTERS = [f"{number:02d}" for number in range(1, 23)] + ["24", "25"]
OTHER_SPECIFIED_RATE = 0.05

STYLES = {
    "typed_consult": (
        "a typed outpatient consultation note of 6 to 12 lines with headers such as S:, O:, "
        "Ix:, A:, P: (or C/O, O/E, Imp, Dx, Plan), clinical abbreviations (c/o, x3/7, O/E, BD, "
        "TDS, PRN), vital signs, relevant investigations, the assessment and a plan with drugs "
        "and doses"
    ),
    "short_consult": (
        "a terse 2 to 4 line clinic note in telegraphic shorthand with abbreviations, key "
        "findings, an inline 'Dx' or 'Imp' and the treatment"
    ),
    "ambient_transcript": (
        "a transcript of the consultation recorded by an ambient scribe, as alternating lines "
        "'Doctor:' and 'Patient:' (or a parent), with numbers spoken in words, where the doctor "
        "explains the diagnosis in plain language near the end"
    ),
}
SETTINGS_BY_PROFILE = {
    "primary-care": ["a primary care clinic in Kenya", "a health centre in Uganda or Tanzania"],
    "all-chapters": [
        "a general outpatient clinic in Kenya",
        "a hospital outpatient department in India",
        "a specialist clinic in South Africa",
        "a family medicine practice in Brazil",
        "a clinic in Francophone West Africa",
        "a hospital outpatient clinic in Spain or Mexico",
        "a clinic in Egypt or the Gulf",
        "a general practice in the United Kingdom",
    ],
}
LANGUAGES_BY_PROFILE = {
    "primary-care": [("English", 1.0)],
    "all-chapters": [
        ("English", 0.40),
        ("French", 0.10),
        ("Spanish", 0.10),
        ("Portuguese", 0.08),
        ("Swahili", 0.10),
        ("Hindi", 0.10),
        ("Arabic", 0.07),
        ("Chinese", 0.05),
    ],
}

LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Swahili": "sw",
    "Hindi": "hi",
    "Arabic": "ar",
    "Chinese": "zh",
}

CONDITIONS_SYSTEM = (
    "You are a physician with long experience of outpatient medicine in many countries."
)
CONDITIONS_PROMPT = (
    "List {count} distinct conditions that clinicians diagnose and manage at outpatient "
    "primary care clinics in East Africa, covering infections, chronic diseases, maternal and "
    "child health, skin, eyes, ears, mouth, musculoskeletal, mental health, minor injuries and "
    "routine visits. Give each as a precise English clinical term with a relative frequency "
    "weight from 1 (rare) to 10 (very common)."
)
CHAPTER_PROMPT = (
    "List 20 distinct conditions classified in the ICD-11 chapter '{title}' that are seen as "
    "the main reason for a visit to an outpatient clinic, general or specialist, anywhere in "
    "the world. Give each as a precise English clinical term with a relative frequency weight "
    "from 1 (rare) to 10 (very common)."
)
CONDITIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}, "weight": {"type": "integer"}},
                "required": ["term", "weight"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["conditions"],
    "additionalProperties": False,
}

WRITER_SYSTEM = """\
You write realistic, varied clinical documentation for testing a medical coding system. \
Each note must support exactly one target ICD-11 category as the main condition of the \
visit.

Requirements:
- Write in the requested language, style and care setting, as a busy clinician would.
- Document the details that make the target the correct category and rule out its \
siblings. If the target is an unspecified category, do not document the details that \
would select a more specific sibling. If it is an "other specified" category, describe a \
form that none of the listed siblings covers.
- Never write the code, and do not copy the category title word for word: use the \
words a clinician would use, including abbreviations and local drug names.
- Add realistic noise: history, vital signs, unrelated findings, a comorbidity that is \
only mentioned, a differential that is excluded.
- If no outpatient clinic could plausibly see this condition as the main reason for a \
visit, set plausible to false and leave the note empty."""

WRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"plausible": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["plausible", "note"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """\
You are a senior ICD-11 coding auditor. Given a consultation note, a proposed code and \
the categories around it, decide whether the proposed code is the most specific ICD-11 \
MMS category that the note supports for the main condition of the visit. A parent is \
wrong when a child fits; an unspecified category is right only when the note lacks the \
distinguishing detail. Answer supported true or false, and when false name the problem \
in a few words."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "problem": {"type": "string"}},
    "required": ["supported", "problem"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Target:
    code: str
    style: str
    language: str
    setting: str


def family_context(category: Category, catalogue: Catalogue) -> str:
    """The target with its definition, parent and siblings, for the writer and the judge."""
    lines = [f"Target: {category.code} {category.title}"]
    if category.definition:
        lines.append(f"Definition: {category.definition}")
    if category.index_terms:
        lines.append(f"Also known as: {'; '.join(category.index_terms[:6])}")
    if category.parent is not None:
        parent = catalogue[category.parent]
        lines.append(f"Parent: {parent.code} {parent.title}")
        siblings = [
            child for child in catalogue.children(parent.code) if child.code != category.code
        ]
        lines.extend(f"Sibling: {sibling.code} {sibling.title}" for sibling in siblings[:15])
    lines.extend(
        f"Child: {child.code} {child.title}" for child in catalogue.children(category.code)
    )
    return "\n".join(lines)


def choose_leaf(family: Sequence[Category], rng: random.Random) -> Category | None:
    leaves = [category for category in family if category.is_leaf]
    if not leaves:
        return None
    usable = [leaf for leaf in leaves if not leaf.is_other_specified]
    if usable and rng.random() > OTHER_SPECIFIED_RATE:
        return rng.choice(usable)
    return rng.choice(leaves)


def condition_stems(
    provider: Provider,
    budget: Budget,
    catalogue: Catalogue,
    prompt: str,
    chapter: str | None,
) -> list[tuple[str, int]]:
    """Families of the conditions a model lists, weighted by how common it says they are.

    Each listed term is mapped to the best label-index match, restricted to the chapter
    when one is given, so the sample follows what clinics see rather than how many codes
    a chapter happens to contain.
    """
    reply = provider.complete_json(CONDITIONS_SYSTEM, prompt, CONDITIONS_SCHEMA, 8000)
    budget.charge(reply.cost_usd)
    index = _label_index(catalogue)
    weighted: list[tuple[str, int]] = []
    for item in reply.data.get("conditions", []):
        ranked = index.search(query_from_text(str(item.get("term", ""))), limit=20)
        matches = [
            scored.code
            for scored in ranked
            if chapter is None or catalogue[scored.code].chapter == chapter
        ]
        if matches:
            weight = item.get("weight", 1)
            weighted.append(
                (catalogue.stem(matches[0]), max(1, int(weight) if isinstance(weight, int) else 1))
            )
    return weighted


def sample_codes(
    weighted_by_group: Sequence[list[tuple[str, int]]],
    catalogue: Catalogue,
    count: int,
    rng: random.Random,
) -> list[str]:
    """Pick a group uniformly, a family by weight within it, then a leaf of that family."""
    groups = [group for group in weighted_by_group if group]
    codes: list[str] = []
    for _ in range(count * 3):
        group = rng.choice(groups)
        stem = rng.choices([stem for stem, _ in group], weights=[weight for _, weight in group])[0]
        leaf = choose_leaf(catalogue.family(stem), rng)
        if leaf is not None:
            codes.append(leaf.code)
        if len(codes) == count:
            break
    return codes


def chapter_titles(catalogue: Catalogue) -> dict[str, str]:
    return {category.chapter: category.ancestors[0] for category in catalogue if category.ancestors}


@lru_cache(maxsize=1)
def _label_index(catalogue: Catalogue) -> LabelIndex:
    settings = Settings.from_environment(os.environ)
    return index_store.load_or_build(catalogue_labels(catalogue), settings.cache_dir)


def make_targets(codes: Sequence[str], profile: str, rng: random.Random) -> list[Target]:
    languages, weights = zip(*LANGUAGES_BY_PROFILE[profile], strict=True)
    return [
        Target(
            code=code,
            style=rng.choice(sorted(STYLES)),
            language=rng.choices(languages, weights=weights)[0],
            setting=rng.choice(SETTINGS_BY_PROFILE[profile]),
        )
        for code in codes
    ]


def write_and_judge(
    target: Target, writer: Provider, judge: Provider | None, budget: Budget, catalogue: Catalogue
) -> dict[str, Any] | None:
    context = family_context(catalogue[target.code], catalogue)
    request = (
        f"{context}\n\nLanguage: {target.language}\nSetting: {target.setting}\n"
        f"Style: {STYLES[target.style]}"
    )
    try:
        written = writer.complete_json(WRITER_SYSTEM, request, WRITER_SCHEMA, 2500)
        budget.charge(written.cost_usd)
        note = str(written.data.get("note", "")).strip()
        if not written.data.get("plausible") or not note:
            return {"rejected": "implausible", "code": target.code}
        if detect(note) != LANGUAGE_CODES[target.language]:
            return {"rejected": f"not in {target.language}", "code": target.code}
        if judge is None:
            return {
                "text": note,
                "gold": [target.code],
                "style": target.style,
                "language": target.language,
            }
        verdict = judge.complete_json(
            JUDGE_SYSTEM,
            f"<note>\n{note}\n</note>\n\nProposed code and context:\n{context}",
            JUDGE_SCHEMA,
            1500,
        )
        budget.charge(verdict.cost_usd)
    except ProviderError as error:
        LOG.warning("target %s failed: %s", target.code, error)
        return None
    if not verdict.data.get("supported"):
        return {"rejected": str(verdict.data.get("problem", "")), "code": target.code}
    return {"text": note, "gold": [target.code], "style": target.style, "language": target.language}


def build_provider(spec: str, cache_dir: Path) -> Provider:
    """A provider from `kind:model[:effort]`, where kind is anthropic, openai or groq."""
    kind, model, *rest = spec.split(":")
    effort = rest[0] if rest else "low"
    provider: Provider
    if kind == "anthropic":
        provider = AnthropicProvider(model, effort=effort, timeout_seconds=180)
    elif kind == "openai":
        provider = OpenAIProvider(model, effort=effort, timeout_seconds=180)
    elif kind == "groq":
        reasoning = effort if model.startswith("openai/") else None
        provider = ChatCompletionsProvider(
            model,
            base_url=GROQ_BASE_URL,
            api_key=os.environ["GROQ_API_KEY"],
            timeout_seconds=180,
            effort=reasoning,
            max_retries=8,
        )
    else:
        raise ValueError(f"unknown provider kind {kind!r}")
    return CachingProvider(provider, cache_dir)


def split_name(note_id: str) -> str:
    return "tune" if int(hashlib.sha256(note_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(SETTINGS_BY_PROFILE), required=True)
    parser.add_argument("--count", type=int, required=True, help="targets to attempt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--slice", default=None, help="slice label; default A or B by profile")
    parser.add_argument("--planner", default="groq:openai/gpt-oss-120b:low")
    parser.add_argument("--writer", default="groq:qwen/qwen3.8-27b")
    parser.add_argument("--judge", default="groq:openai/gpt-oss-120b:low", help="or none")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(Path(".env"))
    rng = random.Random(args.seed)
    cache_dir = Path("cache/synthesis")
    writer = build_provider(args.writer, cache_dir)
    judge = build_provider(args.judge, cache_dir) if args.judge != "none" else None
    budget = Budget(args.budget)
    settings = Settings.from_environment(os.environ)
    catalogue = Catalogue.load(settings.kb_dir)

    planner = build_provider(args.planner, cache_dir)
    if args.profile == "primary-care":
        prompt = CONDITIONS_PROMPT.format(count=150)
        groups = [condition_stems(planner, budget, catalogue, prompt, None)]
    else:
        titles = chapter_titles(catalogue)
        groups = [
            condition_stems(
                planner, budget, catalogue, CHAPTER_PROMPT.format(title=titles[chapter]), chapter
            )
            for chapter in SLICE_CHAPTERS
        ]
    codes = sample_codes(groups, catalogue, args.count, rng)
    targets = make_targets(codes, args.profile, rng)

    lock = threading.Lock()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def attempt(target: Target) -> None:
        if not budget.can_afford(0.05):
            return
        result = write_and_judge(target, writer, judge, budget, catalogue)
        if result is None:
            return
        with lock:
            (rejected if "rejected" in result else accepted).append(result)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(attempt, targets))

    slice_name = args.slice or ("A" if args.profile == "primary-care" else "B")
    out_dir = Path("eval") / args.name
    rows = _write_outputs(out_dir, args.name, slice_name, accepted, rejected)
    _write_manifest(
        out_dir,
        args=args,
        attempted=len(targets),
        rows=rows,
        rejected=rejected,
        spent=budget.spent,
    )
    LOG.info("accepted %d, rejected %d, spent $%.2f", len(rows), len(rejected), budget.spent)


def _write_outputs(
    out_dir: Path,
    name: str,
    slice_name: str,
    accepted: Sequence[dict[str, Any]],
    rejected: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Numbered rows in a stable order, split by id hash, plus the rejections."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(accepted, key=lambda item: (item["gold"][0], item["text"]))
    rows = [
        {"id": f"{name}-{number:04d}", "slice": slice_name, **record}
        for number, record in enumerate(ordered)
    ]
    outputs = {
        "notes.jsonl": rows,
        "tune.jsonl": [row for row in rows if split_name(row["id"]) == "tune"],
        "test.jsonl": [row for row in rows if split_name(row["id"]) == "test"],
        "rejected.jsonl": list(rejected),
    }
    for file_name, records in outputs.items():
        with (out_dir / file_name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return rows


def _write_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    attempted: int,
    rows: Sequence[dict[str, Any]],
    rejected: Sequence[dict[str, Any]],
    spent: float,
) -> None:
    prompt_hash = hashlib.sha256(
        (
            WRITER_SYSTEM + JUDGE_SYSTEM + CONDITIONS_PROMPT + CHAPTER_PROMPT + json.dumps(STYLES)
        ).encode()
    ).hexdigest()[:12]
    languages: dict[str, int] = {}
    for row in rows:
        languages[row["language"]] = languages.get(row["language"], 0) + 1
    reasons: dict[str, int] = {}
    for item in rejected:
        own_reason = item["rejected"] == "implausible" or item["rejected"].startswith("not in")
        reason = item["rejected"] if own_reason else "judge"
        reasons[reason] = reasons.get(reason, 0) + 1
    lines = [
        f"# {args.name}",
        "",
        f"- Created: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `tools/synthesize.py`",
        f"- Profile: {args.profile}; seed {args.seed}; targets attempted {attempted}",
        f"- Planner: `{args.planner}`; writer: `{args.writer}`; judge: `{args.judge}`",
        f"- Prompt hash `{prompt_hash}`",
        f"- Accepted {len(rows)}; rejected by writer or judge {len(rejected)}; spent ${spent:.2f}",
        f"- Languages: {', '.join(f'{name} {count}' for name, count in sorted(languages.items()))}",
        f"- Rejections: {', '.join(f'{name} {count}' for name, count in sorted(reasons.items()))}"
        " (details in `rejected.jsonl`)",
        "- Split: `tune.jsonl` and `test.jsonl` by a hash of the note id. Tune on `tune` only.",
        "- No dev note was shown to any model.",
    ]
    (out_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
