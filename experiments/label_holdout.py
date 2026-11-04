"""Retrieval benchmark that never touches the dev notes.

Hold out a random sample of WHO index terms, rebuild the label index without them,
then query each held-out term and check where its own code ranks. This measures how
well the lexical index generalises to unseen phrasings of a diagnosis.

    uv run python experiments/label_holdout.py --sample 3000
    uv run python experiments/label_holdout.py --language fr --powers 1

With `--language`, the held-out terms are WHO translations in that language, and each
power is scored twice: searching English labels only, and English plus that language.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Collection, Sequence
from pathlib import Path

from icd_coder.catalogue import Catalogue
from icd_coder.lexical import Label, LabelIndex, catalogue_labels, query_from_text
from icd_coder.text import tokens

KB_DIR = Path("data/kb")


def held_out_sample(
    labels: Sequence[Label], catalogue: Catalogue, language: str, size: int, seed: int
) -> list[Label]:
    """Terms in the language that differ from their category's English title."""
    candidates = [
        label
        for label in labels
        if label.language == language
        and tokens(label.text)
        and set(tokens(label.text)) != set(tokens(catalogue[label.code].title))
    ]
    return random.Random(seed).sample(candidates, min(size, len(candidates)))


def evaluate(
    index: LabelIndex,
    queries: Sequence[Label],
    catalogue: Catalogue,
    languages: Collection[str] | None,
) -> dict[str, float]:
    hits = {"top1": 0, "top3": 0, "top10": 0, "top50": 0, "family_top1": 0}
    for query in queries:
        found = index.search(query_from_text(query.text), limit=50, languages=languages)
        ranked = [scored.code for scored in found]
        if ranked and catalogue.stem(ranked[0]) == catalogue.stem(query.code):
            hits["family_top1"] += 1
        for name, cutoff in (("top1", 1), ("top3", 3), ("top10", 10), ("top50", 50)):
            hits[name] += query.code in ranked[:cutoff]
    return {name: count / len(queries) for name, count in hits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--language", default="en")
    parser.add_argument("--powers", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    args = parser.parse_args()

    extra = [] if args.language == "en" else [args.language]
    catalogue = Catalogue.load(KB_DIR, extra)
    labels = catalogue_labels(catalogue, extra)
    held_out = held_out_sample(labels, catalogue, args.language, args.sample, args.seed)
    held_out_set = set(held_out)
    remaining = [label for label in labels if label not in held_out_set]
    print(f"labels {len(labels)}  held out {len(held_out)} ({args.language})")
    searches = {"en+" + args.language: {"en", args.language}}
    if extra:
        searches["en only"] = {"en"}
    for power in args.powers:
        index = LabelIndex.build(remaining, coverage_power=power)
        for name, languages in searches.items():
            metrics = evaluate(index, held_out, catalogue, languages)
            summary = "  ".join(f"{key} {value:.3f}" for key, value in metrics.items())
            print(f"power {power:.1f}  {name:<8} {summary}")


if __name__ == "__main__":
    main()
