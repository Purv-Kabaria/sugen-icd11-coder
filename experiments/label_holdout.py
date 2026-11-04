"""Retrieval benchmark that never touches the dev notes.

Hold out a random sample of WHO index terms, rebuild the label index without them,
then query each held-out term and check where its own code ranks. This measures how
well the lexical index generalises to unseen phrasings of a diagnosis.

    uv run python experiments/label_holdout.py --sample 3000
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

from icd_coder.catalogue import Catalogue
from icd_coder.lexical import Label, LabelIndex, catalogue_labels, query_from_text
from icd_coder.text import tokens

KB_DIR = Path("data/kb")


def held_out_sample(
    labels: Sequence[Label], catalogue: Catalogue, size: int, seed: int
) -> list[Label]:
    """Index terms that differ from their category's title after normalisation."""
    candidates = [
        label
        for label in labels
        if set(tokens(label.text)) != set(tokens(catalogue[label.code].title))
        and tokens(label.text)
    ]
    return random.Random(seed).sample(candidates, size)


def evaluate(index: LabelIndex, queries: Sequence[Label], catalogue: Catalogue) -> dict[str, float]:
    hits = {"top1": 0, "top3": 0, "top10": 0, "top50": 0, "family_top1": 0}
    for query in queries:
        ranked = [scored.code for scored in index.search(query_from_text(query.text), limit=50)]
        if ranked and catalogue.stem(ranked[0]) == catalogue.stem(query.code):
            hits["family_top1"] += 1
        for name, cutoff in (("top1", 1), ("top3", 3), ("top10", 10), ("top50", 50)):
            hits[name] += query.code in ranked[:cutoff]
    return {name: count / len(queries) for name, count in hits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--powers", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    args = parser.parse_args()

    catalogue = Catalogue.load(KB_DIR)
    labels = catalogue_labels(catalogue)
    held_out = held_out_sample(labels, catalogue, args.sample, args.seed)
    held_out_set = set(held_out)
    remaining = [label for label in labels if label not in held_out_set]
    print(f"labels {len(labels)}  held out {len(held_out)}")
    for power in args.powers:
        metrics = evaluate(LabelIndex.build(remaining, coverage_power=power), held_out, catalogue)
        summary = "  ".join(f"{name} {value:.3f}" for name, value in metrics.items())
        print(f"coverage_power {power:.1f}  {summary}")


if __name__ == "__main__":
    main()
