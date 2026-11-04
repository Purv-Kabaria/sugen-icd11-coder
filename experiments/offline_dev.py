"""Score the offline coder on a labelled set and show each miss.

uv run python experiments/offline_dev.py --gold data/dev_notes.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from icd_coder.catalogue import Catalogue
from icd_coder.evaluation import evaluate, format_report, load_examples
from icd_coder.lexical import LabelIndex, catalogue_labels
from icd_coder.offline import OfflineCoder

KB_DIR = Path("data/kb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    catalogue = Catalogue.load(KB_DIR, args.language)
    coder = OfflineCoder(catalogue, LabelIndex.build(catalogue_labels(catalogue, args.language)))
    examples = load_examples(args.gold)
    predictions = {example.id: coder.predict(example.text) for example in examples}
    print(format_report(evaluate(examples, predictions, catalogue)))
    if args.show_misses:
        for example in examples:
            codes = predictions[example.id]
            if not set(codes[:1]) & example.gold:
                gold = ", ".join(f"{code} {catalogue[code].title}" for code in sorted(example.gold))
                guesses = "; ".join(f"{code} {catalogue[code].title}" for code in codes)
                print(f"{example.id}: gold [{gold}] got [{guesses}]")


if __name__ == "__main__":
    main()
