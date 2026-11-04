# Engineering principles

The repository is written as if it were going to production. This page lists the principles behind it, what each one means in this code, and why it matters for this task.

## 1. Never lose a note

The grader rejects a submission with more than 10% of notes missing, and keeps only the first line per note. So the strongest rule is: every note gets an answer, written once.

| Practice | Where |
|---|---|
| Offline answers for every note before any network call | `pipeline.run` |
| Each line written once, when final, and flushed at once; the file is synced to disk on close | `notes.PredictionWriter` |
| Bad input lines (broken JSON, missing id, duplicate id, byte-order mark) are skipped with a warning, not fatal | `notes.read_notes` |
| One failing note is caught, logged with its id, and falls back | `pipeline._upgrade` |
| If the whole run fails, unanswered notes still get a line | `cli.main` |
| A missing or mismatched reranker falls back to rules | `reranker.Reranker.load` |
| An unreadable index cache is rebuilt, never trusted | `index_store.load_or_build` |

## 2. A pure core, with side effects at the edges

The parts that decide answers (search, features, ranking, choosing the level of detail, checking codes) are plain functions over typed values. Reading files, the network, the clock, environment variables and signals happen only at the edges (`cli.py`, `notes.py`, `config.py`, `llm/providers.py`) and are passed in.

Why: the core can be tested quickly and exactly, with no mocks of the outside world. The deadline, for example, takes a clock function, so tests can move time forward instantly.

## 3. Typed, immutable values

Notes, categories, labels, scored codes, settings and usage records are frozen dataclasses. Settings are read from the environment once, at startup, into one object. There is no mutable state at module level.

Why: nothing can change a value behind another component's back, which matters once model calls run in parallel threads.

## 4. Treat model output as untrusted

A model's answer goes through the same checks as user input: normalised, looked up in the official code list, resolved to the most specific valid code, de-duplicated and padded. Structured output (a JSON schema) is requested from every provider, and the reply is still parsed defensively.

## 5. Limits are part of the design

| Limit | How it is respected |
|---|---|
| 60 minutes | Deadline at 45 minutes for model calls; offline path takes seconds |
| US$6 | Budget of US$4.50, checked with a worst-case price before every call and charged from actual usage after |
| Network reaches only two hosts, through a proxy | Official SDKs, which honour `HTTPS_PROXY`; nothing is downloaded at run time |
| No GPU, 8 CPUs, 16 GB | Word-level search and small tree models; index loaded from disk |
| No key at all | The offline coder is a complete solution on its own |

## 6. Performance: measure, cache, parallelise

| Measure | Value |
|---|---|
| Building the label index (700k labels) | 5.7 s, using a character translation table, memoised stemming and a process pool (98 s without them) |
| Loading the stored index | About a second, done at image build so a run never rebuilds it |
| Search | Touches only labels that share a word with the note |
| Offline coding with the reranker | 60 dev notes in 1.2 s (about 20 ms a note) |
| Candidate features | 10.7 ms a note |
| Docker image | 680 MB |

Expensive results are stored with a key made from a hash of their inputs and the source code that produced them (the index, the reranker's training features, model replies during development). Changing the tokenizer therefore rebuilds the index automatically; nothing stale is ever reused.

## 7. Reproducible builds

- Python 3.12, dependencies locked with `uv.lock`, base image and uv version pinned.
- Every data artifact is rebuilt by a committed script with a fixed seed; no data file is edited by hand.
- WHO content is kept unaltered, as its licence (CC BY-ND 3.0 IGO) requires.

## 8. Checks that run before every commit

| Check | Command |
|---|---|
| Formatting | `uv run ruff format --check` |
| Lint (wide rule set) | `uv run ruff check` |
| Strict typing | `uv run mypy --strict src` |
| Tests (97) | `uv run pytest` |

Tests never touch the network: model providers are replaced by fakes that return scripted answers, so budget exhaustion, failure streaks, deadlines and stop signals are all tested deterministically.

## 9. Privacy and secrets

- Note text, prompts that contain note text, and keys are never logged.
- Keys come only from the environment; `.env` is ignored by git; a configured endpoint key is excluded from the settings' printed form.

## 10. No leakage between training and testing

This was treated as the main risk to the numbers being honest.

| Rule | How it is enforced |
|---|---|
| Dev notes are for regression checks, never training or tuning | `train_reranker.py` refuses them; dev guided only the first round of general clinical-text mechanisms |
| Test notes come from code families the model never trained on | Families split 80/20 by a hash of the family code |
| Cross-validation folds are grouped by family | Implemented in `train_reranker.grouped_folds` |
| Model choices use training folds only | Boosting rounds come from cross-validation; held-out notes only report results |
| LLM-written evaluation notes are written by a different model family (Qwen) than the coders (Claude, GPT) | `tools/synthesize.py` defaults |

## 11. Simple, readable code

- Small modules with one job each, and names that say what things are.
- Comments explain why, never what. Public functions have a one-line docstring when the name is not enough.
- No em dashes or emojis anywhere, including commit messages.

## 12. History that tells the story

Each working step is its own commit, with a single imperative subject line. Every real choice has a dated entry in [DECISIONS.md](../DECISIONS.md), committed together with the work it describes, so the log and the code history line up.
