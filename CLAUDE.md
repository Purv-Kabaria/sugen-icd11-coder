# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A take-home for Sugen Technologies: a system that reads an outpatient consult note and proposes up to three ICD-11 MMS 2026-01 codes, best first. `BRIEF.md` is the specification and wins any conflict with this file. Deadline: 2026-11-07 07:37 IST, three days after the email. Nothing submitted later is reviewed.

The files shipped with the pack are read-only: `BRIEF.md`, `score.py`, `data/dev_notes.jsonl`, `data/mms-2026-01.json`, `data/predictions.example.jsonl`. Never edit them. The graders use the originals.

## How the score works

The only thing graded is `score.py`, run on 600 hidden notes.

| Slice | Notes | Points | Formula |
|---|---|---|---|
| A | 450, same clinics and styles as dev | 75 | 50 x top-1 + 25 x top-3 |
| B | 150, different distribution: every ICD-11 chapter, not always English | 25 | 16.7 x top-1 + 8.3 x top-3 |

Design implications:

- Top-1 is worth twice as much as top-3, so getting the order right matters more than catching more candidates.
- Gold is the most specific code the note supports. In the dev set:
  - All 61 gold codes are leaf categories.
  - 14 of them are `.Z` (unspecified) residuals.
  - A parent code scores as a miss when one of its children fits.
- Of the 2,463 non-leaf categories, 206 have no `.Z` descendant. Those are the only cases where a parent can be the right answer.
- Positions 2 and 3 are cheap insurance. Use them to cover a different level of specificity, for example the specific child plus its `.Z` sibling, not near-duplicates.
- These all count as a miss:
  - a code not in `mms-2026-01.json`
  - a missing id
  - a malformed line

  If an id appears twice, the first line wins.
- A postcoordinated cluster (`A&B`, `A/B`) is scored by its stem only, so emit stem codes.
- Dev notes carry `style` and `gold`. Hidden input has only `id` and `text`. Never read any other field at inference.
- The 60 dev notes come from one setting that appears to be East African primary care. Treat them as a regression check, not training data. Rules written for dev notes will cost slice B points.

## Runtime contract

Breaking any of these gets the submission rejected or zeroed.

- From a clean clone, `docker compose build` and then `docker compose run coder /in/notes.jsonl /out/predictions.jsonl` must work.
- The `coder` service uses the CLI as its `ENTRYPOINT`, so the two paths arrive as arguments. The service mounts `/in` and `/out`.
- Input lines are `{"id", "text"}`. Output lines are `{"id", "codes": [up to 3, best first]}`. Every input id appears exactly once.
- The grading box has 8 CPUs, 16 GB RAM and no GPU, and allows 60 minutes of wall clock for all 600 notes.
- Keys come from `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`:
  - Either, both or neither may be set.
  - They share a US$6 cap for the whole run, about 1 cent per note.
  - Past the cap, every call fails.
- At run time the only reachable hosts are `api.anthropic.com` and `api.openai.com`, through the proxy in `HTTPS_PROXY`:
  - The official Python SDKs honour the proxy by default. Never build an HTTP client with `trust_env=False`.
  - Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
  - Load every model, index and dataset from the image.
- `docker compose build` may use the network. Anything the runtime needs must be in the repo or baked into the image at build time.
- With no key set, every note still gets a prediction from the offline path.
- The submission is rejected without review if:
  - more than 10% of notes come back without a prediction, or
  - the image does not build from a clean clone.

## Pipeline invariants

These hold whatever the implementation looks like.

1. Each prediction line is written once, when it is final, and flushed immediately. Because the first line per id wins, never write a provisional line and then a better one.
2. Offline predictions for all notes are computed before any API call. The API only upgrades them.
3. Two limits gate every API call:
   - A wall-clock deadline: 45 minutes by default, configurable.
   - A spend ceiling: US$4.50 by default, below the US$6 cap because retries also bill.

   On any of these, the remaining notes are written with their offline predictions:
   - the deadline passes
   - the process gets SIGTERM
   - the budget runs out
   - the provider keeps failing
4. Model output is untrusted. Before a code is written:
   - It is normalised.
   - It is checked against the MMS code set.
   - It is resolved to a leaf, or to a parent with no `.Z` descendant.

   Invalid codes are dropped, and the list is filled from the offline ranking.
5. One failed note never stops the run. Exceptions are caught for each note, logged with the note id, and that note falls back to its offline prediction.
6. Providers sit behind one interface, with an adapter for Anthropic and one for OpenAI. The adapter is chosen by which key is present. Tests use a fake provider and never touch the network.
7. Cost is computed from each response's `usage` block, using a price table in config. Never estimate spend from prompt length alone.

## Model and API facts (checked 2026-11-04)

- Anthropic prices, in USD per million input/output tokens:

  | Model | Input | Output |
  |---|---|---|
  | `claude-haiku-5-5` | 0.10 | 0.50 |
  | `claude-sonnet-5-5` | 2 | 10 |
  | `claude-opus-5-5` | 4 | 20 |

  Thinking tokens bill as output.
- Current Claude models reject:
  - non-default `temperature`, `top_p` and `top_k`
  - assistant prefill
  - forced `tool_choice` (`any` or `tool`)
- Get JSON from structured outputs (`output_config.format`). Control reasoning depth with `output_config.effort`.
- Thinking cannot be disabled on Opus 5.5 or Sonnet 5.5. Sonnet 5.5 accepts `thinking: {"type": "between_tools"}` at effort `high` or below.
- Check `stop_reason` before reading content. `refusal` and `max_tokens` both mean the answer is unusable.
- `anthropic` 1.x uses `httpx2`, not `httpx`. Check the pinned major version before passing custom HTTP objects to the client.
- OpenAI prices come from third-party listings and are unverified: `gpt-5-mini` 0.25/2.00, `gpt-5-nano` 0.05/0.40. Confirm them on openai.com before relying on them.
- Load the `claude-api` skill before writing Anthropic SDK code instead of relying on memory.

## External data

The MMS file holds titles only: 18,375 categories with `code`, `title`, `chapter`, `leaf`, `residual`, `depth` and `uri`. The `uri` is the WHO linearization entity id and is the join key for adding WHO data.

- **WHO ICD-API v2**
  - Auth: OAuth2 client credentials at `https://icdaccessmanagement.who.int/connect/token`, scope `icdapi_access`.
  - Headers: `API-Version: v2` and `Accept-Language`.
  - Useful entity fields: `definition`, `indexTerm`, `inclusion`, `exclusion`, `codingNote`, `foundationChildElsewhere`, `fullySpecifiedName`.
  - Endpoints under `/icd/release/11/2026-01/mms`:
    - `/{id}`
    - `/{id}/{residual}`, for `/other` and `/unspecified` uris
    - `/search`
    - `/autocode`
    - `/codeinfo/{code}`
- **Bulk crawls**
  - Run the official container `whoicd/icd-api` locally (for example `-e acceptLicense=true -e include=2026-01_en`) to avoid rate limits.
  - It works offline once created.
  - Leave `saveAnalytics` off.
- **Languages.** MMS 2026-01 is available in:
  - Arabic, Chinese, Czech, English, French
  - German (prerelease), Kazakh, Latin (titles only), Portuguese, Russian
  - Slovak, Spanish, Swedish, Turkish, Uzbek

  Notes in any other language, such as Swahili, Hindi or Gujarati, need translation or cross-lingual retrieval.
- **Credentials.** The data scripts read `ICD_API_CLIENT_ID` and `ICD_API_CLIENT_SECRET` from a gitignored `.env`. The Docker build and runtime never need them.
- **Licence.** ICD-11 content is licensed by WHO under CC BY-ND 3.0 IGO. Credit WHO in the README, and keep WHO text unaltered in shipped artifacts.
- **Provenance.** Record every added source in `DECISIONS.md` and the README:
  - URL
  - release
  - retrieval date
  - licence
  - the committed script that fetches and builds it

  Data files are never hand-edited.

## Evaluation

- Dev score: `uv run python score.py --gold data/dev_notes.jsonl --pred out/dev.jsonl --mms data/mms-2026-01.json --per-note out/dev.per-note.jsonl`
- For every change worth logging, report:
  - top-1 and top-3 per slice
  - the no-key score
  - cost per note
  - p95 latency per note
- Held-out sets live under `eval/`, one directory per set:
  - Each has a `MANIFEST.md` recording how it was made, by which model, with what prompt, and when.
  - Out-of-distribution sets are labelled `"slice": "B"` so `score.py` reports them separately.
- Synthetic notes:
  - come from a different model family than the coder
  - cover every chapter and several languages
  - have their gold codes checked in a second pass
- Never tune prompts or rules on a set that is also used to report a score.

## Engineering standards

The repository is judged as production code.

**Tooling**
- Python 3.12, managed with `uv`, with `uv.lock` committed.
- `src/` layout. Dev tools go in PEP 735 dependency groups.
- These must pass before every commit:
  - `uv run ruff format --check`
  - `uv run ruff check`
  - `uv run mypy --strict src`
  - `uv run pytest`
- Pin every dependency, base image (by version) and model revision so the build is reproducible.

**Design**
- Functional core, imperative shell:
  - Candidate retrieval, ranking, code validation and specificity resolution are pure functions over typed values.
  - File I/O, network, clock, environment and signals live at the edges and are passed in.
- Domain values are frozen dataclasses: `Note`, `Candidate`, `Prediction`.
- Config is one frozen settings object, built from the environment once at startup.
- No mutable state at module level.
- Each module and function has one job. Names state intent. Avoid abbreviations other than ICD, MMS and LLM.

**Comments**
- Comments explain why, never what. Write one only for a non-obvious constraint or to name the brief rule being enforced.
- No commented-out code and no banner comments.
- Public functions get a one-line docstring when the name alone is not enough.

**Errors and logging**
- Catch specific exceptions, such as `anthropic.RateLimitError` and `anthropic.APIConnectionError`. Never use a bare `except`.
- Set explicit request timeouts and a small `max_retries`, so a stalled proxy cannot eat the hour.
- Log with `logging` to stderr and keep stdout clean.
- Treat notes as patient data. Never log:
  - note text
  - prompts that contain note text
  - keys

**Portability**
- Open every file with `encoding="utf-8"`. On Windows the default is cp1252, which corrupts non-English notes.
- `.gitattributes` enforces LF line endings. The dev machine runs Windows, and scripts with CRLF endings break in the Linux image.

**Style**
- No em dashes and no emojis anywhere: code, docs or commit messages.

## Git and deliverables

**Commits**
- Commit each working step as it lands, because reviewers read the history.
- Never squash, rebase or rewrite history.
- A commit message is a single subject line:
  - imperative and capitalised
  - about 50 characters, 72 at most
  - no trailing period
  - no body and no trailers, including Co-Authored-By
- When an invariant or command in this file changes, update CLAUDE.md in the same commit.

**What stays out of git**
- Never commit secrets, `.env`, `out/`, caches or large binaries.
- Do not use Git LFS: a bundle does not include LFS objects.
- Keep the bundle well under 25 MB, Gmail's attachment limit. Anything larger is built during `docker compose build` instead.

**DECISIONS.md**
- Add an entry in the same commit as the work it describes, whenever a real choice is made:
  - an approach, model, data source or rule
  - a cost or time trade-off
  - something dropped
- Use this format:

  ```
  ## 2026-11-04 14:30 IST - Short title
  - Commit: <subject line of the commit this entry ships in>
  - Decision: what was chosen, and the options considered
  - Evidence: the score, failure case or cost that decided it
  - Revisit: what to change with more time
  ```

- When AI tooling produces something wrong and it gets caught, add an entry. The README must describe one such case.

**README.md**
- One page at most. It covers:
  - the dev score and an estimate of the hidden score
  - the data added and where it came from
  - evaluation beyond the 60 dev notes
  - the AI tooling mistake that was caught
  - one line with the hours spent

**Submission**
- Create the bundle with `git bundle create <name>.bundle --all` and attach it to a reply to the original email.
- Before sending, clone the bundle into a fresh directory and run the exact build and run commands twice: once with no key, once with a key.

## HPC

The user has access to the Aurora cluster (SLURM, partition `gpu`, one H100 per node split into 1 GB `shard`s) for offline work: synthetic notes from open-weight models, embedding precomputation, large evaluations. The runtime never depends on it. Full cluster notes live outside the repo in `../HPC_FOR_CLAUDE.md`. Never commit that file: it holds host addresses.

- Reach it with `ssh aurora-ts '<command>'` (or `aurora` on campus). Only light queries run on the login node.
- Per job: at most `--cpus-per-task=16`, `--mem=128G`, `--gres=shard:8`, always `--time`, never `--nodes`/`--nodelist`. Anything larger needs the user's approval for this project.
- The `dpr` account allows 2 queued or running jobs in total, shared with other projects. Check `squeue -u $USER` first, and never touch jobs this project did not create.
- Each job caps BLAS/OpenMP/torch threads at `SLURM_CPUS_PER_TASK`, uses a private `TMPDIR`, keeps network retries to 3 or fewer, and runs from a venv in `~/venvs/sugen`. Compute nodes have no `git`.
- Job scripts live in `hpc/`. Every artifact they produce can be rebuilt from a committed script; record inputs and outputs in `DECISIONS.md`.
- Ship only artifacts that fit the size budget above. Anything larger is regenerated during the image build.
