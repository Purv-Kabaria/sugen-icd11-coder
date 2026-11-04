# Decisions

A running log of the choices behind the coder. Each entry names the commit it ships in by
subject line. Times are IST.

## 2026-11-04 13:10 IST - Toolchain and layout
- Commit: Scaffold Python project with uv
- Decision: Python 3.12 managed by uv with a committed lockfile, `src/` layout, ruff for lint and format, mypy in strict mode, pytest. Considered plain pip with requirements files and Poetry.
- Evidence: uv resolves and installs in seconds, gives a lockfile the Docker build can sync with `--locked`, and runs every tool through `uv run` without activating environments.
- Revisit: nothing pressing.

## 2026-11-04 13:13 IST - Enrich the code list from WHO's own ICD-API container
- Commit: Add WHO MMS crawler and knowledge base builder
- Decision: crawl the full MMS 2026-01 linearization from the official `whoicd/icd-api` Docker image running locally, and keep titles, index terms, inclusions, exclusions, coding notes, definitions and the hierarchy. Considered the public WHO API (needs registered credentials and is rate limited), scraping the browser, and titles only.
- Evidence: the container needs no credentials and serves the same data as the public API. The crawl of 19,299 entities takes 27 seconds. It covers all 18,375 scoreable codes; 16,171 have index terms (5.4 on average), 6,561 have definitions, 2,229 have exclusions. Index terms carry the clinical vocabulary the titles lack, for example "amoebic dysentery" under `1A36.00 Acute amoebiasis`.
- Source: https://hub.docker.com/r/whoicd/icd-api (image digest `sha256:1b77eb6dc43e...`), release 2026-01, retrieved 2026-11-04, ICD-11 licensed CC BY-ND 3.0 IGO. Rebuild with `tools/crawl_who_mms.py` and `tools/build_kb.py`.
- Revisit: Foundation-level synonyms and the WHO ICD-10 to ICD-11 mapping tables.

## 2026-11-04 13:23 IST - Six translations, stored apart from the English core
- Commit: Add WHO MMS crawler and knowledge base builder
- Decision: ship WHO titles, index terms and inclusions in French, Spanish, Portuguese, Arabic, Russian and Chinese, each in its own file. Considered all eight crawled languages (adds German and Turkish) and English only.
- Evidence: all languages in one file came to 10.7 MB, and every rebuild of the English core would have added that again to the history of a bundle that must stay under 25 MB. Split files cost 1.6 MB for the core and 0.8 to 1.2 MB per language, committed once. The UN languages plus Portuguese cover the likeliest non-English notes in East African and Indian clinics among those WHO has translated. No WHO translation exists for Swahili, Hindi or Gujarati.
- Revisit: German and Turkish if evaluation shows notes in those languages.

## 2026-11-04 13:20 IST - Laptop first, HPC only if a slot frees up
- Commit: Add WHO MMS crawler and knowledge base builder
- Decision: run all experiments on the laptop. Considered queueing HPC jobs for synthetic data generation.
- Evidence: the cluster account allows two queued jobs in total, and both were held by another project's jobs with 1.5 to 2 days left. Nothing so far needs a GPU.
- Revisit: open-weight generation of synthetic notes on the cluster once a slot frees up.

## 2026-11-04 13:30 IST - Score labels, not documents, and tune on held-out WHO terms
- Commit: Add catalogue, text normalisation and lexical index
- Decision: every WHO label (title, index term, inclusion, translation) is its own retrieval unit, scored by the IDF mass of its tokens the query covers times the covered fraction to the power p, and a code takes its best label. Considered BM25 over one concatenated document per code.
- Evidence: a benchmark that never touches the dev notes. 3,000 index terms that differ from their title were held out of the index and used as queries (`experiments/label_holdout.py`). Exact code at rank 1: p=0 0.462, p=1 0.510, p=2 0.489, p=3 0.471. Recall at 50 with p=1 is 0.908. Chose p=1.
- Revisit: a dense multilingual encoder alongside the lexical index.

## 2026-11-04 13:45 IST - Offline coder: generic clinical text mechanisms
- Commit: Add offline coder and evaluation metrics
- Decision: weight clauses by role (assessment header or clinician's spoken diagnosis 1.0, other findings 0.25, plan 0.05), cut negated spans with NegEx-style scope, expand common clinical abbreviations, fuse numbers only with qualifiers ("type 2", "stage 1"), and resolve non-leaf codes to the best retrieved descendant or to ICD-11's default child (`.Z` residual, then an "unspecified" title, then "without complication").
- Evidence: dev notes were used for error analysis only, and each fix is a general mechanism, not a rule for a note. Dev top-1/top-3: 0.550/0.767 for the first version, 0.567/0.750 after default-child and number fixes, 0.600/0.817 (50.4 of 75 points) after negation scope, header stripping, speaker roles and abbreviations. The default-child conventions hold across the release: 2,204 parents have a `.Z` child, 12 an "unspecified" titled child, 8 a "without complication" child. To avoid overfitting, offline tuning on dev stops here.
- Revisit: a learned reranker trained on synthetic notes.

## AI tooling mistakes caught

- 2026-11-04: the first tokenizer (written by Claude Code) kept bare numbers as tokens. Error analysis showed dose and vital-sign numbers matching chemical labels ("2,8 dihydroxyadenine urolithiasis" for a constipation note). Fixed by fusing numbers with qualifiers only.
- 2026-11-04: the first negation rule (also Claude Code's) dropped any clause containing a negation word, which discarded "Functional constipation (no alarm features" entirely. Replaced with NegEx-style scope that keeps the words before the cue.
