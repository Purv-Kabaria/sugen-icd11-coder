# Design decisions

Each section records one decision in the same shape: what was decided, why, which alternatives were considered and why they were not chosen, and the measured result. The dated, commit-by-commit record of the same decisions is in [DECISIONS.md](../DECISIONS.md).

## Summary

| # | Area | Decision | Main alternatives | Deciding evidence |
|---|---|---|---|---|
| 1 | Code knowledge | WHO's own ICD-API text, crawled offline | Titles only; UMLS or SNOMED CT | Clinicians rarely write titles; WHO text is official and licensed for this use |
| 2 | Retrieval | Word-level search over 700k short WHO labels | Text embeddings; a classifier over 18k codes | No GPU or training data needed, covers every chapter, explainable |
| 3 | Label scoring | Rare-word weight times share of the label covered | The same without coverage; BM25 over whole code documents | 0.510 vs 0.462 top-1 on held-out WHO terms |
| 4 | Reading the note | Clause weights, scoped negation, speakers, abbreviations | The note as one bag of words | Dev top-1 0.550 to 0.600 with general mechanisms |
| 5 | Level of detail | ICD-11 default-child rules; a child must outscore its parent | Always the best-scoring code | A parent scores as a miss when a child fits |
| 6 | Other languages | WHO translations, searched only for the note's language | English only; machine translation; every language always | French top-1 0.253 to 0.495, English unharmed |
| 7 | Ranking | LightGBM learning-to-rank ensemble, rules as fallback | Rules only; logistic regression; cross-encoder | +2.3 top-1, +3.7 top-3 on unseen code families |
| 8 | Training data | 12,000 notes generated from WHO text | LLM-written notes; the dev notes | Free, reproducible, no leakage |
| 9 | Model path | Two calls: read the note, then choose from a WHO-built shortlist | One call; tool-use loop; codes from memory | Dev 50.8 to 68.7 points at $0.0023 a note |
| 10 | Run order | Offline answers for every note first; models only upgrade | Model first, offline on failure | No note can be left blank |
| 11 | Providers | One interface: Anthropic, OpenAI, OpenAI-compatible, Bedrock | One SDK per vendor; LiteLLM | Small adapters, no large dependency |
| 12 | Concurrency | Threads in one process | A job queue with workers | One-command contract; no speed gain from a broker |
| 13 | Measuring the model path | DeepSeek V3.2 on Bedrock, billed to AWS credits | Groq free tier; Claude on Bedrock | Strong model, inside credits, same prompts |
| 14 | Packaging | Index built at image build; container runs as root | Index built per run; non-root user | Fast start; `/out` must be writable |

## 1. Code knowledge: WHO's own text

**Decision.** Run WHO's official ICD-API container locally and crawl release 2026-01: index terms, inclusions, exclusions, definitions, coding notes and the hierarchy, in English and six translations. Store the result compactly in `data/kb` (7.4 MB) so the image build needs no network for it.

**Why.** The provided list has titles only, and clinicians rarely write titles. They write "cerebral malaria", not "Plasmodium falciparum malaria with cerebral complications". WHO's index terms are exactly the other names clinicians use.

**Alternatives.**
- *The public WHO API:* the same data, but rate limited and needing credentials.
- *UMLS or SNOMED CT:* their licences do not allow shipping them in a public image, and their mapping to ICD-11 MMS is incomplete.

## 2. Retrieval: word-level search over WHO labels

**Decision.** Store every title, index term, inclusion and translation as its own short label (about 700,000 in total) and search them with weighted words.

**Why.** It needs no GPU and no training data, treats every chapter equally, and every answer can be traced to the WHO label that matched.

**Alternatives.**
- *Text embeddings:* better on paraphrase, but they add hundreds of megabytes, are slow on 8 CPUs across 700k labels, and blur distinctions ICD depends on, such as "with" versus "without complication".
- *A classifier over all 18,375 codes:* most codes would have no training examples, and slice B covers every chapter.

## 3. Label scoring

**Decision.** A label scores the rare-word weight of the words the note covers, multiplied by the share of the label covered. A code scores as its best label.

**Why.** A fully mentioned long label ("acute amoebic dysentery") should beat a short one ("dysentery"), and a half-mentioned label should lose.

**Alternatives.**
- *The same score without the coverage factor.*
- *BM25 over one document per code:* long documents reward codes with many synonyms and let a half-matched label win.

**Result.** Top-1 on held-out WHO terms: 0.462 without coverage, 0.510 with it.

## 4. Reading the note

**Decision.** Split the note into clauses and weight them: diagnosis 1.0, history 0.25, plan 0.05. Negation removes only the words in its scope. In transcripts, the patient's words count less than the clinician's. Abbreviations are expanded. Numbers are kept only with a qualifier ("type 2").

**Why.** Each mechanism fixes a general kind of mistake:

| Mechanism | Mistake it prevents |
|---|---|
| Clause weights | A drug in the plan, or an old condition, outranking today's diagnosis |
| Scoped negation | "No alarm features" removing the clause that holds the real diagnosis |
| Speaker roles | A patient's guess counting like the doctor's assessment |
| Abbreviations | "AF" matching "AF type amyloidosis" instead of atrial fibrillation |
| Qualified numbers only | Doses and vital signs matching chemical names |

**Alternative.** *The whole note as one bag of words:* simpler, but history and plans pull the ranking away from the diagnosis.

**Result.** Dev top-1 0.550 to 0.600 and top-3 0.767 to 0.817. These are general mechanisms, not rules for particular notes, and dev was not used for tuning after this.

## 5. Level of detail

**Decision.** Follow ICD-11's conventions. A heading becomes its default child: the `.Z` "unspecified" code, otherwise a child titled "unspecified", otherwise a child "without complication". A more specific child replaces its parent only when it outscores the parent, which happens when the note mentions what makes the child different.

**Why.** Gold is the most specific code the note supports, and a heading scores as a miss when a child exists.

**Alternative.** *Always take the best-scoring code:* a half-mentioned child ("mumps" matching "mumps orchitis") would beat the correct "mumps without complication".

## 6. Notes in other languages

**Decision.** Add WHO's translated labels, detect the note's language, and search English labels plus that language only.

**Why.** WHO publishes official translations, so non-English notes can match official terms directly.

**Alternatives.**
- *English only:* misses most non-English notes.
- *Machine translation:* no translation model fits the runtime without a GPU, and the network reaches only the two model APIs.
- *Search every language for every note:* foreign labels add noise and lowered English dev top-1 from 0.600 to 0.583.

**Result.** On held-out translated terms, French top-1 rose from 0.253 to 0.495 and Spanish from 0.159 to 0.497, with English unharmed. WHO has no Swahili, Hindi or Gujarati release; with a key, the model reads those languages instead.

## 7. Ranking: a learned reranker

**Decision.** Rank candidates with five LightGBM LambdaRank models trained with different seeds, averaging their scores, over 30 simple features shared by training and run time. If the model files are missing or were built for different features, the rules rank instead.

**Why.** The correct code is on the candidate list 93% of the time, but the rules put it first only 72% of the time. That gap is an ordering problem, which is what learning-to-rank solves.

**Alternatives.**
- *Rules only:* the ordering gap above stays.
- *Logistic regression:* scores candidates one at a time instead of learning an order.
- *A cross-encoder:* too slow without a GPU, and no public model is trained for ICD-11 MMS.

**Result.** On 2,387 notes from code families never seen in training, top-1 rose from 0.721 to 0.744 (95% interval +1.6 to +3.2 points) and top-3 from 0.800 to 0.837.

## 8. Training data

**Decision.** Generate 12,000 labelled notes from WHO text with templates (`tools/build_who_notes.py`, about one second). There are four kinds: a paraphrased definition, a noisy index term, a parent term whose answer is the default child, and a translated title. English notes add complaints, history from another chapter, a ruled-out sibling condition and a plan, in SOAP, narrative or transcript style.

**Why.** It is free, reproducible from a seed, covers every chapter, and its labels are exact.

**Leakage control.** Code families are split 80/20 by a hash, so test notes are about families the model never saw. Cross-validation folds are grouped by family, and the training tool refuses dev notes.

**Alternatives.**
- *LLM-written notes:* they read more like real consults, and a generator for them exists (`tools/synthesize.py`), but producing enough at free-tier rate limits takes hours. They are the natural next training set.
- *The dev notes:* sixty notes from one clinic would teach that clinic's habits and leave nothing to check against.

## 9. The model path

**Decision.** The first call reads the note in any language and returns the main condition in English, search phrases and code guesses. The coder searches WHO labels with those phrases, merges the rankings, and shows the model at most 10 code families as small trees with WHO's notes. The second call chooses up to three codes from that list. The answer is validated against the official list, resolved to the most specific valid code and padded from the offline answer.

**Why.** Models misremember ICD-11 codes; in testing, a model guessed `1A61.0` and `1B01.0` for malaria. Choosing from a list built from WHO data removes that failure, and two short calls keep cost and time predictable.

**Alternatives.**
- *One call asking for codes directly:* invents codes.
- *A tool-use loop where the model searches by itself:* less predictable cost and time.

**Result.** Dev rose from 50.8 to 68.7 points, held-out top-1 from 0.760 to 0.853, at $0.0023 to $0.0026 a note.

## 10. Run order

**Decision.** Compute offline answers for every note first; model calls only upgrade them. Each line is written once, when final, and flushed at once.

**Why.** More than 10% missing notes rejects the submission, and the scorer keeps only the first line per note.

**Alternative.** *Model first, offline answer on failure:* a crash, timeout or exhausted budget partway through could leave notes without an answer.

## 11. Providers

**Decision.** One interface with adapters for Anthropic, OpenAI, any server that speaks the OpenAI Chat Completions protocol (Groq, OpenRouter, Together, Mistral, Gemini, Azure, vLLM, Ollama) and Amazon Bedrock (Claude through the Anthropic SDK, other models through Bedrock's Converse API). Anthropic and OpenAI are chosen automatically when their keys are set; the others only when configured. Cost always comes from the usage each reply reports, and an unpriced server is billed at a cautious rate so the budget still stops the run.

**Why.** The grading box reaches only Anthropic and OpenAI, but development and other deployments need other options.

**Alternatives.**
- *LiteLLM:* covers more providers, but it is a large, fast-changing dependency in an image that must build the same way every time.
- *One SDK per vendor:* more code for the same protocol.

## 12. Concurrency

**Decision.** One process; model calls overlap in a thread pool of 16, under the deadline and budget.

**Why.** The offline path needs about one second per 60 notes, and the model path waits on network replies, which threads overlap well.

**Alternative.** *An asynchronous job pipeline with a queue and workers:* the contract is a single `docker compose run` with no other services, so a broker would add a service that must start, stay healthy and drain inside the hour, with no speed gain.

## 13. Measuring the model path

**Decision.** Measure with DeepSeek V3.2 on Amazon Bedrock, billed to AWS credits, through the unchanged prompts and pipeline.

**Why.** No Anthropic or OpenAI key was available. DeepSeek V3.2 is a strong general model at $0.62 / $1.85 per million tokens, and AWS bills it directly.

**Alternatives.**
- *Claude on Bedrock:* the closest match to grading, but AWS sells it through Marketplace, which needs a card and is not covered by promotional credits.
- *Groq's free tier:* free, but its smaller models and rate limits make it a weaker and slower stand-in.
- *Kimi K2.5, GLM 5, Mistral Large 3, Qwen3 Next, gpt-oss-120b, Nova Pro:* all reachable and similarly priced; DeepSeek V3.2 was chosen as a strong general model at low cost.

## 14. Packaging

**Decision.** A two-stage image on a pinned Python 3.12.9 base with a locked dependency set. The label index is built during `docker compose build`, so a run loads it in about a second. The container runs as root.

**Why.** Building the index takes several seconds and should never count against the graded hour. The grader mounts `/out` from the host, and a non-root user may be unable to write there, which would zero the submission.

**Alternative.** *A non-root user:* better hardening, but robustness wins for this one case.
