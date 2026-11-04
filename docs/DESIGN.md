# Design choices

Each section below covers one real choice: what was picked, what else was considered, and why the others lost. Numbers come from measurements logged in [DECISIONS.md](../DECISIONS.md), which records the same choices in the order they were made, each tied to its commit.

## At a glance

| # | Area | Chosen | Main alternatives | Deciding reason |
|---|---|---|---|---|
| 1 | Where code knowledge comes from | WHO's own ICD-API text, crawled offline | Titles only; a medical vocabulary such as UMLS | Titles alone miss most of the words clinicians use; WHO text is official and licensed for this use |
| 2 | Core search | Word-level search over 700k short WHO labels | Text embeddings; a classifier over 18k codes | Works with no GPU, no training data, in any chapter, and explains itself |
| 3 | Label scoring | Coverage-weighted rare-word score | The same score without coverage; BM25 over one document per code | Held-out WHO terms: 0.510 top-1 against 0.462 without coverage |
| 4 | Reading the note | Clause weights, scoped negation, speakers, abbreviations | Treat the note as one bag of words | Dev top-1 0.550 to 0.600 with general mechanisms only |
| 5 | Level of detail | Default child, and a child must beat its parent | Always the best-scoring code | A parent scores as a miss when a child fits |
| 6 | Other languages | WHO translations, searched only for that language | English only; machine translation; all languages always | French top-1 0.253 to 0.495; English unharmed |
| 7 | Ranking | LightGBM learning-to-rank ensemble, rules as fallback | Rules only; logistic regression; a cross-encoder | +2.3 points top-1, +3.7 top-3 on unseen code families |
| 8 | Training data | 12,000 notes built from WHO text | LLM-written notes; the 60 dev notes | Free, instant, reproducible, no leakage |
| 9 | Model path | Two calls: read, then choose from a WHO-built list | One call; a tool-use loop; codes from memory | Models misremember codes; two calls keep cost near US$0.0006 a note |
| 10 | Run order | Offline for all notes first, models only upgrade | Model first with offline fallback per note | Every note has an answer before any risk is taken |
| 11 | Providers | Anthropic, OpenAI and any OpenAI-compatible server | One SDK per vendor; LiteLLM | One small adapter covers most servers without a large dependency |
| 12 | Concurrency | Threads inside one process | A job queue with workers (Celery, RQ) | The contract is one command; a broker adds failure points and no speed |
| 13 | Packaging | Index built during `docker compose build`, container runs as root | Build index at run time; non-root user | Saves a minute per run; a non-root user may not be able to write to `/out` |

## 1. Where code knowledge comes from

The provided code list has only titles. Clinicians rarely write titles: they write "falciparum malaria" or "cerebral malaria", not "Plasmodium falciparum malaria with cerebral complications". WHO publishes, for every code, index terms (other names), inclusions, exclusions, definitions and coding notes, plus official translations.

- **Chosen:** run WHO's official ICD-API container locally and crawl release 2026-01 in English, French, Spanish, Portuguese, Arabic, Russian and Chinese. The result is stored compactly in `data/kb` (7.4 MB) and committed, so the build never needs the network for it.
- **Not chosen: the public WHO API.** Same data, but rate limited and needs credentials.
- **Not chosen: UMLS or SNOMED CT.** Licences that cannot ship in a public image, and the mapping to ICD-11 MMS is incomplete.

## 2. Core search: word-level search over WHO labels

- **Chosen:** split every WHO title, index term, inclusion and translation into its own short label and search them with weighted words. It needs no GPU and no training data, covers every chapter equally, and every answer can be traced to the WHO label that matched.
- **Not chosen: text embeddings.** A good multilingual embedding model would help with paraphrase, but would add hundreds of megabytes, be slow on 8 CPUs for 700k labels, and blur the precise distinctions ICD needs ("with" versus "without complication").
- **Not chosen: a classifier over all 18,375 codes.** Most codes would have no training examples, and slice B covers every chapter.

## 3. Label scoring

A label's score is the rare-word weight of the words the note covers, multiplied by the share of the label covered. So a fully covered long label beats a partly covered one.

- **Chosen** by a benchmark that never touches dev notes (`experiments/label_holdout.py`): hide one WHO index term per code and use it as the query. Top-1 for the coverage power: 0 gave 0.462, 1 gave 0.510, 2 gave 0.489 and 3 gave 0.471. Power 1 won.
- **Not chosen: classic BM25 over one long document per code.** Long documents reward codes with many index terms and let a half-matched label win.

## 4. Reading the note

Dev notes were used only to find kinds of mistakes, and each fix is a general mechanism, not a rule for one note.

| Mechanism | Mistake it fixes |
|---|---|
| Clause weights (diagnosis 1.0, history 0.25, plan 0.05) | A drug in the plan or an old condition outranking today's diagnosis |
| Negation with a scope that ends at the phrase | "No alarm features" removing the whole clause with the real diagnosis |
| Speakers in transcripts | The patient's guess ("I think it's malaria") counting like the doctor's |
| Abbreviation expansion | "AF" matching "AF type amyloidosis" instead of atrial fibrillation |
| Numbers kept only with a qualifier ("type 2") | Doses and vital signs matching chemical names |

Dev top-1 and top-3 went from 0.550/0.767 to 0.600/0.817. Tuning on dev then stopped, to avoid fitting 60 notes.

## 5. Choosing the level of detail

The right answer is the most specific code the note supports. A heading such as "malaria" scores zero when "malaria, unspecified" exists.

- **Chosen:** follow ICD-11's own conventions. A heading becomes its default child: the `.Z` "unspecified" code, else a child titled "unspecified", else a child "without complication". A child replaces its parent only when it outscores the parent, which means the note mentions what makes the child different.
- **Not chosen: always take the best-scoring code.** A half-mentioned child ("mumps" matching "mumps orchitis") would beat the correct "mumps without complication".

## 6. Notes in other languages

- **Chosen:** add WHO's translated labels, detect the note's language, and search English labels plus that language only.
- **Evidence:** on held-out translated WHO terms, top-1 rose from 0.253 to 0.495 for French and from 0.159 to 0.497 for Spanish. Searching every language for every note had dropped English dev top-1 from 0.600 to 0.583; the language filter restored it.
- **Not chosen: machine translation.** No translation model fits the runtime without a GPU, and the network reaches only the two model APIs.
- **Gap:** WHO has no Swahili or Hindi release. With an API key, the model's first call rewrites any language into English search phrases; without one, those notes depend on shared medical words.

## 7. Ranking: a learned reranker

The candidate list contains the right code 93% of the time on held-out notes, but rules put it first only about 72% of the time. That gap is an ordering problem, which is exactly what learning-to-rank models solve.

- **Chosen:** five LightGBM LambdaRank models trained with different seeds; their scores are averaged. They use 30 simple features computed by the same code at training and run time. If the model files are missing or were built for different features, the coder uses the rules, so the reranker can only add accuracy, never break a run.
- **Evidence:** on 2,387 notes from code families never seen in training, top-1 rose from 0.721 to 0.744 and top-3 from 0.800 to 0.837; the 95% confidence interval for the top-1 gain is +1.6 to +3.2 points. On the hardest kind (paraphrased WHO definitions) top-1 rose from 0.425 to 0.483.
- **Not chosen: logistic regression.** It scores candidates one at a time instead of learning an order.
- **Not chosen: a cross-encoder neural model.** Too slow without a GPU, and no public model is trained for ICD-11 MMS.

## 8. Training data for the reranker

- **Chosen:** `tools/build_who_notes.py` writes 12,000 notes from templates around WHO text in about a second. Four kinds: the code's definition (prose that is never searched, so it cannot match word for word), a noisy index term, the parent's term with the default child as the answer, and a translated title. English notes add complaints, history from another chapter, a ruled-out sibling and a plan, in SOAP, narrative or transcript style.
- **Leakage control:** code families are split 80/20 by a hash, so test notes are about families the model never saw. The tool refuses dev notes as input.
- **Not chosen: LLM-written notes.** They read more like real consults and the generator exists (`tools/synthesize.py`, with judging and rate limiting), but a free API tier needed hours, and the run was stopped by request. They remain the best next step.
- **Not chosen: training on dev notes.** Sixty notes from one clinic would teach that clinic's habits and leave nothing to check against.

## 9. The model path

- **Chosen:** call 1 reads the note in any language and returns the main condition in English, search phrases and code guesses. The coder searches WHO labels with those phrases, merges the rankings and shows the model at most 10 code families as small trees with WHO's notes. Call 2 picks up to three codes from that list.
- **Cost:** with Claude Haiku 5.5 (US$0.10 / 0.50 per million tokens), about US$0.0006 per note, a twentieth of the one cent a note allowed.
- **Not chosen: asking for codes directly.** Models misremember ICD-11 codes; in testing a model guessed `1A61.0` and `1B01.0` for malaria.
- **Not chosen: a tool-use loop where the model searches by itself.** Less predictable cost and time.
- **Limit:** no Anthropic or OpenAI key was available while building this, so the path is covered by tests with a fake provider but its accuracy is unmeasured.

## 10. Run order

- **Chosen:** offline answers for all notes first; model calls only upgrade them; each line is written once, when final.
- **Not chosen: model first, offline per note on failure.** A crash, timeout or exhausted budget halfway through could leave notes with nothing, and more than 10% missing rejects the submission.

## 11. Providers

- **Chosen:** one interface with three adapters: Anthropic, OpenAI, and any server that speaks the OpenAI Chat Completions protocol (Groq, OpenRouter, Together, Mistral, Gemini, Azure, vLLM, Ollama). Anthropic and OpenAI win automatic choice because the grading box reaches only those two. Cost always comes from the usage the server reports; an unpriced server is billed at a cautious rate so the budget still stops the run.
- **Not chosen: LiteLLM.** It covers more providers, but it is a large, fast-changing dependency in an image that must build the same way every time.

## 12. Concurrency

- **Chosen:** one process; model calls overlap in a thread pool of 16, under the deadline and budget.
- **Not chosen: an asynchronous job pipeline with a queue and workers.** The contract is a single `docker compose run` with no other services. The offline path needs about a second per 60 notes, and the model path waits on network replies, which threads already overlap. A broker would add a service that must start, stay healthy and drain inside the hour, with no speed gain.

## 13. Packaging

- **Chosen:** a two-stage image on a pinned Python 3.12.9 base with a locked dependency set. The index is built and stored during `docker compose build`, so a run loads it in about a second. The container runs as root.
- **Why root:** the grader mounts `/out` from the host. A non-root user may not be allowed to write there, which would zero the submission. Robustness wins over hardening for this one case.
