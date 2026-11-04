# ICD-11 coder

Reads an outpatient consult note and proposes up to three ICD-11 MMS 2026-01 codes, best first.

```
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

Work in progress. Design decisions are logged in `DECISIONS.md`.
