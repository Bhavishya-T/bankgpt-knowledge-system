# Implementation notes

I built this submission around retrieval and evidence checking rather than asking a model to answer from memory. The required entry point is `solution.py`; `build()` returns the object used by the evaluator through `ingest(snapshot)`, `answer(question, as_of=None)`, and `conflicts()`.

## Answer pipeline

1. Each snapshot is ingested and chunked separately. Table rows retain their headers and nearby product context.
2. The requested date selects the newest whole snapshot available on or before that date. With no date, the newest snapshot is used.
3. BM25 ranks passages. General source and qualifier signals help distinguish official product information and explicit terms, tiers, or model years.
4. If an OpenAI API key is available, `gpt-4.1-mini` writes the response from the retrieved passages and returns evidence IDs for its claims. The implementation rejects unknown evidence IDs and numeric claims that do not appear in the cited evidence. If the API is unavailable, the fallback returns retrieved text without claiming confidence.

I do not store benchmark answers in the implementation, and the system does not make live website requests. There is no vector database or embedding code in the final path.

## Results and known limits

My last successful API-enabled run on the public 40-question evaluation reported:

| Measure | Result |
| --- | ---: |
| QA score | 0.6446 |
| Overall score | 0.6310 |
| Consistency | 0.975 |
| p95 answer time | 3.69 s |
| Ingestion time | 5.0 s |
| Runtime errors | 0 |

During final preparation, I reran the set but the OpenAI request could not connect. The no-API fallback scored 0.5271; it returns retrieved passages as unconfident evidence rather than composing answers. The 0.6310 overall result above depends on successful API access, and the fallback run did not reproduce it.

The main remaining weaknesses are stale-trap handling, current-value questions, and conflict resolution. The synthetic conflict test contains only one gold item; passing it is not evidence that `conflicts()` is precise on the full corpus. On the full corpus, the method reaches its 300-item cap, and over-reporting remains possible.

I tried an embedding-based retrieval path as well. It improved retrieval coverage slightly but lowered the end-to-end score and increased ingestion time from about 5 seconds to 851.7 seconds. I removed it from the final implementation because it did not improve end-to-end answer quality.

## Reproducing the public run

The participant-kit snapshots are not included in this repository. Copy the provided `snapshots_zip/` directory into the repository root, install `submission/requirements.txt`, and run:

```bash
python3 harness/run_eval.py --submission submission --eval eval/public_eval.jsonl --out results/run
```

Set `OPENAI_API_KEY` in the environment to use API-backed generation. Without it, the code uses the unconfident evidence fallback. Keep API keys, snapshot data, and generated result files out of Git.
