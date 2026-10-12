# BankGPT: a knowledge system that knows what is true now

I built BankGPT for the Knowledge Hackathon to answer credit-union questions from five dated snapshots of Dupaco's website. The challenge is not just finding a matching sentence: rates change, old promotions remain online, and tables contain similar products with different terms and tiers. BankGPT has to use the right evidence, respect the requested date, and admit when the snapshots do not support an answer.

## How it works

I kept the final system deliberately small in concept:

1. **Keep snapshots separate.** During ingestion, each dated crawl is indexed independently. An `as_of` question uses the latest complete snapshot on or before that date; otherwise the latest snapshot is used.
2. **Preserve useful document structure.** Markdown tables are split into rows while keeping column headings and nearby labels with their values. This helps distinguish, for example, a 60-month auto-loan tier from a similar term.
3. **Retrieve with BM25.** The system ranks passages using lexical relevance, then applies small, general signals for source type and explicit qualifiers. It has no vector database or embedding dependency.
4. **Answer from evidence.** When `OPENAI_API_KEY` is set, `gpt-4.1-mini` receives the retrieved passages and must cite their IDs for its claims. The implementation checks cited IDs and numeric claims against the supplied evidence. If the API is unavailable, it returns retrieved evidence as an unconfident fallback.
5. **Report possible conflicts.** `conflicts()` compares numeric passages in similar page sections and favors later snapshots. This is heuristic and needs stronger validation on the full corpus.

I inspected the five supplied snapshots, but the answer pipeline does not load evaluation questions, question IDs, or expected-answer strings at runtime. I used the public set to test and adjust general retrieval behavior; I did not add per-question branches or encode expected values. I could not see the private evaluation questions, and their wording may differ, so I focused on reusable behavior: lexical relevance, snapshot dates, source quality, product qualifiers, and evidence checks. These general signals are intended to transfer across questions; they do not guarantee performance on unseen wording. The system reads the supplied snapshots and does not call live websites.

## Evaluation and limitations

The best successful API-enabled development run scored **0.6488 QA** and **0.6352 overall** on the 40-question public set. The final cleaned revision in this repository scored **0.6446 QA** and **0.6310 overall**, with 0 runtime errors, 0.975 consistency, a 3.69-second p95 answer time, and 5.0-second ingestion. I kept the final revision because it fixed a general table-chunking issue and passed the supplied synthetic conflict check; its public-set score was slightly lower. These are measured public-set results, not a claim of complete coverage.

API availability matters: `OPENAI_API_KEY` and a working connection are needed for the reported API-enabled behavior. If the request cannot be made, the system deliberately falls back to retrieved passages marked unconfident rather than composing an unsupported answer.

The strongest categories were entity attribution (1.00) and tier disambiguation (0.83). Stale traps (0.45), current values (0.33), and conflict resolution (0.33) remain weak. The supplied synthetic conflict test matched its one gold item, but that single test does not establish conflict precision on the real corpus; `conflicts()` reaches its 300-item cap there, so over-reporting remains a known risk.

I also tested embeddings. They slightly improved retrieval coverage but reduced the end-to-end score from 0.6352 to 0.6319 and increased ingestion from about 5 seconds to 851.7 seconds. I removed that path from the final implementation because it added cost without improving the result that matters: end-to-end answer quality.

## Run the public evaluation

The snapshot data is distributed separately in the participant kit and is intentionally not committed here. Copy the provided `snapshots_zip/` directory into the repository root before running the evaluator.

```bash
python3 -m pip install -r submission/requirements.txt
export OPENAI_API_KEY="your-key"
python3 harness/run_eval.py --submission submission --eval eval/public_eval.jsonl --out results/run
```

Python 3.9 or newer is required. The API key is optional for running the fallback, but the API-enabled result above was produced with a key available. `OPENAI_API_KEY` and local result files should not be committed.

## Project layout

- `submission/solution.py` implements the required `build()` entry point.
- `experiments/rag_openai/solution.py` contains the snapshot-aware answer pipeline.
- `experiments/bm25/solution.py` contains table-aware chunking and BM25 retrieval.
- `experiments/hybrid/solution.py` contains the general source and qualifier signals used during retrieval.
- `harness/` and `eval/public_eval.jsonl` provide the public evaluation workflow.
