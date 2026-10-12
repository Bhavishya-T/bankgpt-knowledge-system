# BM25 retrieval experiment

This folder is an experimental submission. It does not replace `submission/`.

BM25 is a standard ranking formula for text search. It gives a passage more
credit when it contains query terms, gives rarer terms more weight than common
ones, and normalizes for passage length. This implementation indexes each
snapshot separately and turns Markdown tables into rows that carry their
column headings. A question with `as_of` selects one whole snapshot before
searching.

Run it from the kit root:

```bash
python3 harness/run_eval.py --submission experiments/bm25 --eval eval/public_eval.jsonl --out results/bm25
```

This experiment has no vector database, embeddings, or language model. Its
answers are retrieved evidence chunks, which lets us see what BM25 finds
before evaluating answer generation.
