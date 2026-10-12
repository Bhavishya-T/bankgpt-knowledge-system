# Hybrid retrieval experiment

This experiment layers general evidence checks on the BM25 retriever:

- BM25 ranks paragraphs, list items, and table rows.
- A small source prior favors rate/product information pages and discounts
  calculators, PDFs, applications, blogs, and promotions.
- Explicit durations, model years, and ARM reset periods are checked as
  qualifiers, so a similar product with a different term is weaker evidence.
- Multi-item questions can return several passages and cite multiple pages.
- A rate question without a matching rate passage can decline instead of
  presenting related prose as the answer.
- Historical questions select a complete snapshot before searching it.

These are general retrieval and evidence rules, not answers for individual
practice questions. This version has no vector database or language model.

Run from the kit root:

```bash
python3 harness/run_eval.py --submission experiments/hybrid --eval eval/public_eval.jsonl --out results/hybrid
```
