"""Hybrid retrieval experiment: BM25 plus general evidence-quality checks.

This experimental submission is separate from the team's current submission.
It combines statistical text retrieval with source, qualifier, and evidence
checks. The checks are generic (page type and requested fields), not a list of
answers for the public questions.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BM25_PATH = ROOT / "experiments" / "bm25" / "solution.py"
spec = importlib.util.spec_from_file_location("bm25_experiment", BM25_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load BM25 experiment at {BM25_PATH}")
bm25_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm25_module)

Answer = bm25_module.Answer
Source = bm25_module.Source
BM25Knowledge = bm25_module.BM25Knowledge
tokenize = bm25_module.tokenize


def _source_prior(url: str, question: str) -> float:
    """A small, general prior for page type; relevance still comes from BM25."""
    path = url.lower()
    q = question.lower()
    prior = 0.0

    # Penalize sources that often repeat old or contextual information.
    if ".pdf" in path or "/hubfs/" in path:
        prior -= 4.0
    if "/calculators/" in path and not any(word in q for word in ("link", "url", "calculator")):
        prior -= 3.5
    if "/learn/blog/" in path:
        prior -= 2.5
    if "/newsroom/" in path:
        prior -= 1.5
    if "application" in path or "forms/" in path:
        prior -= 3.0
    if "switch-and-earn" in path:
        prior -= 2.0

    # Prefer first-party information pages when the question asks for a fact.
    if path.rstrip("/").endswith("/rates"):
        prior += 2.5
    elif "/personal/" in path or "/business/" in path:
        prior += 1.0
    elif path.rstrip("/").endswith("/contact"):
        prior += 0.75
    return prior


def _term_qualifiers(question: str) -> list[tuple[str, str]]:
    """Extract explicit durations such as 60 months or 30 years."""
    return [
        (number, unit.rstrip("s"))
        for number, unit in re.findall(r"\b(\d{1,3})\s*[- ]?\s*(months?|years?)\b", question.lower())
    ]


def _year_qualifier(question: str):
    match = re.search(r"\b(20\d{2})\s+model[- ]year\b", question.lower())
    return int(match.group(1)) if match else None


def _year_range_covers(text: str, year: int) -> bool:
    for start, end in re.findall(r"\b(20\d{2})\s*[-–]\s*(\d{2}|20\d{2})\b", text):
        first = int(start)
        last = int(end) if len(end) == 4 else (first // 100) * 100 + int(end)
        if first <= year <= last:
            return True
    return bool(re.search(rf"\b{year}\b", text))


def _qualifier_adjustment(question: str, evidence: str) -> float:
    """Reward matching requested fields and penalize passages missing them."""
    adjustment = 0.0
    lower_evidence = evidence.lower().replace("–", "-")

    # Explicit loan/certificate durations are constraints, not merely keywords.
    for number, unit in _term_qualifiers(question):
        pattern = rf"\b{re.escape(number)}\s*[- ]?\s*{unit}s?\b"
        if re.search(pattern, lower_evidence):
            adjustment += 2.5
        else:
            adjustment -= 2.0

    # A model year may be represented by a range in the source table.
    year = _year_qualifier(question)
    if year is not None:
        adjustment += 2.5 if _year_range_covers(evidence, year) else -3.0

    # Exact ARM reset periods are essential qualifiers (e.g. 7/1 is not 5/1).
    ratios = re.findall(r"\b\d{1,2}\s*/\s*\d{1,2}\b", question)
    for ratio in ratios:
        normalized = re.sub(r"\s+", "", ratio)
        if normalized in re.sub(r"\s+", "", lower_evidence):
            adjustment += 4.0
        else:
            adjustment -= 8.0

    # When a numeric rate is requested, evidence should actually state one.
    asks_for_rate = bool(re.search(r"\b(rate|rates|apr|apy|interest)\b", question.lower()))
    if asks_for_rate:
        if "%" in evidence:
            adjustment += 1.0
        else:
            adjustment -= 1.5
    return adjustment


def _is_multi_item_question(question: str) -> bool:
    q = question.lower()
    return (
        "all " in q
        or "list " in q
        or q.startswith("what are ")
        or q.startswith("provide ")
        or (" and " in q and any(word in q for word in ("cards", "rates", "wallet", "hours")))
    )


class HybridKnowledge(BM25Knowledge):
    def answer(self, question, as_of=None):
        snapshot = self._snapshot_for_date(as_of)
        query_tokens = tokenize(question)
        if snapshot is None or not query_tokens:
            return Answer(text="I don't know.", confident=False)

        scored = []
        for chunk in snapshot["chunks"]:
            bm25_score = self._score(query_tokens, chunk, snapshot)
            if bm25_score <= 0:
                continue
            evidence = chunk["body"]
            score = (
                bm25_score
                + _source_prior(chunk["url"], question)
                + _qualifier_adjustment(question, evidence)
            )
            scored.append((score, bm25_score, chunk))

        if not scored:
            return Answer(text="I don't know.", confident=False)

        scored.sort(key=lambda item: item[0], reverse=True)

        # The slash-period qualifier is a strict identifier. Related ARM text
        # without the requested period is not evidence for the requested rate.
        ratios = re.findall(r"\b\d{1,2}\s*/\s*\d{1,2}\b", question)
        if ratios:
            exact = [
                item for item in scored
                if all(re.sub(r"\s+", "", ratio) in re.sub(r"\s+", "", item[2]["body"].lower()) for ratio in ratios)
            ]
            if not exact:
                return Answer(text="The website does not list that specific rate.", confident=False)
            scored = exact

        # A rate answer must be grounded in a passage that actually states a
        # percentage, not merely discusses loans or interest in general.
        if re.search(r"\b(rate|rates|apr|apy|interest)\b", question.lower()):
            rate_evidence = [item for item in scored if "%" in item[2]["body"]]
            if rate_evidence:
                scored = rate_evidence
            elif scored[0][0] < 2.0:
                return Answer(text="I don't know; the website does not publish that rate.", confident=False)

        selected = [scored[0]]
        if _is_multi_item_question(question):
            seen_bodies = {scored[0][2]["body"]}
            seen_urls = {scored[0][2]["url"]}
            for item in scored[1:]:
                chunk = item[2]
                if chunk["body"] in seen_bodies:
                    continue
                # Keep multiple useful rows from one table, and allow a second
                # source when a list naturally spans multiple pages.
                selected.append(item)
                seen_bodies.add(chunk["body"])
                seen_urls.add(chunk["url"])
                if len(selected) >= 6:
                    break

        answer_parts = [item[2]["body"][:700] for item in selected]
        sources = []
        seen_sources = set()
        for item in selected:
            chunk = item[2]
            key = chunk["url"]
            if key not in seen_sources:
                sources.append(Source(url=key, snapshot_id=snapshot["snapshot_id"]))
                seen_sources.add(key)

        return Answer(text="\n".join(answer_parts), sources=sources)


def build():
    return HybridKnowledge()
