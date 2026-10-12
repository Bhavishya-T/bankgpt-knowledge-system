"""BM25 retrieval experiment for the BankGPT knowledge hackathon.

This is deliberately separate from submission/solution.py. It implements the
same required contract so the supplied evaluator can compare it directly.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            directory / "harness"
            for directory in Path(__file__).resolve().parents
            if (directory / "harness" / "contract.py").exists()
        )
    ),
)

from contract import Answer, Conflict, Snapshot, Source  # noqa: E402


TOKEN_RE = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?%?(?:[-/]\d[\d,]*(?:\.\d+)?%?)?"
    r"|[a-z]+(?:[-'][a-z0-9]+)*",
    re.I,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do",
    "does", "for", "from", "get", "give", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "tell", "that", "the", "their",
    "them", "there", "to", "was", "what", "when", "where", "which", "with",
    "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Keep important numbers, terms, and percentages as searchable tokens."""
    return [
        token.lower().replace(",", "")
        for token in TOKEN_RE.findall(text or "")
        if token.lower() not in STOP_WORDS
    ]


def expand_query_tokens(tokens: list[str], snapshot: dict) -> list[str]:
    """Add corpus-backed singular forms to bridge ordinary query phrasing.

    This is intentionally conservative: a variant is added only when that
    exact term exists in the indexed snapshot, and the original query term is
    retained. It helps cases such as "APRs" versus a page/table labeled "APR"
    without hard-coding product names or answer values.
    """
    expanded = set(tokens)
    vocabulary = snapshot.get("df", {})
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            candidate = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            candidate = token[:-1]
        else:
            continue
        if vocabulary.get(candidate, 0):
            expanded.add(candidate)
    return list(expanded)


def _is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[\s|:-]+", line))


def _make_chunks(page) -> list[dict]:
    """Split Markdown into readable paragraphs, list entries, and table rows."""
    chunks: list[dict] = []
    headings: list[str] = []
    pending_context: list[str] = []
    block: list[str] = []
    block_is_table = False

    def section_name() -> str:
        return " > ".join(headings)

    def add(body: str, context: str = "") -> None:
        body = body.strip()
        if not body:
            return
        section = section_name()
        if context:
            section = " > ".join(filter(None, (section, context)))
            body = f"{context}: {body}"
        searchable = " ".join((page.title, page.url, section, body))
        tokens = tokenize(searchable)
        if tokens:
            chunks.append({
                "body": body,
                "section": section,
                "title": page.title,
                "url": page.url,
                "tokens": tokens,
            })

    def flush() -> None:
        nonlocal block, block_is_table, pending_context
        if not block:
            return
        context = " > ".join(pending_context)
        if block_is_table:
            rows = [line.strip() for line in block if line.strip() and not _is_table_separator(line)]
            if rows:
                header = rows[0].strip("|").replace("|", " ; ").strip()
                data_rows = [row.strip("|").replace("|", " ; ").strip() for row in rows[1:]]
                # Preserve row-level passages for exact lookups, and also add
                # small adjacent-row windows so multi-value questions can
                # retrieve a useful portion of a table in one evidence item.
                window_size = 6
                if 1 < len(data_rows) <= 20:
                    add(f"Table columns: {header}. Complete table rows: " + " | ".join(data_rows), context)
                elif len(data_rows) > 20:
                    for start in range(0, len(data_rows), window_size):
                        window = data_rows[start:start + window_size]
                        add(f"Table columns: {header}. Rows: " + " | ".join(window), context)
                for values in data_rows:
                    add(f"Table columns: {header}. Row: {values}", context)
            pending_context = []
        else:
            nonempty = [line.strip() for line in block if line.strip()]
            bullets = [line for line in nonempty if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line)]
            # Many crawled pages use standalone plain-text labels immediately
            # before a table instead of Markdown headings (for example,
            # "Credit Cards" then "Platinum Visa"). Carry these labels into
            # the next content block so a row retains its product context.
            is_label = (
                len(nonempty) == 1
                and len(nonempty[0]) <= 90
                and len(tokenize(nonempty[0])) <= 8
                # A short line containing a number or colon is usually a fact,
                # not a heading (e.g. "6-month certificate APY: 3.50%").
                and not re.search(r"\d|:|[.!?]$|https?://|^[-*+]\s", nonempty[0], re.I)
            )
            if is_label:
                pending_context.append(nonempty[0].strip("#* "))
                pending_context = pending_context[-3:]
            elif len(bullets) == len(nonempty) and bullets:
                for bullet in bullets:
                    add(bullet, context)
                pending_context = []
            else:
                add(" ".join(nonempty), context)
                pending_context = []
        block = []
        block_is_table = False

    for raw in page.markdown.splitlines():
        line = raw.strip()
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            pending_context = []
            level = len(heading.group(1))
            headings = headings[: level - 1]
            headings.append(heading.group(2).strip("* "))
            continue
        if not line:
            flush()
            continue
        if "|" in line:
            if block and not block_is_table:
                flush()
            block_is_table = True
            block.append(line)
            continue
        if block_is_table:
            flush()
        block.append(line)

    flush()
    return chunks


class BM25Knowledge:
    """Rank evidence chunks with BM25 inside the requested whole snapshot."""

    def __init__(self):
        self.snapshots: list[dict] = []

    def ingest(self, snapshot: Snapshot) -> None:
        chunks = []
        for page in snapshot.pages:
            chunks.extend(_make_chunks(page))

        document_frequency = Counter()
        total_length = 0
        for chunk in chunks:
            counts = Counter(chunk["tokens"])
            chunk["term_counts"] = counts
            chunk["length"] = len(chunk["tokens"])
            total_length += chunk["length"]
            document_frequency.update(counts.keys())

        self.snapshots.append({
            "date": snapshot.taken_on,
            "snapshot_id": snapshot.snapshot_id,
            "chunks": chunks,
            "df": document_frequency,
            "avgdl": total_length / max(1, len(chunks)),
        })

    def _snapshot_for_date(self, as_of):
        eligible = [item for item in self.snapshots if as_of is None or item["date"] <= as_of]
        return eligible[-1] if eligible else None

    @staticmethod
    def _score(query_tokens, chunk, snapshot, k1=1.5, b=0.75):
        """Okapi BM25: IDF-weighted term frequency, normalized for chunk length."""
        score = 0.0
        n_docs = len(snapshot["chunks"])
        avgdl = snapshot["avgdl"] or 1.0
        counts = chunk["term_counts"]
        for term in set(query_tokens):
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = snapshot["df"].get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            length_norm = tf + k1 * (1.0 - b + b * chunk["length"] / avgdl)
            score += idf * (tf * (k1 + 1.0)) / length_norm
        return score

    def answer(self, question, as_of=None):
        snapshot = self._snapshot_for_date(as_of)
        query_tokens = expand_query_tokens(tokenize(question), snapshot) if snapshot else tokenize(question)
        if snapshot is None or not query_tokens:
            return Answer(text="I don't know.", confident=False)

        best = None
        best_score = 0.0
        for chunk in snapshot["chunks"]:
            score = self._score(query_tokens, chunk, snapshot)
            if score > best_score:
                best = chunk
                best_score = score

        if best is None:
            return Answer(text="I don't know.", confident=False)

        # Return the retrieved evidence as-is; answer synthesis is a separate
        # experiment so we can measure retrieval quality first.
        return Answer(
            text=best["body"][:900],
            sources=[Source(url=best["url"], snapshot_id=snapshot["snapshot_id"])],
        )

    def conflicts(self):
        # Conflict extraction is intentionally held constant for this retrieval test.
        return []


def build():
    return BM25Knowledge()
