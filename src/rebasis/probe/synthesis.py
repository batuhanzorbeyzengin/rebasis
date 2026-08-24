"""Synthesised queries — the T2 tier.

T0 has a blind spot that costs users money. Its ground truth is the new model's
own output, so it cannot measure ``upgrade_gain`` — whether the new model is
actually better on this corpus — and without that figure the tool cannot tell
whether bridging beats changing nothing. Measured across six real corpus/model
pairs, T0 recommended bridging in **6 of 6** cases where the independent
measurement said bridging lost ground.

T2 closes that gap without a query log. The trick is where the *judgement*
comes from: a synthesised query is derived from one document, so that document
is its answer **by construction**, independently of either embedding model.
That makes the oracle's recall a real number, and therefore makes
``upgrade_gain`` computable.

**These are not real queries and the report says so.** A title is not how
someone searches, and a corpus whose users type three keywords is not well
represented by its own prose. T2 sits between T0 and T1 for a reason: it can
answer "is the new model better here", and it cannot answer it as well as
asking your users.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["STRATEGIES", "SynthesisResult", "synthesize_queries"]

#: Shortest text worth using as a query. Below this a "query" is usually a
#: fragment of markup or a heading like "Introduction", which matches
#: everything and measures nothing.
MIN_QUERY_CHARS = 15

#: Longest. A query that is most of a document is the T0 proxy again, and
#: carries the same blind spot.
MAX_QUERY_CHARS = 200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


class SynthesisResult:
    """Synthesised queries and the documents they came from."""

    __slots__ = ("queries", "source_positions", "strategy")

    def __init__(self, queries: list[str], source_positions: list[int], strategy: str) -> None:
        self.queries = queries
        #: Position in the sample of the document each query was derived from.
        #: That document is the query's relevant answer, by construction.
        self.source_positions = source_positions
        self.strategy = strategy

    def __len__(self) -> int:
        return len(self.queries)


def _first_line(text: str) -> str:
    """The document's title, where it has one.

    Titles are the closest thing a corpus holds to a question about itself:
    written by a person, independent of any model, and usually a genuine
    description of the content rather than a sample of it.
    """
    for line in text.splitlines():
        stripped = _WHITESPACE.sub(" ", line).strip()
        if stripped:
            return stripped
    return ""


def _lead_sentence(text: str) -> str:
    """The first sentence.

    Weaker than a title and available on far more corpora. A lead sentence
    states what the document is about often enough to be a usable probe.
    """
    flat = _WHITESPACE.sub(" ", text).strip()
    parts = _SENTENCE_END.split(flat, maxsplit=1)
    return parts[0].strip() if parts else ""


def _keywords(text: str, n: int = 6) -> str:
    """The longest distinct words, as a keyword query.

    The closest of the three to how people actually search — short, no
    grammar — and the furthest from the document's own prose, which is what
    makes it a useful contrast with the other two rather than a worse copy.
    """
    seen: dict[str, None] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text):
        seen.setdefault(word.lower(), None)
    ranked = sorted(seen, key=len, reverse=True)[:n]
    return " ".join(ranked)


#: Strategy name to extractor. All three are deterministic and need no model:
#: a synthesis step that itself required an LLM would make the cheap tier the
#: expensive one, and would make the measurement depend on that model's biases.
STRATEGIES: dict[str, Callable[[str], str]] = {
    "title": _first_line,
    "lead": _lead_sentence,
    "keywords": _keywords,
}


def synthesize_queries(
    texts: Sequence[str],
    positions: Sequence[int],
    *,
    strategy: str = "lead",
    limit: int | None = None,
) -> SynthesisResult:
    """Derive one query per document, keeping only usable ones.

    Args:
        texts: Document texts.
        positions: Position of each text within the sample, so the judgement
            can point back at the right record.
        strategy: ``title``, ``lead`` or ``keywords``.
        limit: Stop after this many usable queries.

    Raises:
        ConfigInvalid: When the strategy is not one of :data:`STRATEGIES`.
    """
    from rebasis.errors import ConfigError

    if strategy not in STRATEGIES:
        raise ConfigError(
            f"Unknown query synthesis strategy {strategy!r}.",
            hint=f"Available: {', '.join(sorted(STRATEGIES))}.",
        )

    extract = STRATEGIES[strategy]
    queries: list[str] = []
    sources: list[int] = []
    seen: set[str] = set()

    for text, position in zip(texts, positions, strict=True):
        candidate = extract(text)[:MAX_QUERY_CHARS].strip()
        if len(candidate) < MIN_QUERY_CHARS:
            continue
        # A query that appears twice would have two correct answers, and the
        # judgement says there is one. Dropping duplicates keeps the qrels true.
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(candidate)
        sources.append(int(position))
        if limit is not None and len(queries) >= limit:
            break

    return SynthesisResult(queries, sources, strategy)
