"""A small library of previously-approved migrations, retrieved per request.

The alternative to this module is a long static rulebook: one exhaustive
system prompt covering every edge case the migration might hit. That costs its
full length on *every* request, and it gets worse over time, because the
natural response to a mistake is to append another paragraph.

Retrieving two relevant examples instead costs a fraction and performs better,
for the same reason that a worked example beats a specification. The examples
here are also the natural place for institutional knowledge to accumulate:
when a reviewer corrects a transform, the corrected pair becomes an exemplar
and the next similar cluster gets it for free.

Similarity is lexical overlap over the shape vocabulary, not embeddings. That
is a deliberate scope choice -- the fingerprints are drawn from a small closed
vocabulary of shape names and kwarg names, so there are no synonyms to
resolve, and lexical matching is exact, free, and trivially debuggable. Reach
for embeddings when the corpus of exemplars grows natural-language notes worth
matching on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from . import tokens
from .config import ExemplarBudget


@dataclass(frozen=True)
class Exemplar:
    """One approved cluster answer, ready to be shown as a few-shot example."""

    #: Shape vocabulary this exemplar is indexed on.
    keys: Tuple[str, ...]
    snippet: str
    answer: str

    def render(self) -> str:
        return f"{self.snippet}\n=>\n{self.answer}"

    @property
    def token_cost(self) -> int:
        return tokens.count(self.render())


#: Seeded from migrations a human already reviewed and approved. In a real
#: deployment this is persisted and appended to, not hard-coded.
LIBRARY: List[Exemplar] = [
    Exemplar(
        keys=("ambiguous_timeout", "call", "get", "batch", "in_try"),
        snippet=(
            "requests.get(f\"{BASE_URL}/ledgers/export\", params={\"window\": days},\n"
            "             headers=DEFAULT_HEADERS)   # nightly cron, streams slowly"
        ),
        answer=(
            "ACTION set_timeout value=120.0\n"
            "NOTE batch export path; preserves long-running behaviour without "
            "reintroducing an unbounded wait"
        ),
    ),
    Exemplar(
        keys=("ambiguous_timeout", "call", "get", "interactive"),
        snippet=(
            "requests.get(f\"{BASE_URL}/customers/{cid}\",\n"
            "             headers=DEFAULT_HEADERS)   # called from the request path"
        ),
        answer=(
            "ACTION set_timeout value=10.0\n"
            "NOTE interactive path; a long timeout here holds a worker thread"
        ),
    ),
    Exemplar(
        keys=("ambiguous_retry", "except", "requests.exceptions.ConnectionError", "in_loop"),
        snippet=(
            "except requests.exceptions.ConnectionError:\n"
            "    time.sleep(2 ** attempt)"
        ),
        answer=(
            "ACTION map_exception from=requests.exceptions.ConnectionError "
            "to=httpx.ConnectError\n"
            "NOTE narrowest equivalent; httpx.TransportError would also swallow "
            "read timeouts handled by the next clause"
        ),
    ),
    Exemplar(
        keys=("ambiguous_status", "except", "requests.exceptions.HTTPError"),
        snippet=(
            "except requests.exceptions.HTTPError as exc:\n"
            "    if exc.response.status_code == 404:\n"
            "        return False"
        ),
        answer=(
            "ACTION map_exception from=requests.exceptions.HTTPError "
            "to=httpx.HTTPStatusError\n"
            "NOTE exc.response is always set on HTTPStatusError, so the None "
            "guard in the body becomes dead but harmless"
        ),
    ),
]


def _similarity(query: Sequence[str], keys: Sequence[str]) -> float:
    """Jaccard overlap over shape vocabulary."""
    left, right = set(query), set(keys)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def select(
    query_keys: Sequence[str],
    budget: ExemplarBudget,
    library: Sequence[Exemplar] = (),
) -> List[Exemplar]:
    """Return the best exemplars for a cluster, honouring the token ceiling.

    Returning nothing is a valid and common answer. An exemplar below
    ``min_similarity`` is not a weak signal, it is a misleading one: the model
    treats any example in the prompt as relevant and will pattern-match onto
    it. Empty beats irrelevant.
    """
    pool = list(library) if library else LIBRARY
    scored = [
        (_similarity(query_keys, exemplar.keys), exemplar)
        for exemplar in pool
    ]
    scored = [
        (score, exemplar)
        for score, exemplar in scored
        if score >= budget.min_similarity
        and exemplar.token_cost <= budget.max_exemplar_tokens
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].snippet))
    return [exemplar for _, exemplar in scored[: budget.top_k]]
