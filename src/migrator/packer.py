"""Packs clusters into as few requests as the budget allows.

With a ~900 token system prompt, asking about six clusters in six requests
pays that prefix six times. Packing them into two pays it twice. The saving is
mechanical and the code is short, but there are two traps worth naming.

**Packing fights prefix caching.** If the provider caches on a shared leading
span, the *stable* part must come first and the variable part last. That is
why ``prompts.build_request`` returns the system prompt separately and why
exemplars are ordered ahead of cluster payloads: exemplars repeat across
requests, cluster payloads do not.

**Packing degrades answers past a point.** Beyond roughly six items a model
stops answering every question and starts dropping some -- and a dropped
answer is silent. ``max_items_per_request`` exists for that, not for tokens.
``patch.parse`` reports a missing block per unanswered cluster, which is how a
regression here becomes visible instead of becoming a half-migrated codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from . import tokens
from .config import PackBudget


@dataclass
class PackItem:
    """One cluster's rendered question plus its retrieved exemplars."""

    fingerprint: str
    body: str
    exemplars: List[str] = field(default_factory=list)

    @property
    def token_cost(self) -> int:
        return tokens.count(self.body) + tokens.count_all(self.exemplars)


@dataclass
class Request:
    items: List[PackItem] = field(default_factory=list)

    @property
    def fingerprints(self) -> List[str]:
        return [item.fingerprint for item in self.items]

    @property
    def payload_tokens(self) -> int:
        return sum(item.token_cost for item in self.items)

    def exemplars(self) -> List[str]:
        """Deduplicated exemplars across the packed items.

        This is a real saving and an easy one to miss: clusters with similar
        shapes retrieve the same exemplars, so packing four clusters naively
        can ship the same worked example four times in one request.
        """
        seen: List[str] = []
        for item in self.items:
            for exemplar in item.exemplars:
                if exemplar not in seen:
                    seen.append(exemplar)
        return seen

    def clusters_block(self) -> str:
        return "\n\n".join(item.body for item in self.items)


def pack(items: Sequence[PackItem], budget: PackBudget) -> List[Request]:
    """Greedily fill requests up to the token and item ceilings.

    Greedy rather than optimal bin-packing. The optimal solution saves a
    fraction of one request on realistic inputs and costs an algorithm nobody
    will want to debug at 2am.

    An item larger than the whole payload budget still gets its own request
    rather than being dropped or truncated -- the ceiling is there to split
    batches, not to silently discard work.
    """
    requests: List[Request] = []
    current = Request()
    current_cost = 0

    for item in items:
        cost = item.token_cost
        too_many = len(current.items) >= budget.max_items_per_request
        too_big = current_cost + cost > budget.max_payload_tokens

        if current.items and (too_many or too_big):
            requests.append(current)
            current = Request()
            current_cost = 0

        current.items.append(item)
        current_cost += cost

    if current.items:
        requests.append(current)
    return requests
