"""Model backends, plus the offline stand-in that makes this repo runnable.

``OfflineModel`` is not a mock in the usual sense. It parses the prompt it is
given and derives its answer from the cluster shape and the representative
source in front of it, exactly as the real thing has to. That matters for two
reasons:

* The **plumbing is genuinely exercised**. Prompts get built, packed, parsed,
  applied and verified end to end in CI with no API key and no network. A mock
  that returned a canned string would let a prompt-format regression through.
* The **token accounting is real**. Every number this repo reports is measured
  with ``tiktoken`` over the actual bytes that would have been sent.

What it cannot tell you is answer *quality* on genuinely novel code -- it
resolves the shapes this migration defines and nothing else. Treat the token
figures as measurements and the migration decisions as a fixture.

Prompt caching is modelled explicitly rather than assumed. Providers bill
cached prefix tokens at a discount, and that discount only materialises if the
prefix is byte-identical across requests, which is a property of
``prompts.py`` that is easy to break. ``prefix_cached_tokens`` makes a
regression in it visible as a number instead of a surprise invoice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from . import tokens
from .config import EscalationBudget, Pricing

_BATCH_MARKERS = ("report", "export", "sync", "bulk", "nightly", "batch", "cron")

#: Timeouts the offline planner chooses. Separated by path kind because that
#: is the judgement the codemod could not make -- see corpus.py.
_BATCH_TIMEOUT = "120.0"
_INTERACTIVE_TIMEOUT = "10.0"

_EXCEPTION_MAP = {
    "requests.exceptions.ConnectionError": "httpx.ConnectError",
    "requests.exceptions.Timeout": "httpx.TimeoutException",
    "requests.exceptions.HTTPError": "httpx.HTTPStatusError",
    "requests.exceptions.TooManyRedirects": "httpx.TooManyRedirects",
    "requests.exceptions.RequestException": "httpx.HTTPError",
}

_CLUSTER_HEAD_RE = re.compile(r"^CLUSTER\s+(\S+)\s*$", re.MULTILINE)


@dataclass
class Usage:
    """Token accounting for one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    #: Portion of ``input_tokens`` served from the provider's prefix cache.
    prefix_cached_tokens: int = 0
    model: str = ""

    @property
    def billable_input(self) -> int:
        return self.input_tokens - self.prefix_cached_tokens


@dataclass
class ModelResponse:
    text: str
    usage: Usage


class Model(Protocol):
    name: str

    def complete(self, system: str, user: str) -> ModelResponse:  # pragma: no cover
        ...


def _split_clusters(user: str) -> List[str]:
    """Split a packed user message back into per-cluster blocks."""
    matches = list(_CLUSTER_HEAD_RE.finditer(user))
    blocks: List[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(user)
        blocks.append(user[match.start() : end])
    return blocks


def _looks_like_batch(block: str) -> bool:
    """Decide batch vs interactive from the surrounding source.

    This is the judgement the whole ambiguous-timeout shape exists to force,
    and it is genuinely a *reading* task: the signal lives in the function
    name, the docstring and the comments, not in the call expression. Which is
    also why the hunk window includes the enclosing ``def`` -- strip that and
    this decision degrades to a coin flip.
    """
    lowered = block.lower()
    return any(marker in lowered for marker in _BATCH_MARKERS)


class OfflineModel:
    """Deterministic planner used by tests, benchmarks and the default CLI run."""

    def __init__(self, name: str = "offline-planner") -> None:
        self.name = name
        self.calls = 0
        #: Prefixes already seen, so repeat requests can be billed as cached.
        self._seen_prefixes: Dict[str, int] = {}

    def complete(self, system: str, user: str) -> ModelResponse:
        self.calls += 1
        answer = self._plan(user)

        system_tokens = tokens.count(system)
        cached = system_tokens if system in self._seen_prefixes else 0
        self._seen_prefixes[system] = self._seen_prefixes.get(system, 0) + 1

        usage = Usage(
            input_tokens=system_tokens + tokens.count(user),
            output_tokens=tokens.count(answer),
            prefix_cached_tokens=cached,
            model=self.name,
        )
        return ModelResponse(text=answer, usage=usage)

    def _plan(self, user: str) -> str:
        blocks = []
        for block in _split_clusters(user):
            fingerprint = _CLUSTER_HEAD_RE.search(block).group(1)
            blocks.append(self._plan_cluster(fingerprint, block))
        return "\n\n".join(blocks)

    def _plan_cluster(self, fingerprint: str, block: str) -> str:
        lines = [f"CLUSTER {fingerprint}"]
        notes: List[str] = []

        if "ambiguous_timeout" in block:
            if _looks_like_batch(block):
                lines.append(f"ACTION set_timeout value={_BATCH_TIMEOUT}")
                notes.append(
                    "batch path; long but bounded, replacing an unbounded wait"
                )
            else:
                lines.append(f"ACTION set_timeout value={_INTERACTIVE_TIMEOUT}")
                notes.append("interactive path; bounded to protect the worker pool")

        # Order the exception mappings deterministically and narrowest-first, so
        # a cluster containing both ConnectionError and Timeout always produces
        # the same two ACTION lines in the same order.
        for source in sorted(_EXCEPTION_MAP):
            if source in block:
                lines.append(
                    f"ACTION map_exception from={source} to={_EXCEPTION_MAP[source]}"
                )
                notes.append("narrowest equivalent; clause order preserved")

        if len(lines) == 1:
            lines.append("ESCALATE no known action covers this shape")
            return "\n".join(lines)

        lines.append(f"NOTE {notes[0]}")
        return "\n".join(lines)


class ScriptedModel:
    """Returns canned responses in order. For tests that need a specific reply."""

    def __init__(self, responses: List[str], name: str = "scripted") -> None:
        self.name = name
        self._responses = list(responses)
        self.calls = 0
        self.seen: List[str] = []

    def complete(self, system: str, user: str) -> ModelResponse:
        self.calls += 1
        self.seen.append(user)
        text = self._responses.pop(0) if self._responses else ""
        return ModelResponse(
            text=text,
            usage=Usage(
                input_tokens=tokens.count(system) + tokens.count(user),
                output_tokens=tokens.count(text),
                model=self.name,
            ),
        )


@dataclass
class Ledger:
    """Running cost across a migration run."""

    pricing: Pricing = field(default_factory=Pricing)
    usages: List[Usage] = field(default_factory=list)

    def record(self, usage: Usage) -> None:
        self.usages.append(usage)

    @property
    def requests(self) -> int:
        return len(self.usages)

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usages)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usages)

    @property
    def cached_tokens(self) -> int:
        return sum(u.prefix_cached_tokens for u in self.usages)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, strong_model: Optional[str] = None) -> float:
        """Total USD, billing cached prefix tokens at half rate.

        Half is the common provider discount for a cache read. It is a
        constant here rather than a config knob because getting it wrong in
        either direction only scales the dollar figures, while the token counts
        -- the thing this repo is actually measuring -- are unaffected.
        """
        strong = strong_model or EscalationBudget.strong_model
        total = 0.0
        for usage in self.usages:
            full = self.pricing.cost(usage.model, usage.billable_input, usage.output_tokens)
            discounted = (
                self.pricing.cost(usage.model, usage.prefix_cached_tokens, 0) * 0.5
            )
            total += full + discounted
        return total

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_prefix_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost(), 4),
        }
