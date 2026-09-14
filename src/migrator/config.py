"""Tuning knobs for the migration agent.

Every number here was picked by measuring, not by taste. Where a value is a
judgement call rather than a measurement, the docstring says so explicitly --
that distinction matters when someone else has to retune this for a different
migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HunkBudget:
    """How much source text travels with each unresolved site.

    The agent never sends whole files to the model. It sends a window around
    the call site. `context_lines` is the single most sensitive knob in the
    system: too small and the model cannot see the surrounding `try` block or
    the variable the response is assigned to, so it guesses; too large and you
    have reinvented sending the whole file.
    """

    #: Lines of source kept above the first line of the match.
    context_before: int = 6
    #: Lines of source kept below the last line of the match.
    context_after: int = 6
    #: Hard ceiling on a single hunk. A hunk larger than this is a signal that
    #: the deterministic pass failed to localise the change, so it is flagged
    #: for human review rather than silently truncated.
    max_hunk_tokens: int = 400
    #: Include the enclosing function signature even when it falls outside the
    #: window. Cheap (one line) and removes most "what is `self` here" guessing.
    include_enclosing_def: bool = True


@dataclass(frozen=True)
class ClusterBudget:
    """Controls how unresolved sites are grouped before they reach the model.

    Clustering is the difference between 400 model calls and 9. Two sites join
    the same cluster when their *change shape* fingerprint matches -- same rule
    that failed, same call shape, same surrounding control flow. The model then
    sees one representative and returns one transform.
    """

    #: Below this, clustering costs more in prompt overhead than it saves.
    min_cluster_size: int = 2
    #: Representatives sent per cluster. Two is meaningfully better than one:
    #: a single example lets the model overfit to an incidental detail (a
    #: variable name, a specific URL) and emit a transform that only matches
    #: the representative. Three did not measurably beat two.
    representatives_per_cluster: int = 2
    #: A cluster larger than this is split, so one bad transform cannot
    #: corrupt an unbounded number of files before verification catches it.
    max_cluster_size: int = 40


@dataclass(frozen=True)
class PackBudget:
    """Controls how many independent clusters ride in one request.

    Packing amortises the static prefix. With a ~900 token system prompt, one
    request per cluster means paying 900 tokens nine times. Packing them into
    two requests pays it twice.
    """

    #: Ceiling on the variable (non-prefix) part of a packed request.
    max_payload_tokens: int = 2_400
    #: Ceiling on clusters per request regardless of token count. Beyond this,
    #: response quality degrades -- the model starts dropping items rather than
    #: answering them, which is far worse than a second request.
    max_items_per_request: int = 6


@dataclass(frozen=True)
class ExemplarBudget:
    """Few-shot exemplars retrieved per request.

    The alternative -- a long static rulebook covering every edge case --
    costs ~2,800 tokens on *every* request. Retrieving the two nearest
    previously-approved migrations costs ~260 tokens and performs better,
    because the examples are relevant rather than exhaustive.
    """

    top_k: int = 2
    #: Below this similarity an exemplar is noise. Sending a barely-related
    #: example is worse than sending none: the model pattern-matches on it.
    min_similarity: float = 0.25
    max_exemplar_tokens: int = 200


@dataclass(frozen=True)
class EscalationBudget:
    """Cheap-model-first routing.

    Most residual sites are boring. Running them through a small model and
    escalating only on verification failure is a large saving, but only if
    verification is trustworthy -- see `verify.py`. Escalating on a *silent*
    failure is impossible, which is why verification is syntactic and
    rule-based rather than model-judged.
    """

    cheap_model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"
    #: Attempts on the cheap model before escalating.
    cheap_attempts: int = 1
    #: Attempts on the strong model before giving up and flagging for a human.
    strong_attempts: int = 1


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens. Illustrative list prices, not a quote."""

    cheap_input: float = 0.15
    cheap_output: float = 0.60
    strong_input: float = 2.50
    strong_output: float = 10.00

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model == EscalationBudget.strong_model:
            return (
                input_tokens * self.strong_input + output_tokens * self.strong_output
            ) / 1_000_000
        return (input_tokens * self.cheap_input + output_tokens * self.cheap_output) / 1_000_000


@dataclass(frozen=True)
class Settings:
    hunks: HunkBudget = field(default_factory=HunkBudget)
    clusters: ClusterBudget = field(default_factory=ClusterBudget)
    packing: PackBudget = field(default_factory=PackBudget)
    exemplars: ExemplarBudget = field(default_factory=ExemplarBudget)
    escalation: EscalationBudget = field(default_factory=EscalationBudget)
    pricing: Pricing = field(default_factory=Pricing)

    #: Master switches, used by the ablation benchmark to price each lever.
    #:
    #: Every flag here is read by the pipeline. A config knob that nothing
    #: consumes is worse than no knob: it reads as a feature, benchmarks as a
    #: no-op, and eventually someone tunes it and wonders why nothing changed.
    deterministic_pass: bool = True
    windowing: bool = True
    clustering: bool = True
    packing_enabled: bool = True
    exemplars_enabled: bool = True
    escalation_enabled: bool = True


DEFAULT = Settings()
