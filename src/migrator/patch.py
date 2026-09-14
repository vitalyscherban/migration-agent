"""Parses the model's response into transforms.

The parser is deliberately strict. It is the only thing standing between a
free-text generation and an automated rewrite of hundreds of files, so its job
is to reject cleanly rather than to interpret charitably.

Concretely, it refuses to:

* accept an action outside the closed vocabulary in ``transforms.py``,
* guess at a malformed ``key=value`` pair,
* return transforms for a cluster fingerprint that was never asked about.

That last one is the interesting guard. A model that is packed with six
clusters occasionally answers about a *seventh* -- either hallucinating a
fingerprint or repeating one from an exemplar. Without the check those
transforms get applied to whatever cluster happens to hold that key. With it,
they are dropped and recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from .transforms import Transform, TransformError

_CLUSTER_RE = re.compile(r"^CLUSTER\s+(\S+)\s*$")
_ACTION_RE = re.compile(r"^ACTION\s+(\w+)\s*(.*)$")
_ESCALATE_RE = re.compile(r"^ESCALATE\s+(.*)$")
_NOTE_RE = re.compile(r"^NOTE\s+(.*)$")
_KV_RE = re.compile(r"(\w+)=(\S+)")


@dataclass
class ClusterAnswer:
    fingerprint: str
    transforms: List[Transform] = field(default_factory=list)
    note: str = ""
    escalate: str = ""

    @property
    def is_escalation(self) -> bool:
        return bool(self.escalate)


@dataclass
class ParseResult:
    answers: Dict[str, ClusterAnswer] = field(default_factory=dict)
    #: Problems worth surfacing but not worth crashing over.
    warnings: List[str] = field(default_factory=list)

    def for_cluster(self, fingerprint: str) -> ClusterAnswer:
        return self.answers.get(fingerprint, ClusterAnswer(fingerprint=fingerprint))


def parse(response: str, expected: Sequence[str]) -> ParseResult:
    """Parse ``response``, keeping only blocks for ``expected`` fingerprints."""
    allowed: Set[str] = set(expected)
    result = ParseResult()
    current: ClusterAnswer | None = None

    def flush() -> None:
        if current is None:
            return
        if current.fingerprint not in allowed:
            result.warnings.append(
                f"dropped block for unrequested cluster {current.fingerprint!r}"
            )
            return
        result.answers[current.fingerprint] = current

    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue

        match = _CLUSTER_RE.match(line)
        if match:
            flush()
            current = ClusterAnswer(fingerprint=match.group(1))
            continue

        if current is None:
            continue

        match = _ESCALATE_RE.match(line)
        if match:
            current.escalate = match.group(1).strip()
            continue

        match = _NOTE_RE.match(line)
        if match:
            current.note = match.group(1).strip()
            continue

        match = _ACTION_RE.match(line)
        if match:
            name, rest = match.group(1), match.group(2)
            params = dict(_KV_RE.findall(rest))
            # A value containing whitespace would be silently clipped by the
            # regex, so compare what was consumed against what was offered.
            consumed = " ".join(f"{k}={v}" for k, v in params.items())
            if rest.strip() and len(consumed.split()) != len(rest.split()):
                result.warnings.append(f"unparsed tokens in action line: {line!r}")
            try:
                current.transforms.append(Transform(action=name, params=params))
            except TransformError as exc:
                result.warnings.append(f"rejected action in {current.fingerprint}: {exc}")
            continue

        result.warnings.append(f"ignored unrecognised line: {line!r}")

    flush()

    for fingerprint in allowed:
        if fingerprint not in result.answers:
            result.warnings.append(f"no answer returned for cluster {fingerprint}")

    return result
