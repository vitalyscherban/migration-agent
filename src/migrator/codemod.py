"""Drives the deterministic pass over a file and reports what is left.

Separating this from ``rules.py`` keeps the rules declarative and testable in
isolation. This module owns the ordering, the failure handling, and the
invariant that a file is only ever written back in a state that parses.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List

from . import rules
from .corpus import Shape
from .rules import Finding


@dataclass
class CodemodResult:
    path: str
    original: str
    text: str
    resolved: List[Finding] = field(default_factory=list)
    unresolved: List[Finding] = field(default_factory=list)
    error: str = ""

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def fully_resolved(self) -> bool:
        return not self.unresolved and not self.error

    @property
    def needs_model(self) -> bool:
        return bool(self.unresolved)


def run(path: str, text: str) -> CodemodResult:
    """Apply every mechanical rule to ``text``.

    The post-condition worth noting: if the edited text fails to parse, the
    original is returned untouched and the whole file is escalated to the
    model. A codemod that can produce broken output and still write it is
    worse than no codemod, because it converts a cheap model call into an
    expensive debugging session.
    """
    try:
        findings = rules.scan(text)
    except SyntaxError as exc:
        return CodemodResult(path=path, original=text, text=text, error=f"parse error: {exc}")

    resolved = [f for f in findings if f.resolved]
    unresolved = [f for f in findings if not f.resolved]

    edits = [edit for finding in resolved for edit in finding.edits]
    updated = rules.apply_edits(text, edits) if edits else text

    try:
        ast.parse(updated)
    except SyntaxError as exc:
        return CodemodResult(
            path=path,
            original=text,
            text=text,
            unresolved=findings,
            error=f"codemod produced invalid syntax, reverted: {exc}",
        )

    if not unresolved:
        updated = finalize(updated)

    return CodemodResult(
        path=path,
        original=text,
        text=updated,
        resolved=resolved,
        unresolved=unresolved,
    )


def finalize(text: str) -> str:
    """Swap imports once the file has no ``requests`` call sites left.

    Guarded by a re-scan rather than by the caller's bookkeeping, because the
    caller's bookkeeping is exactly the thing that goes stale after a patch is
    applied.
    """
    remaining = [f for f in rules.scan(text) if not f.resolved]
    if remaining:
        return text
    return rules.rewrite_imports(text)


def summarise(results: List[CodemodResult]) -> dict:
    counts = {shape.value: 0 for shape in Shape}
    for result in results:
        for finding in result.unresolved:
            counts[finding.shape.value] += 1
    return {
        "files": len(results),
        "files_changed": sum(1 for r in results if r.changed),
        "files_fully_resolved": sum(1 for r in results if r.changed and r.fully_resolved),
        "files_needing_model": sum(1 for r in results if r.needs_model),
        "resolved_sites": sum(len(r.resolved) for r in results),
        "unresolved_sites": sum(len(r.unresolved) for r in results),
        "unresolved_by_shape": {k: v for k, v in counts.items() if v},
        "errors": sum(1 for r in results if r.error),
    }
