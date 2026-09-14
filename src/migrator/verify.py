"""Post-migration verification.

Verification is what makes cheap-model-first routing safe. Escalating on
failure only helps if failures are actually detected, so every check here is
syntactic or structural -- things that are *true or false about the file*, not
things a model judges. A model-graded check would mean trusting a cheap model
to notice its own mistake, which is precisely the thing it is bad at.

The checks are ordered cheapest-first and short-circuit, because a file that
does not parse will fail every subsequent check in a confusing way.

What this deliberately does not do is run the migrated code. Executing a
migrated codebase is the real acceptance test, and in a live deployment this
module would shell out to the project's own test suite per touched package.
That is out of scope here for the obvious reason that the corpus is synthetic
and has no tests -- and pretending otherwise would be the dishonest kind of
green checkmark.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List

from . import rules


@dataclass
class VerifyResult:
    path: str
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def verify(path: str, text: str) -> VerifyResult:
    """Check that ``text`` is a complete, syntactically valid migration."""
    result = VerifyResult(path=path)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        result.failures.append(f"does not parse: {exc}")
        return result

    if rules.has_residual_references(text):
        result.failures.append("still references `requests` after migration")

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "allow_redirects":
            result.failures.append(
                f"line {node.lineno}: `allow_redirects` is not an httpx keyword"
            )

    result.warnings.extend(_timeout_warnings(tree))
    return result


def _timeout_warnings(tree: ast.AST) -> List[str]:
    """Flag httpx calls with no explicit timeout.

    A warning rather than a failure, and the distinction is deliberate. Once
    migrated, such a call inherits httpx's 5 second default, so it is bounded
    and safe -- just possibly not what the author intended. Treating it as a
    failure would send correctly-migrated files into the escalation path and
    burn strong-model tokens on a non-problem, which is how a verification
    layer quietly becomes the most expensive component in the system.
    """
    warnings: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = rules._dotted(node.func)
        if not dotted or not dotted.startswith("httpx."):
            continue
        if dotted.split(".", 1)[1] not in rules._HTTP_VERBS:
            continue
        if not any(kw.arg == "timeout" for kw in node.keywords):
            warnings.append(f"line {node.lineno}: relies on the httpx 5s default timeout")
    return warnings


def verify_all(files: dict) -> List[VerifyResult]:
    return [verify(path, text) for path, text in sorted(files.items())]
