"""The transforms a model is allowed to request, and the code that applies them.

This module encodes the central design decision of the whole agent:

    **The model writes the rule. The codemod runs it.**

The obvious design is to have the model return rewritten code, or a unified
diff, for each site. That works, and it is what most "AI migration" demos do,
but it scales badly in three separate ways at once. Output tokens grow with
the number of sites rather than the number of *distinct problems*. Every
returned line is a fresh opportunity to hallucinate an unrelated change. And
there is no artefact left over afterwards -- nothing you can review, diff,
version, or re-run next quarter.

Instead the model receives a cluster of similar sites and returns a small
parameterised instruction, like ``set_timeout(value=30.0)``. Applying it is
deterministic AST work. The consequences are worth spelling out:

* **Output cost becomes O(clusters), not O(sites).** Six clusters cost six
  short answers whether they cover sixty sites or six thousand.
* **The blast radius is inspectable before it lands.** A reviewer reads six
  rules, not four hundred diffs.
* **A malformed or unknown action is rejected, not merged.** The vocabulary
  below is closed; anything outside it fails parsing and the cluster escalates.
  A model cannot invent an edit this layer will apply.

The price is expressiveness: a migration whose sites each need genuinely
bespoke handling cannot be expressed as a handful of rules, and this design
degrades into one cluster per site. That is the honest failure mode, and
``cluster.stats()['compression']`` is the number that tells you it is
happening.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import rules
from .rules import Edit

#: The closed vocabulary. Adding an action here is a deliberate act that comes
#: with a test; a model cannot widen it at runtime.
KNOWN_ACTIONS = {"set_timeout", "map_exception", "rename_symbol"}


class TransformError(ValueError):
    """Raised when a requested transform is malformed or outside the vocabulary."""


@dataclass
class Transform:
    """One parameterised instruction returned by the model."""

    action: str
    params: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.action not in KNOWN_ACTIONS:
            raise TransformError(f"unknown action {self.action!r}")

    def describe(self) -> str:
        args = " ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.action}({args})"


@dataclass
class ApplyReport:
    text: str
    applied: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _last_argument_end(node: ast.Call) -> Optional[tuple]:
    """Position just past the final argument of a call.

    Inserting here rather than before the closing paren is what makes this
    work on multi-line calls with trailing commas. Given::

        requests.get(
            url,
            headers=H,
        )

    inserting before ``)`` means reasoning about the trailing comma and the
    indentation of the closing line. Inserting after ``headers=H`` yields
    ``headers=H, timeout=30.0,`` -- valid, minimally invasive, and identical in
    shape to the single-line case.
    """
    candidates = list(node.args) + [kw.value for kw in node.keywords]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda n: ((n.end_lineno or 0), (n.end_col_offset or 0)),
    )
    if best.end_lineno is None or best.end_col_offset is None:
        return None
    return (best.end_lineno, best.end_col_offset)


def _target_calls(tree: ast.AST) -> List[ast.Call]:
    """Every ``requests.<verb>(...)`` call still present in the tree."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = rules._dotted(node.func)
        if not dotted or not dotted.startswith("requests."):
            continue
        tail = dotted.split(".", 1)[1]
        if tail in rules._HTTP_VERBS:
            found.append(node)
    return found


def _apply_set_timeout(
    text: str,
    transform: Transform,
    line_filter: Optional[Sequence[int]],
) -> ApplyReport:
    """Add an explicit ``timeout=`` and rename the module.

    Only touches calls that genuinely lack a timeout. Re-running the transform
    is therefore a no-op rather than a second ``timeout=`` kwarg, which matters
    because verification failures cause retries and a non-idempotent transform
    turns one retry into a syntax error.
    """
    value = transform.params.get("value")
    if value is None:
        raise TransformError("set_timeout requires a 'value' parameter")
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"set_timeout value must be numeric, got {value!r}") from exc

    tree = ast.parse(text)
    edits: List[Edit] = []
    applied = skipped = 0

    for call in _target_calls(tree):
        if line_filter is not None and call.lineno not in line_filter:
            continue
        if any(kw.arg == "timeout" for kw in call.keywords):
            skipped += 1
            continue
        position = _last_argument_end(call)
        if position is None:
            skipped += 1
            continue
        end_line, end_col = position
        edits.append(
            Edit(
                line=end_line,
                col=end_col,
                end_line=end_line,
                end_col=end_col,
                new_text=f", timeout={value}",
                rule="set_timeout",
            )
        )
        rename = rules._name_span(call.func)
        if rename is not None:
            edits.append(rename)
        applied += 1

    return ApplyReport(text=rules.apply_edits(text, edits), applied=applied, skipped=skipped)


def _apply_map_exception(
    text: str,
    transform: Transform,
    line_filter: Optional[Sequence[int]],
) -> ApplyReport:
    """Rewrite one ``except`` target to its counterpart in the new library."""
    source = transform.params.get("from")
    target = transform.params.get("to")
    if not source or not target:
        raise TransformError("map_exception requires 'from' and 'to' parameters")

    tree = ast.parse(text)
    edits: List[Edit] = []
    applied = skipped = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        if line_filter is not None and node.lineno not in line_filter:
            continue
        elements = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        for element in elements:
            if rules._dotted(element) != source:
                skipped += 1
                continue
            if element.end_lineno != element.lineno:
                skipped += 1
                continue
            edits.append(
                Edit(
                    line=element.lineno,
                    col=element.col_offset,
                    end_line=element.end_lineno or element.lineno,
                    end_col=element.end_col_offset or element.col_offset,
                    new_text=target,
                    rule="map_exception",
                )
            )
            applied += 1

    return ApplyReport(text=rules.apply_edits(text, edits), applied=applied, skipped=skipped)


def _apply_rename_symbol(
    text: str,
    transform: Transform,
    line_filter: Optional[Sequence[int]],
) -> ApplyReport:
    """Rename a dotted symbol, e.g. ``requests.Session`` -> ``httpx.Client``.

    Present mainly as an escape hatch for shapes the other two actions do not
    cover. It is still AST-scoped, so it cannot hit a docstring.
    """
    source = transform.params.get("from")
    target = transform.params.get("to")
    if not source or not target:
        raise TransformError("rename_symbol requires 'from' and 'to' parameters")

    tree = ast.parse(text)
    edits: List[Edit] = []
    applied = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Attribute, ast.Name)):
            continue
        if rules._dotted(node) != source:
            continue
        if line_filter is not None and node.lineno not in line_filter:
            continue
        if node.end_lineno != node.lineno:
            continue
        edits.append(
            Edit(
                line=node.lineno,
                col=node.col_offset,
                end_line=node.end_lineno or node.lineno,
                end_col=node.end_col_offset or node.col_offset,
                new_text=target,
                rule="rename_symbol",
            )
        )
        applied += 1

    return ApplyReport(text=rules.apply_edits(text, edits), applied=applied)


_DISPATCH = {
    "set_timeout": _apply_set_timeout,
    "map_exception": _apply_map_exception,
    "rename_symbol": _apply_rename_symbol,
}


def apply(
    text: str,
    transforms: Sequence[Transform],
    line_filter: Optional[Sequence[int]] = None,
) -> ApplyReport:
    """Apply ``transforms`` in order, re-parsing between each.

    Re-parsing between transforms is not paranoia. Each transform computes AST
    positions against the text it was handed, so running two of them against
    stale positions reproduces exactly the offset-drift bug that
    ``rules.apply_edits`` exists to avoid -- just one level up.

    If any transform produces text that will not parse, the whole batch is
    abandoned and the original is returned. Partial application is the one
    outcome worse than failure, because it leaves a file that is neither
    migrated nor untouched and no longer matches the answer the model gave.
    """
    current = text
    total_applied = total_skipped = 0
    errors: List[str] = []

    for transform in transforms:
        handler = _DISPATCH.get(transform.action)
        if handler is None:
            errors.append(f"no handler for {transform.action}")
            continue
        try:
            report = handler(current, transform, line_filter)
        except TransformError as exc:
            errors.append(str(exc))
            continue
        except SyntaxError as exc:
            errors.append(f"{transform.action}: source no longer parses: {exc}")
            continue

        try:
            ast.parse(report.text)
        except SyntaxError as exc:
            return ApplyReport(
                text=text,
                applied=0,
                skipped=0,
                errors=[f"{transform.describe()} produced invalid syntax: {exc}"],
            )

        current = report.text
        total_applied += report.applied
        total_skipped += report.skipped
        errors.extend(report.errors)

    return ApplyReport(
        text=current,
        applied=total_applied,
        skipped=total_skipped,
        errors=errors,
    )
