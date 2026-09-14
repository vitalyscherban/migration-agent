"""Turns an unresolved finding into the smallest useful slice of source.

This is where "send the file" becomes "send twelve lines". The saving is
large and obvious; the risk is subtle. A window that is too tight produces a
model answer that is confidently wrong, because the model cannot see that the
call it is editing sits inside a ``try`` that already handles the failure, or
that the response is assigned to a variable used three lines later.

Two decisions carry most of the quality here:

1. **The enclosing ``def`` is always included**, even when the window does not
   reach it. It is one line, and without it the model cannot tell a method
   from a function, cannot see the parameter it is being asked to thread a
   timeout through, and cannot read the function name -- which in practice is
   the strongest available signal for whether a call is a batch job or an
   interactive request.

2. **Windows that overlap are merged.** Two findings six lines apart would
   otherwise ship largely the same source twice, and worse, arrive as two
   independent questions whose answers can contradict each other.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import tokens
from .config import HunkBudget
from .corpus import Shape
from .rules import Finding


@dataclass
class Hunk:
    """A windowed slice of one file, carrying everything the model needs."""

    path: str
    #: 1-based inclusive line range of the slice itself.
    start_line: int
    end_line: int
    text: str
    shapes: Tuple[Shape, ...]
    reasons: Tuple[str, ...]
    #: The enclosing ``def`` line, prepended when it fell outside the window.
    enclosing_def: Optional[str] = None
    #: Set when the hunk exceeded ``max_hunk_tokens`` and was flagged instead
    #: of truncated.
    oversized: bool = False
    signature: Tuple[str, ...] = ()
    #: Exact source lines of the findings in this hunk.
    #:
    #: Kept separate from ``start_line``/``end_line`` because the window
    #: includes padding, and a transform must land on the sites only. Using the
    #: window as the target range would let a rule aimed at one call also hit
    #: an unrelated call that merely happened to fall within six lines of it.
    site_lines: Tuple[int, ...] = ()

    @property
    def token_cost(self) -> int:
        return tokens.count(self.render())

    def render(self) -> str:
        """Format for the prompt, with line numbers.

        Line numbers are not decoration. The model returns edits addressed by
        line, so numbering the input is what makes a patch-shaped response
        possible at all -- and patch-shaped responses are the entire output
        token saving.
        """
        header = f"--- {self.path}:{self.start_line}-{self.end_line}"
        body_lines = self.text.splitlines()
        numbered = [
            f"{self.start_line + offset:>4} | {line}"
            for offset, line in enumerate(body_lines)
        ]
        parts = [header]
        if self.enclosing_def:
            parts.append(f"     | (enclosing) {self.enclosing_def.strip()}")
        parts.extend(numbered)
        return "\n".join(parts)


def _def_lines(tree: ast.AST) -> Dict[int, Tuple[int, str]]:
    """Map every line inside a function body to ``(def_lineno, name)``.

    The line number is part of the value, not just the name, so two functions
    that happen to share a name in different scopes stay distinct as merge
    keys.
    """
    spans: Dict[int, Tuple[int, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            for line in range(node.lineno, end + 1):
                # Inner functions win: the nearest enclosing def is the useful
                # one, and ast.walk yields outer nodes first.
                spans[line] = (node.lineno, node.name)
    return spans


def _merge_spans(
    spans: Sequence[Tuple[int, int, Finding]],
    keys: Sequence[str],
) -> List[Tuple[int, int, List[Finding]]]:
    """Coalesce overlapping or adjacent windows *within the same function*.

    Adjacent -- not just overlapping -- because two windows separated by a
    single line would otherwise produce two hunks whose rendered forms are
    almost identical, and the join costs one line while removing a whole
    duplicated payload plus a whole opportunity for contradictory answers.

    The ``keys`` guard is the subtle half, and it was added after a test found
    the bug. Merging purely on line proximity will happily join the last call
    of one function to the first call of the next, because those lines are
    adjacent. The merged hunk then carries sites from two different functions,
    lands in one cluster, and receives *one* answer.

    For most shapes that is harmless. For the missing-timeout shape it is a
    silent correctness bug: a nightly export and an interactive page-load
    handler sitting next to each other would be merged, and whichever answer
    came back -- 120 seconds or 10 -- would be wrong for one of them. Keying
    the merge on the enclosing function keeps sites together only when they
    genuinely share context.
    """
    if not spans:
        return []
    ordered = sorted(
        zip(spans, keys), key=lambda item: (item[0][0], item[0][1])
    )
    merged: List[Tuple[int, int, List[Finding]]] = []
    (start, end, first), current_key = ordered[0]
    group = [first]
    for (next_start, next_end, finding), key in ordered[1:]:
        if next_start <= end + 1 and key == current_key:
            end = max(end, next_end)
            group.append(finding)
        else:
            merged.append((start, end, group))
            start, end, group = next_start, next_end, [finding]
            current_key = key
    merged.append((start, end, group))
    return merged


def extract(
    path: str,
    text: str,
    findings: Sequence[Finding],
    budget: HunkBudget,
) -> List[Hunk]:
    """Build merged, context-padded hunks for ``findings``."""
    if not findings:
        return []

    lines = text.splitlines()
    total = len(lines)
    # Built unconditionally: even when the enclosing def is not shown to the
    # model, it is still needed as the merge key that keeps sites from
    # different functions in different hunks.
    try:
        def_map = _def_lines(ast.parse(text))
    except SyntaxError:
        def_map = {}

    spans = [
        (
            max(1, f.line - budget.context_before),
            min(total, f.end_line + budget.context_after),
            f,
        )
        for f in findings
    ]
    keys = [str(def_map.get(f.line, (0, "<module>"))[0]) for f in findings]

    hunks: List[Hunk] = []
    for start, end, group in _merge_spans(spans, keys):
        body = "\n".join(lines[start - 1 : end])
        enclosing = None
        if budget.include_enclosing_def:
            entry = def_map.get(group[0].line)
            enclosing = f"def {entry[1]}(...)" if entry else None
        # Suppress the enclosing-def line when the window already shows it,
        # otherwise the model sees the signature twice and occasionally
        # "helpfully" emits an edit against the duplicate.
        if enclosing and any(
            line.lstrip().startswith("def ") for line in lines[start - 1 : end]
        ):
            enclosing = None

        signature: Tuple[str, ...] = ()
        for finding in group:
            signature = signature + finding.signature

        hunk = Hunk(
            path=path,
            start_line=start,
            end_line=end,
            text=body,
            shapes=tuple(f.shape for f in group),
            reasons=tuple(dict.fromkeys(f.reason for f in group if f.reason)),
            enclosing_def=enclosing,
            signature=signature,
            site_lines=tuple(sorted(f.line for f in group)),
        )
        if hunk.token_cost > budget.max_hunk_tokens:
            # Flag, do not truncate. An oversized hunk means the deterministic
            # pass failed to localise the change; truncating it would send the
            # model a question it cannot answer and then apply the guess.
            hunk.oversized = True
        hunks.append(hunk)

    return hunks
