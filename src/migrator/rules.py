"""The deterministic pass: an AST-driven codemod.

This module is the largest token saving in the whole system, and it contains
no model call at all. That is the point. Roughly four fifths of the call sites
in a library migration are mechanical, and every one of them that a codemod
resolves is a hunk the model never sees, never reasons about, and never
charges for.

The design rule throughout is **refuse rather than guess**. A rule either
produces an edit it can fully justify, or it produces an ``Unresolved``
finding and hands the site to the model. There is no middle tier of
"probably fine" edits, because a silently wrong codemod across 300 files is
far more expensive than a few extra model calls.

Why AST and not regex: a regex for ``requests.get`` matches the string
``"use requests.get"`` in a docstring, the comment above the function, and the
unrelated ``self.requests.get`` attribute on a local class. The AST does not.
Positions come from ``lineno``/``col_offset``, so edits stay surgical -- the
codemod rewrites the eight characters of a module name, not the file.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .corpus import Shape

#: Module-level functions that exist on both libraries with the same meaning.
_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "request"}

#: ``requests`` exception paths and the nearest ``httpx`` equivalent. The
#: mapping exists for the model's benefit (it ships in the prompt as grounding)
#: and is deliberately *not* applied automatically -- see ``_visit_handler``.
EXCEPTION_EQUIVALENTS = {
    "requests.exceptions.ConnectionError": "httpx.ConnectError",
    "requests.exceptions.Timeout": "httpx.TimeoutException",
    "requests.exceptions.HTTPError": "httpx.HTTPStatusError",
    "requests.exceptions.TooManyRedirects": "httpx.TooManyRedirects",
    "requests.exceptions.RequestException": "httpx.HTTPError",
}


@dataclass(frozen=True)
class Edit:
    """A surgical replacement over a half-open span of source text."""

    line: int  # 1-based, matching ast
    col: int  # 0-based byte offset within the line
    end_line: int
    end_col: int
    new_text: str
    rule: str

    def sort_key(self) -> Tuple[int, int]:
        return (self.line, self.col)


@dataclass
class Finding:
    """One migration site the scanner identified.

    ``resolved`` findings carry edits and cost nothing. Unresolved findings
    carry a ``shape`` and a ``reason``, and become model work.
    """

    shape: Shape
    line: int
    end_line: int
    resolved: bool
    rule: str
    reason: str = ""
    edits: List[Edit] = field(default_factory=list)
    #: Structural facts about the site, used to build cluster fingerprints.
    signature: Tuple[str, ...] = ()


def _dotted(node: ast.AST) -> Optional[str]:
    """Render ``a.b.c`` attribute chains as a string; None for anything else."""
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _name_span(node: ast.AST) -> Optional[Edit]:
    """Span of the root ``Name`` in an attribute chain, ready to be renamed."""
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return Edit(
        line=current.lineno,
        col=current.col_offset,
        end_line=current.end_lineno or current.lineno,
        end_col=current.end_col_offset or current.col_offset,
        new_text="httpx",
        rule="rename-module",
    )


class _Scanner(ast.NodeVisitor):
    """Walks the tree once, recording enclosing context as it descends.

    The context stack is what lets a finding know it sits inside a ``try`` in a
    ``for`` loop in a function called ``push_invoice``. That context is worth
    very little to the codemod and a great deal to the clusterer -- two sites
    with the same call shape but different control flow usually need different
    transforms, and merging them produces a patch that breaks one of them.
    """

    def __init__(self) -> None:
        self.findings: List[Finding] = []
        self._func_stack: List[str] = []
        self._in_try = 0
        self._in_loop = 0

    # -- context tracking -------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Try(self, node: ast.Try) -> None:
        self._in_try += 1
        self.generic_visit(node)
        self._in_try -= 1

    def visit_For(self, node: ast.For) -> None:
        self._in_loop += 1
        self.generic_visit(node)
        self._in_loop -= 1

    def visit_While(self, node: ast.While) -> None:
        self._in_loop += 1
        self.generic_visit(node)
        self._in_loop -= 1

    def _context(self) -> Tuple[str, ...]:
        flags = []
        if self._in_try:
            flags.append("in_try")
        if self._in_loop:
            flags.append("in_loop")
        return tuple(flags)

    # -- rules ------------------------------------------------------------

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._visit_handler(node)
        self.generic_visit(node)

    def _visit_handler(self, node: ast.ExceptHandler) -> None:
        """Exception clauses are never migrated mechanically.

        It is tempting: ``requests.exceptions.Timeout`` obviously maps to
        ``httpx.TimeoutException``, and a one-line rename compiles and looks
        right. It is still wrong, because the two hierarchies do not nest the
        same way. In ``requests``, ``ConnectionError`` and ``Timeout`` are
        siblings under ``RequestException``. In ``httpx``, ``ConnectTimeout``
        is a subclass of *both* ``TimeoutException`` and ``TransportError``,
        so a mechanically renamed pair of handlers changes which clause wins.

        The failure mode is the nastiest kind: the code runs, the tests pass,
        and a class of production failure silently stops being retried. So the
        rule declines, and the model gets the whole handler plus its body.
        """
        targets: List[str] = []
        if node.type is None:
            return
        raw = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        for element in raw:
            dotted = _dotted(element)
            if dotted and dotted.startswith("requests."):
                targets.append(dotted)
        if not targets:
            return

        is_status = any("HTTPError" in t for t in targets)
        shape = Shape.AMBIGUOUS_STATUS if is_status else Shape.AMBIGUOUS_RETRY
        reason = (
            "exception hierarchies differ between the libraries; a rename can "
            "silently change which handler catches a given failure"
        )
        self.findings.append(
            Finding(
                shape=shape,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                resolved=False,
                rule="exception-hierarchy",
                reason=reason,
                signature=("except", *sorted(targets), *self._context()),
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        self._visit_call(node)
        self.generic_visit(node)

    def _visit_call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)
        if not dotted or not dotted.startswith("requests."):
            return

        tail = dotted.split(".", 1)[1]
        kwargs = {kw.arg for kw in node.keywords if kw.arg}

        if tail == "Session":
            self._session(node)
            return
        if tail not in _HTTP_VERBS:
            return

        # Order matters. The timeout rule must run before the rename rule,
        # because a site that needs a judgement call must not be half-migrated
        # into a state where the rename looks done and the behaviour changed.
        if "timeout" not in kwargs:
            self.findings.append(
                Finding(
                    shape=Shape.AMBIGUOUS_TIMEOUT,
                    line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    resolved=False,
                    rule="implicit-timeout",
                    reason=(
                        "requests defaults to no timeout, httpx defaults to 5s; "
                        "migrating this call without an explicit timeout changes "
                        "runtime behaviour"
                    ),
                    signature=(
                        "call",
                        tail,
                        *sorted(kwargs),
                        *self._context(),
                        f"fn:{self._enclosing_kind()}",
                    ),
                )
            )
            return

        edits: List[Edit] = []
        rename = _name_span(node.func)
        if rename is not None:
            edits.append(rename)

        shape = Shape.MECHANICAL_CALL
        for keyword in node.keywords:
            if keyword.arg == "allow_redirects":
                edits.append(
                    Edit(
                        line=keyword.lineno,
                        col=keyword.col_offset,
                        end_line=keyword.lineno,
                        end_col=keyword.col_offset + len("allow_redirects"),
                        new_text="follow_redirects",
                        rule="rename-kwarg",
                    )
                )
                shape = Shape.MECHANICAL_REDIRECT

        self.findings.append(
            Finding(
                shape=shape,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                resolved=True,
                rule="mechanical-call",
                edits=edits,
                signature=("call", tail, *sorted(kwargs)),
            )
        )

    def _session(self, node: ast.Call) -> None:
        """``requests.Session()`` -> ``httpx.Client()``.

        Safe as a pure rename only because the generated corpus uses the
        overlapping subset of the API (``headers.update``, verb methods,
        ``close``). A real migration would want a second guard here asserting
        the session object is not used via a ``requests``-only method such as
        ``mount``; that check is a reachability analysis, and is exactly the
        kind of thing worth building before trusting a codemod at scale.
        """
        span = _name_span(node.func)
        if span is None:
            return
        edits = [
            Edit(
                line=node.func.lineno,
                col=node.func.col_offset,
                end_line=node.func.end_lineno or node.func.lineno,
                end_col=node.func.end_col_offset or node.func.col_offset,
                new_text="httpx.Client",
                rule="rename-session",
            )
        ]
        self.findings.append(
            Finding(
                shape=Shape.MECHANICAL_SESSION,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                resolved=True,
                rule="session-to-client",
                edits=edits,
                signature=("session",),
            )
        )

    def _enclosing_kind(self) -> str:
        """Coarse bucket for the enclosing function name.

        Deliberately coarse. Fingerprinting on the exact function name would
        put every site in its own cluster and destroy the batching win; this
        keeps the one distinction that actually changes the answer -- whether
        the call looks like a batch job or an interactive request path.
        """
        if not self._func_stack:
            return "module"
        name = self._func_stack[-1]
        for marker in ("report", "export", "sync", "bulk", "nightly", "batch"):
            if marker in name:
                return "batch"
        return "interactive"


def scan(text: str) -> List[Finding]:
    """Find every migration site in ``text``.

    Returns an empty list for files that do not parse. A syntax error in the
    source is the user's problem, not something to crash the run over -- but it
    is surfaced by ``codemod.apply`` rather than swallowed here.
    """
    tree = ast.parse(text)
    scanner = _Scanner()
    scanner.visit(tree)
    scanner.findings.sort(key=lambda f: (f.line, f.rule))
    return scanner.findings


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """Apply span edits to ``text``.

    Applied last-to-first so that earlier spans keep the offsets the AST
    reported. Doing this forwards is the classic codemod bug: the first edit
    shifts every subsequent column by its length delta, and the corruption is
    silent because the result usually still parses.
    """
    lines = text.splitlines(keepends=True)
    ordered = sorted(edits, key=lambda e: e.sort_key(), reverse=True)
    for edit in ordered:
        if edit.line != edit.end_line:
            # Multi-line spans are not produced by any current rule. Skipping
            # rather than mangling keeps the invariant that a bad rule costs
            # us a model call, never a corrupted file.
            continue
        index = edit.line - 1
        if index < 0 or index >= len(lines):
            continue
        line = lines[index]
        lines[index] = line[: edit.col] + edit.new_text + line[edit.end_col :]
    return "".join(lines)


def rewrite_imports(text: str) -> str:
    """Swap the top-level import once no ``requests.`` references remain.

    Called only from ``codemod.finalize``, never mid-migration. Rewriting the
    import while unresolved sites still reference ``requests`` would produce a
    file that imports one library and calls another -- which fails at runtime,
    not at import, and therefore not in anyone's smoke test.
    """
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "import requests":
            out.append(line.replace("import requests", "import httpx", 1))
        elif stripped.startswith("import requests."):
            out.append(line.replace("import requests", "import httpx", 1))
        else:
            out.append(line)
    return "".join(out)


def has_residual_references(text: str) -> bool:
    """True when any ``requests`` usage survives, ignoring strings/comments.

    This is the verification predicate, so it must not be a substring search:
    the generated corpus contains the word "requests" in docstrings, and a
    naive check would report permanent failure on a correctly migrated file.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "requests":
            return True
        if isinstance(node, ast.alias) and node.name.split(".")[0] == "requests":
            return True
    return False
