"""Tests for the deterministic pass.

The theme: a codemod's dangerous failures are the silent ones. Most of these
tests exist to pin behaviour that would otherwise break without any visible
error -- a match inside a docstring, an edit applied at a stale offset, an
import swapped while callers still reference the old library.
"""

from __future__ import annotations

import ast

import pytest
from conftest import AMBIGUOUS_SOURCE, MECHANICAL_SOURCE

from migrator import codemod, rules
from migrator.corpus import Shape


def test_docstring_and_comment_mentions_are_not_matched():
    """The single most important reason this is an AST pass, not a regex.

    MECHANICAL_SOURCE mentions `requests.get` in a docstring and
    `requests.post` in a comment. A regex-based codemod rewrites both, which
    corrupts documentation and, worse, makes the diff look plausible.
    """
    findings = rules.scan(MECHANICAL_SOURCE)
    assert len(findings) == 1
    assert findings[0].shape is Shape.MECHANICAL_REDIRECT

    migrated = codemod.run("x.py", MECHANICAL_SOURCE).text
    assert "Historically this used requests.get" in migrated
    assert "# requests.post here would be wrong" in migrated


def test_allow_redirects_renamed_and_module_swapped():
    result = codemod.run("x.py", MECHANICAL_SOURCE)
    assert result.fully_resolved
    assert "httpx.get(" in result.text
    assert "follow_redirects=False" in result.text
    assert "allow_redirects" not in result.text
    assert "import httpx" in result.text


def test_two_edits_on_one_call_do_not_corrupt_offsets():
    """Applying edits left-to-right silently shifts every later column.

    This call needs both a module rename (8 chars -> 5) and a kwarg rename
    (15 chars -> 16). Applied forwards, the second edit lands in the wrong
    place and usually still parses -- which is why it needs a test rather than
    a crash to catch it.
    """
    source = 'import requests\nx = requests.head("u", allow_redirects=True, timeout=5)\n'
    migrated = codemod.run("x.py", source).text
    assert 'x = httpx.head("u", follow_redirects=True, timeout=5)' in migrated
    ast.parse(migrated)


def test_call_without_timeout_is_declined_not_renamed():
    """A half-migrated ambiguous site is the worst possible outcome.

    If the rename fired but the timeout decision did not, the file would look
    migrated, compile, pass review, and silently acquire a 5 second timeout on
    a path that previously had none.
    """
    findings = rules.scan(AMBIGUOUS_SOURCE)
    timeouts = [f for f in findings if f.shape is Shape.AMBIGUOUS_TIMEOUT]
    assert len(timeouts) == 1
    assert not timeouts[0].resolved

    result = codemod.run("x.py", AMBIGUOUS_SOURCE)
    assert 'requests.get("https://x.internal/ledger"' in result.text


def test_exception_clauses_are_always_declined():
    findings = [f for f in rules.scan(AMBIGUOUS_SOURCE) if f.rule == "exception-hierarchy"]
    assert len(findings) == 2
    assert all(not f.resolved for f in findings)
    assert all("hierarch" in f.reason for f in findings)


def test_import_is_not_rewritten_while_unresolved_sites_remain():
    """Otherwise the file imports httpx and calls requests.

    That combination fails at call time, not at import time, so it survives
    any smoke test that merely imports the module.
    """
    result = codemod.run("x.py", AMBIGUOUS_SOURCE)
    assert result.needs_model
    assert "import requests" in result.text
    assert "import httpx" not in result.text


def test_import_is_rewritten_once_the_file_is_clean():
    result = codemod.run("x.py", MECHANICAL_SOURCE)
    assert result.fully_resolved
    assert "import httpx" in result.text
    assert "import requests" not in result.text


def test_unparseable_file_is_reported_not_crashed():
    result = codemod.run("bad.py", "def broken(:\n    pass\n")
    assert result.error
    assert result.text == result.original


def test_residual_check_ignores_docstrings():
    """The verification predicate must not be a substring search.

    A correctly migrated file whose docstring says "ported from requests"
    would otherwise be reported as permanently unmigrated, sending it into an
    escalation loop that can never succeed.
    """
    clean = '"""Ported from requests to httpx."""\nimport httpx\nhttpx.get("u", timeout=1)\n'
    assert not rules.has_residual_references(clean)

    dirty = "import requests\nrequests.get('u', timeout=1)\n"
    assert rules.has_residual_references(dirty)


def test_scan_matches_ground_truth_across_the_corpus(corpus):
    """Full-corpus agreement between the generator's answer key and the scanner.

    Guards both directions at once: a rule that starts over-matching and a
    generator template that starts emitting a shape it does not declare.
    """
    from collections import Counter

    truth: Counter = Counter()
    found: Counter = Counter()
    for source in corpus:
        for shape, n in source.shapes.items():
            truth[shape.value] += n
        for finding in rules.scan(source.text):
            found[finding.shape.value] += 1
    assert dict(truth) == dict(found)


@pytest.mark.parametrize("snippet", ["session.get('u')", "self.requests.get('u')"])
def test_non_module_level_calls_are_ignored(snippet):
    source = f"import requests\ndef f(session, self):\n    return {snippet}\n"
    assert [f for f in rules.scan(source) if f.rule != "exception-hierarchy"] == []
