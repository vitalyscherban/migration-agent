"""Tests for the transform vocabulary.

This layer applies model output to source code, so its tests are mostly about
what it *refuses* to do. The closed vocabulary and the all-or-nothing apply
are the two properties that keep a bad generation from becoming a bad commit.
"""

from __future__ import annotations

import ast

import pytest

from migrator import transforms
from migrator.transforms import Transform, TransformError


def test_unknown_action_is_rejected_at_construction():
    """The vocabulary is closed; a model cannot widen it at runtime."""
    with pytest.raises(TransformError):
        Transform(action="delete_file", params={"path": "/etc/passwd"})


def test_set_timeout_requires_a_numeric_value():
    source = "import requests\nrequests.get('u')\n"
    with pytest.raises(TransformError):
        transforms._apply_set_timeout(
            source, Transform(action="set_timeout", params={"value": "soon"}), None
        )


def test_set_timeout_on_a_multiline_call():
    """Insertion goes after the last argument, not before the closing paren.

    The trailing comma and the dedented `)` are what break naive insertion.
    """
    source = (
        "import requests\n"
        "r = requests.get(\n"
        "    'u',\n"
        "    headers=H,\n"
        ")\n"
    )
    report = transforms.apply(source, [Transform("set_timeout", {"value": "30.0"})])
    assert report.ok
    assert "timeout=30.0" in report.text
    assert "httpx.get" in report.text
    ast.parse(report.text)


def test_set_timeout_is_idempotent():
    """Verification failures cause retries, so transforms must be replayable.

    A non-idempotent version emits a second `timeout=` on the second run,
    which is a TypeError at call time -- reached only in production.
    """
    source = "import requests\nr = requests.get('u')\n"
    once = transforms.apply(source, [Transform("set_timeout", {"value": "5.0"})])
    twice = transforms.apply(once.text, [Transform("set_timeout", {"value": "5.0"})])
    assert once.text == twice.text
    assert twice.applied == 0
    assert once.text.count("timeout=") == 1


def test_line_filter_limits_the_blast_radius():
    """A rule for one cluster must not touch a site belonging to another."""
    source = (
        "import requests\n"
        "def a():\n"
        "    return requests.get('one')\n"
        "def b():\n"
        "    return requests.get('two')\n"
    )
    report = transforms.apply(
        source, [Transform("set_timeout", {"value": "9.0"})], line_filter=(3,)
    )
    assert report.applied == 1
    assert "httpx.get('one', timeout=9.0)" in report.text
    assert "requests.get('two')" in report.text


def test_map_exception_rewrites_only_the_named_target():
    source = (
        "import requests\n"
        "try:\n"
        "    pass\n"
        "except requests.exceptions.ConnectionError:\n"
        "    pass\n"
        "except requests.exceptions.Timeout:\n"
        "    pass\n"
    )
    report = transforms.apply(
        source,
        [
            Transform(
                "map_exception",
                {
                    "from": "requests.exceptions.ConnectionError",
                    "to": "httpx.ConnectError",
                },
            )
        ],
    )
    assert "except httpx.ConnectError:" in report.text
    assert "except requests.exceptions.Timeout:" in report.text


def test_clause_order_is_preserved():
    """Order decides which handler wins, and the hierarchies are not parallel."""
    source = (
        "import requests\n"
        "try:\n"
        "    pass\n"
        "except requests.exceptions.ConnectionError:\n"
        "    first()\n"
        "except requests.exceptions.Timeout:\n"
        "    second()\n"
    )
    report = transforms.apply(
        source,
        [
            Transform(
                "map_exception",
                {"from": "requests.exceptions.ConnectionError", "to": "httpx.ConnectError"},
            ),
            Transform(
                "map_exception",
                {"from": "requests.exceptions.Timeout", "to": "httpx.TimeoutException"},
            ),
        ],
    )
    tree = ast.parse(report.text)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    names = [ast.unparse(h.type) for h in handlers]
    assert names == ["httpx.ConnectError", "httpx.TimeoutException"]


def test_sequential_transforms_do_not_drift():
    """Each transform re-parses, so the second sees the first's real positions.

    Without the re-parse this is the offset-drift bug one level up: positions
    computed against the original text, applied to the edited text.
    """
    source = (
        "import requests\n"
        "def f():\n"
        "    try:\n"
        "        return requests.get('u')\n"
        "    except requests.exceptions.Timeout:\n"
        "        return None\n"
    )
    report = transforms.apply(
        source,
        [
            Transform("set_timeout", {"value": "15.0"}),
            Transform(
                "map_exception",
                {"from": "requests.exceptions.Timeout", "to": "httpx.TimeoutException"},
            ),
        ],
    )
    assert report.ok
    assert "httpx.get('u', timeout=15.0)" in report.text
    assert "except httpx.TimeoutException:" in report.text
    ast.parse(report.text)


def test_a_batch_producing_bad_syntax_is_abandoned_whole():
    """Partial application is the one outcome worse than outright failure.

    It leaves a file that is neither migrated nor original, and no longer
    corresponds to any answer the model actually gave.
    """
    source = "import requests\nr = requests.get('u')\n"
    bad = Transform("rename_symbol", {"from": "requests.get", "to": "httpx.get("})
    report = transforms.apply(source, [bad])
    assert not report.ok
    assert report.text == source


def test_rename_symbol_ignores_docstring_mentions():
    source = '"""Uses requests.Session heavily."""\nimport requests\ns = requests.Session()\n'
    report = transforms.apply(
        source, [Transform("rename_symbol", {"from": "requests.Session", "to": "httpx.Client"})]
    )
    assert "httpx.Client()" in report.text
    assert "Uses requests.Session heavily" in report.text
