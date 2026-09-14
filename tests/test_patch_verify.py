"""Tests for response parsing and verification.

The parser is the trust boundary between a free-text generation and an
automated rewrite of hundreds of files. It is supposed to be pedantic.
"""

from __future__ import annotations

from migrator import patch, verify


def test_parses_a_well_formed_answer():
    response = (
        "CLUSTER abc123\n"
        "ACTION set_timeout value=30.0\n"
        "NOTE batch path\n"
    )
    result = patch.parse(response, ["abc123"])
    answer = result.for_cluster("abc123")
    assert len(answer.transforms) == 1
    assert answer.transforms[0].params["value"] == "30.0"
    assert answer.note == "batch path"


def test_blocks_for_unrequested_clusters_are_dropped():
    """A packed request sometimes comes back with an extra cluster.

    The fingerprint is either hallucinated or copied from an exemplar. Without
    this guard those transforms get applied to whatever cluster holds that
    key -- a rule answering a question nobody asked, landing on real files.
    """
    response = (
        "CLUSTER real\nACTION set_timeout value=5.0\n"
        "CLUSTER ghost\nACTION set_timeout value=999.0\n"
    )
    result = patch.parse(response, ["real"])
    assert "ghost" not in result.answers
    assert any("ghost" in w for w in result.warnings)


def test_unknown_action_is_rejected_with_a_warning():
    response = "CLUSTER a\nACTION drop_table name=users\nACTION set_timeout value=5.0\n"
    result = patch.parse(response, ["a"])
    answer = result.for_cluster("a")
    assert [t.action for t in answer.transforms] == ["set_timeout"]
    assert any("drop_table" in w for w in result.warnings)


def test_missing_answer_is_reported():
    """A packed request that silently drops a cluster must be visible.

    This is the failure mode that makes over-packing dangerous: the model
    answers five of six questions and says nothing about the sixth.
    """
    result = patch.parse("CLUSTER a\nACTION set_timeout value=5.0\n", ["a", "b"])
    assert any("no answer returned for cluster b" in w for w in result.warnings)


def test_escalation_block_is_recognised():
    result = patch.parse("CLUSTER a\nESCALATE shape not covered\n", ["a"])
    answer = result.for_cluster("a")
    assert answer.is_escalation
    assert not answer.transforms


def test_prose_around_blocks_is_ignored_not_misparsed():
    response = (
        "Sure! Here are the transforms you asked for:\n\n"
        "CLUSTER a\nACTION set_timeout value=5.0\n\n"
        "Let me know if you'd like anything changed.\n"
    )
    result = patch.parse(response, ["a"])
    assert len(result.for_cluster("a").transforms) == 1


def test_empty_response_yields_no_transforms():
    result = patch.parse("", ["a"])
    assert not result.for_cluster("a").transforms
    assert result.warnings


# -- verification --------------------------------------------------------


def test_clean_migration_verifies():
    text = "import httpx\n\ndef f(u):\n    return httpx.get(u, timeout=10).json()\n"
    assert verify.verify("f.py", text).ok


def test_residual_requests_reference_fails():
    text = "import httpx\n\ndef f(u):\n    return requests.get(u, timeout=10)\n"
    result = verify.verify("f.py", text)
    assert not result.ok
    assert "still references" in result.failures[0]


def test_docstring_mention_of_requests_does_not_fail():
    """The false positive that would make escalation loop forever.

    A file that merely *talks* about requests is correctly migrated. Failing
    it sends it to the strong model, which returns the same file, which fails
    again.
    """
    text = '"""Migrated from requests."""\nimport httpx\nhttpx.get("u", timeout=1)\n'
    assert verify.verify("f.py", text).ok


def test_leftover_allow_redirects_fails():
    text = "import httpx\nhttpx.get('u', allow_redirects=True, timeout=5)\n"
    result = verify.verify("f.py", text)
    assert not result.ok
    assert "allow_redirects" in result.failures[0]


def test_missing_timeout_warns_but_does_not_fail():
    """A warning, deliberately.

    Post-migration the call inherits httpx's 5s default, so it is bounded and
    safe. Failing it would push correctly-migrated files into escalation and
    spend strong-model tokens on a non-problem.
    """
    text = "import httpx\nhttpx.get('u')\n"
    result = verify.verify("f.py", text)
    assert result.ok
    assert result.warnings
    assert "5s default" in result.warnings[0]


def test_unparseable_output_fails():
    result = verify.verify("f.py", "import httpx\ndef broken(:\n")
    assert not result.ok
    assert "does not parse" in result.failures[0]
