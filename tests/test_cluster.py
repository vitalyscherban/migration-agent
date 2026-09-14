"""Tests for clustering.

Clustering is what turns 62 questions into 10. The risk is not that it fails
to group -- it is that it groups too eagerly, merging two shapes whose correct
answers differ, and then applies one of those answers to all of them.
"""

from __future__ import annotations

from migrator import cluster, codemod, hunks
from migrator.config import ClusterBudget, HunkBudget
from migrator.corpus import Shape, build_corpus


def _all_hunks(files):
    collected = []
    for source in files:
        result = codemod.run(source.path, source.text)
        collected.extend(
            hunks.extract(source.path, result.text, result.unresolved, HunkBudget())
        )
    return collected


def test_identical_shapes_collapse(corpus):
    collected = _all_hunks(corpus)
    clusters = cluster.build(collected, ClusterBudget())
    stats = cluster.stats(clusters)
    assert stats["sites"] == len(collected)
    assert stats["clusters"] < len(collected) / 5
    assert stats["compression"] > 5


def test_batch_and_interactive_timeouts_do_not_merge(corpus):
    """The sharpest correctness test in this module.

    Both shapes are `ambiguous_timeout` to the scanner -- same rule, same call
    shape, same missing kwarg. Only the enclosing function distinguishes them,
    and the right answers differ by more than an order of magnitude (120s vs
    10s). If the fingerprint drops that context, the two merge and every
    interactive request path in the codebase inherits a two minute timeout.
    """
    collected = _all_hunks(corpus)
    timeout_only = [
        h for h in collected if set(h.shapes) == {Shape.AMBIGUOUS_TIMEOUT}
    ]
    assert timeout_only, "corpus should contain pure timeout hunks"

    clusters = cluster.build(timeout_only, ClusterBudget())
    kinds = set()
    for group in clusters:
        marker = {s for s in group.members[0].signature if s.startswith("fn:")}
        assert len(marker) == 1, "each cluster must have one consistent path kind"
        kinds |= marker

    assert kinds == {"fn:batch", "fn:interactive"}
    assert len(clusters) >= 2


def test_fingerprint_ignores_identifiers():
    """Otherwise every site is unique and clustering does nothing.

    Two calls that differ only in variable and parameter names are the same
    migration problem.
    """
    left = "import requests\ndef export_a(x):\n    return requests.get(x)\n"
    right = "import requests\ndef export_b(banana):\n    return requests.get(banana)\n"

    def one(source):
        result = codemod.run("f.py", source)
        return hunks.extract("f.py", result.text, result.unresolved, HunkBudget())[0]

    assert cluster.fingerprint(one(left)) == cluster.fingerprint(one(right))


def test_oversized_clusters_are_split_for_blast_radius():
    files = build_corpus(file_count=200, untouched_ratio=0.0, seed=3)
    collected = _all_hunks(files)
    budget = ClusterBudget(max_cluster_size=5)
    clusters = cluster.build(collected, budget)
    assert clusters
    assert all(c.size <= 5 for c in clusters)


def test_representatives_are_the_smallest_members(corpus):
    """Small clean examples generalise; large ones carry incidental noise."""
    collected = _all_hunks(corpus)
    clusters = cluster.build(collected, ClusterBudget())
    biggest = max(clusters, key=lambda c: c.size)
    budget = ClusterBudget(representatives_per_cluster=2)
    reps = biggest.representatives(budget)
    assert len(reps) == 2
    costs = sorted(h.token_cost for h in biggest.members)
    assert [r.token_cost for r in reps] == costs[:2]


def test_every_member_is_covered_by_followers(corpus):
    """No site may be silently dropped between clustering and application."""
    collected = _all_hunks(corpus)
    clusters = cluster.build(collected, ClusterBudget())
    covered = sum(len(c.followers) for c in clusters)
    assert covered == len(collected)


def test_singletons_are_marked():
    files = build_corpus(file_count=4, untouched_ratio=0.0, seed=99)
    clusters = cluster.build(_all_hunks(files), ClusterBudget(min_cluster_size=99))
    assert all(c.singleton for c in clusters)


def test_empty_input():
    assert cluster.build([], ClusterBudget()) == []
    assert cluster.stats([])["clusters"] == 0
