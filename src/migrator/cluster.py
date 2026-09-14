"""Groups hunks that need the same transform.

A migration is repetitive by nature. The same three-line retry block appears
in forty files because it was copy-pasted into forty files. Asking the model
about it forty times is not just wasteful, it is actively worse than asking
once: forty independent answers drift, and you end up with four slightly
different retry idioms in a codebase that previously had one.

Clustering fingerprints each hunk by its *change shape* -- the rule that
declined, the call signature, the surrounding control flow -- and sends a
couple of representatives. The returned transform is then applied to the whole
cluster and verified per-site.

The fingerprint deliberately excludes identifiers. Including the variable name
or the URL would make every site unique and collapse the cluster count back to
the number of sites, which is the failure mode this module exists to prevent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .config import ClusterBudget
from .corpus import Shape
from .hunks import Hunk


@dataclass
class Cluster:
    fingerprint: str
    shapes: Tuple[Shape, ...]
    members: List[Hunk] = field(default_factory=list)
    #: True when the cluster is too small to be worth a shared transform and
    #: its members are handled individually instead.
    singleton: bool = False

    @property
    def size(self) -> int:
        return len(self.members)

    def representatives(self, budget: ClusterBudget) -> List[Hunk]:
        """Pick the examples that go in the prompt.

        Sorted by token cost, then the *smallest* are taken. This looks
        backwards -- surely the richest example teaches most? -- but the
        opposite held when measured. Large representatives are large because
        they carry incidental surrounding code, and the model reliably
        pattern-matches on that incidental code, producing a transform that
        only fits hunks with the same incidental shape. Small, clean examples
        generalise better and cost less, which is a rare case of the cheap
        option also being the good one.
        """
        ordered = sorted(self.members, key=lambda h: h.token_cost)
        return ordered[: budget.representatives_per_cluster]

    @property
    def followers(self) -> List[Hunk]:
        """Members that ride on the representatives' answer, costing nothing."""
        return self.members[:]


def fingerprint(hunk: Hunk) -> str:
    """Stable identity for a hunk's change shape.

    Hashed rather than kept as a readable tuple purely so cluster identity is
    a short stable string in logs and reports; the readable form is retained
    on the cluster as ``shapes``.
    """
    parts = [shape.value for shape in sorted(set(hunk.shapes), key=lambda s: s.value)]
    parts.extend(sorted(set(hunk.signature)))
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=6).hexdigest()
    return digest


def build(hunks: Sequence[Hunk], budget: ClusterBudget) -> List[Cluster]:
    """Group ``hunks`` into clusters, splitting any that grow too large."""
    buckets: Dict[str, Cluster] = {}
    for hunk in hunks:
        key = fingerprint(hunk)
        cluster = buckets.get(key)
        if cluster is None:
            cluster = Cluster(
                fingerprint=key,
                shapes=tuple(dict.fromkeys(hunk.shapes)),
            )
            buckets[key] = cluster
        cluster.members.append(hunk)

    clusters: List[Cluster] = []
    for cluster in buckets.values():
        if cluster.size > budget.max_cluster_size:
            clusters.extend(_split(cluster, budget.max_cluster_size))
        else:
            clusters.append(cluster)

    for cluster in clusters:
        cluster.singleton = cluster.size < budget.min_cluster_size

    # Largest first: the biggest clusters carry the most leverage per token,
    # so if a run is budget-capped it should spend on those first.
    clusters.sort(key=lambda c: (-c.size, c.fingerprint))
    return clusters


def _split(cluster: Cluster, limit: int) -> List[Cluster]:
    """Chop an oversized cluster into capped shards.

    The cap is a blast-radius control, not an optimisation. One bad transform
    applied to 300 sites is a bad afternoon; applied to 40 it is caught by the
    first shard's verification and the remaining shards are re-derived.
    """
    shards: List[Cluster] = []
    for index in range(0, cluster.size, limit):
        shards.append(
            Cluster(
                fingerprint=f"{cluster.fingerprint}-{index // limit}",
                shapes=cluster.shapes,
                members=cluster.members[index : index + limit],
            )
        )
    return shards


def stats(clusters: Sequence[Cluster]) -> dict:
    sizes = [c.size for c in clusters]
    return {
        "clusters": len(clusters),
        "sites": sum(sizes),
        "largest": max(sizes) if sizes else 0,
        "singletons": sum(1 for c in clusters if c.singleton),
        "compression": round(sum(sizes) / len(clusters), 2) if clusters else 0.0,
    }
