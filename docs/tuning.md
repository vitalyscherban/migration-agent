# Tuning

All budgets live in `src/migrator/config.py` as frozen dataclasses. Defaults
below are the values in `DEFAULT`. Each is tagged **measured** (chosen by
running the benchmark) or **judgement** (chosen by reasoning, not yet
falsified).

## Hunk budget

| setting | default | basis |
|---|---:|---|
| `context_before` | 6 | measured |
| `context_after` | 6 | measured |
| `max_hunk_tokens` | 400 | judgement |
| `include_enclosing_def` | `True` | measured |

Six lines either side is the point where the offline model's answers stopped
changing on this corpus. Three lines lost the enclosing `for` in the retry
shape; twelve added ~35% tokens and changed nothing.

`include_enclosing_def` is not a nicety. The batch-vs-interactive judgement is
made almost entirely from the function name and signature — without it the
model cannot tell `nightly_export` from `handle_page_load`, and every timeout
regresses to one value.

## Cluster budget

| setting | default | basis |
|---|---:|---|
| `min_cluster_size` | 2 | judgement |
| `representatives_per_cluster` | 2 | measured |
| `max_cluster_size` | 40 | judgement |

One representative was enough for the timeout clusters but not the exception
clusters, where a single sample under-specified which exception subtype the
cluster covered. Three cost more and changed no answers.

`max_cluster_size` caps blast radius, not tokens: a rule landing on more than 40
sites from one answer is a review problem regardless of what it costs.

## Pack budget

| setting | default | basis |
|---|---:|---|
| `max_payload_tokens` | 2400 | judgement |
| `max_items_per_request` | 6 | judgement |

The item cap matters more than the token cap. Beyond roughly six clusters in one
request the offline model starts mis-attributing answers to cluster IDs, and a
mis-attributed rule is worse than a second request. This is the one budget that
should be re-derived against a real model before production use — 6 is a guess
calibrated against a harness, not a finding.

## Exemplar budget

| setting | default | basis |
|---|---:|---|
| `top_k` | 2 | judgement |
| `min_similarity` | 0.25 | judgement |
| `max_exemplar_tokens` | 200 | judgement |

Exemplars **cost** about 10% of total spend and the offline harness scores their
benefit at zero, because it cannot measure answer quality. They are on by
default anyway. See `docs/evaluation.md` for why that is a deliberate choice
rather than an oversight.

## Levers

Six boolean flags, each independently ablatable, each consumed by real code:

| lever | effect when off |
|---|---|
| `deterministic_pass` | all 260 sites go to the model; 76 files fail to migrate |
| `windowing` | whole files sent instead of hunks |
| `clustering` | one question per hunk (72 instead of 4) |
| `packing_enabled` | one request per cluster |
| `exemplars_enabled` | no retrieved examples in the prompt |
| `escalation_enabled` | verification failures are reported, not retried |

> A config knob that nothing consumes is worse than no knob — it reads like a
> capability and behaves like a comment.

Two flags were removed during development for exactly this reason.
`patch_output` was decorative; its story is told better by the naive-vs-optimised
output-token comparison. `escalation_enabled` was decorative too, and was
implemented for real rather than deleted.

## Retuning for a different migration

The architecture generalises to any migration where a large fraction of sites
are mechanical and the remainder cluster into a small number of judgement calls.
To port it:

1. Rewrite the shape taxonomy in `rules.py` — this is the bulk of the work.
2. Define the closed action vocabulary in `transforms.py`. Keep it small. If the
   vocabulary needs more than a handful of actions, the migration probably is
   not clusterable and this design will not pay.
3. Adjust the fingerprint in `cluster.py` to whatever distinguishes intent in
   your domain.
4. Run `benchmarks/ablation.py` and check `cluster.stats()["compression"]`
   before trusting any of it.

The single number that predicts whether this approach is worth the effort is
that compression ratio. On this corpus it is 18:1 (72 hunks → 4 clusters). Below
roughly 3:1 the machinery costs more than it saves and a straightforward
per-file agent is the better engineering choice.
