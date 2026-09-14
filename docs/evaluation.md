# Evaluation

## What is actually being measured

Token counts come from `tiktoken` (`cl100k_base`) over the exact strings sent to
and returned from the model. They are not estimates. `tokens.is_exact` reports
whether `tiktoken` loaded; if it did not, a `len(text) / 3.4` approximation is
used and the flag goes false, so a degraded run can never be mistaken for a
precise one.

Costs use published per-million rates in `config.Pricing`
(`gpt-4o-mini` at $0.15/$0.60, `gpt-4o` at $2.50/$10.00). Cached prefix tokens
are billed at a 50% discount in `model.Ledger`.

Two cost rows are reported, always:

- **as run** — cheap model for cluster questions, strong model for escalations.
  Real, but it bundles a *routing* win into a *token* claim.
- **same model** — both sides priced at the strong model, isolating the token
  reduction: 98.7%.

Quoting only the first number would be the easiest way to make this project look
better than it is.

## The corpus

`corpus.py` generates 120 files from six templates with a fixed seed
(`20240914`), of which 30% deliberately do not import `requests` at all — a
realistic pipeline must spend nothing on files it does not need to touch, and a
benchmark where every file is relevant would hide that.

The generator maintains an **answer key**: the exact number of sites of each
shape it emitted. `tests/test_rules.py` asserts the scanner's counts match the
key exactly on all six shapes. This is what turns the benchmark from a
plausible-looking number into a checked one.

That assertion caught a real error. The generator originally counted *templates
emitted* rather than *sites*, and several templates emit multiple sites — the
retry template contains a mechanical `requests.post(timeout=20)` plus two
`except` clauses. The codemod was being under-credited in its own benchmark.
Wrong numbers that flatter you are easy to catch; wrong numbers that understate
you are not, because nobody goes looking.

## The model

`model.OfflineModel` is not a stub returning canned strings. It parses the
rendered prompt, extracts cluster IDs and representative source, and derives
answers from the change shape and enclosing-function kind it finds there.

This is the difference between a test suite that verifies plumbing and one that
verifies behaviour. If someone changes the prompt format, drops line numbers
from `Hunk.render()`, or breaks cluster-ID attribution during packing, the
offline model stops being able to answer and CI fails — with no API key and no
network.

`ScriptedModel` exists separately for tests that need a specific adversarial
response: unknown actions, malformed rules, `ESCALATE`.

## What this harness cannot tell you

Stated plainly, because the numbers above are precise enough to be mistaken for
complete:

**Answer quality is unmeasured.** The offline model is deterministic and
agreeable. It cannot tell you whether a real GPT-4o would pick 120 seconds or
120 minutes for a nightly export. Every correctness claim here is about
*mechanical* correctness — does it parse, are there residual references, did the
rule land on the right sites — not about judgement.

This is why the exemplar ablation shows a **saving**. Exemplars cost ~10% of
spend and buy answer quality; the harness can see the cost side of that trade
and is structurally blind to the benefit side. The row is in the table with a
warning attached rather than omitted, because omitting an inconvenient
measurement is worse than publishing one that needs a caveat.

**The corpus is synthetic.** It is realistic in structure — mixed shapes, files
that need nothing, batch and interactive call sites, exception handling in
retry loops — but real codebases have decorators, wrappers, dynamic dispatch,
and vendored copies of the library. Expect the mechanical fraction to be lower
and the cluster count higher.

**`max_items_per_request = 6` is calibrated against the offline model.** It is
the budget most likely to be wrong against a real one.

## Regression protection

The benchmarks run in CI, not just the unit tests. A change that quietly
regresses clustering or token accounting fails the build rather than making the
README wrong at leisure.

Specific claims are pinned by specific tests:

| claim | test |
|---|---|
| scanner matches the answer key | `test_rules.py::test_scan_matches_ground_truth_across_the_corpus` |
| windows never merge across functions | `test_hunks.py::test_windows_do_not_merge_across_function_boundaries` |
| sites inside one function still merge | `test_hunks.py::test_sites_within_one_function_still_merge` |
| batch and interactive get different answers | `test_pipeline.py::test_batch_and_interactive_get_different_timeouts` |
| clustering is the largest lever | `test_pipeline.py::test_clustering_is_the_largest_token_lever` |
| codemod is load-bearing for correctness | `test_pipeline.py::test_disabling_the_codemod_also_breaks_correctness` |
| static prefix is byte-stable | `test_pipeline.py::test_system_prompt_is_stable_across_runs` |
| prefix caching actually engages | `test_pipeline.py::test_static_prefix_is_cached_across_requests` |
| unknown actions cannot be applied | `test_transforms.py` |
| invalid syntax abandons the batch | `test_transforms.py` |
| edits apply last-to-first | `test_rules.py::test_two_edits_on_one_call_do_not_corrupt_offsets` |

70 tests, ~4 seconds, no network.
