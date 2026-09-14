"""End-to-end tests.

These are the ones that would catch a regression nobody predicted: the whole
funnel runs, produces valid Python, and costs what it claims to cost.
"""

from __future__ import annotations

import ast
import dataclasses

from migrator import model as model_mod
from migrator.config import DEFAULT
from migrator.corpus import build_corpus
from migrator.pipeline import MigrationAgent


def test_full_run_migrates_everything_and_verifies(corpus):
    report = MigrationAgent().run(corpus)
    assert report.files_migrated > 0
    assert report.unverified == []
    assert report.declined == 0


def test_every_output_file_is_valid_python(corpus):
    report = MigrationAgent().run(corpus)
    for path, text in report.output.items():
        ast.parse(text)  # raises with the path in the traceback on failure


def test_untouched_files_are_returned_byte_identical(corpus):
    """A migration tool that reformats files it had no business touching
    produces a diff nobody can review."""
    report = MigrationAgent().run(corpus)
    for source in corpus:
        if "requests" not in source.text:
            assert report.output[source.path] == source.text


def test_batch_and_interactive_get_different_timeouts(corpus):
    """The judgement call the whole ambiguous-timeout shape exists to force.

    If these ever collapse to a single value, either the fingerprint stopped
    carrying function context or the hunk window stopped including the
    enclosing def.
    """
    report = MigrationAgent().run(corpus)
    values = {
        described.split("value=")[1].rstrip(")")
        for items in report.transforms_by_cluster.values()
        for described in items
        if described.startswith("set_timeout")
    }
    assert values == {"120.0", "10.0"}

    joined = "\n".join(report.output.values())
    assert "timeout=120.0" in joined
    assert "timeout=10.0" in joined


def test_no_migrated_file_still_imports_requests(corpus):
    report = MigrationAgent().run(corpus)
    for path, text in report.output.items():
        if "httpx" in text:
            assert "import requests" not in text


def test_optimised_run_is_far_cheaper_than_naive(corpus):
    agent = MigrationAgent()
    optimised = agent.run(corpus)
    naive = MigrationAgent().run_naive(corpus)

    assert optimised.total_tokens < naive.total_tokens * 0.15
    # Output tokens are the asymmetry people miss: rules, not rewritten files.
    assert optimised.ledger.output_tokens < naive.ledger.output_tokens * 0.05
    assert optimised.requests < naive.requests / 10


def test_static_prefix_is_cached_across_requests(corpus):
    """Prefix caching only pays if the prefix is byte-identical every time.

    An interpolated count or timestamp in the system prompt would break this
    silently and roughly double input spend.

    Packing is disabled here to force multiple requests. With packing on, this
    corpus fits in a single request -- which is a nice result but leaves
    nothing for the cache to hit, so it cannot test the property.
    """
    settings = dataclasses.replace(DEFAULT, packing_enabled=False)
    report = MigrationAgent(settings=settings).run(corpus)

    assert report.requests > 1
    assert report.ledger.cached_tokens > 0
    first = report.ledger.usages[0]
    assert first.prefix_cached_tokens == 0
    assert all(u.prefix_cached_tokens > 0 for u in report.ledger.usages[1:])


def test_system_prompt_is_stable_across_runs():
    """The property prefix caching actually depends on, asserted directly."""
    from migrator import prompts

    first, _ = prompts.build_request("CLUSTER a\n", [])
    second, _ = prompts.build_request("CLUSTER b\n", ["some exemplar"])
    assert first == second == prompts.SYSTEM_PROMPT


def test_clustering_collapses_questions(corpus):
    report = MigrationAgent().run(corpus)
    assert report.clusters < report.hunks / 4
    assert report.requests < report.clusters


def test_disabling_the_codemod_costs_more_but_correctness_costs_more_still(corpus):
    """Documents a result that inverted once clustering got good.

    Early on, the deterministic pass was by far the biggest token lever. Once
    hunk merging was fixed and clusters became clean, clustering absorbed most
    of the repetition the codemod had been absorbing, and the codemod's *token*
    advantage fell to roughly 2x.

    Its real value moved somewhere more interesting. The action vocabulary is
    closed and deliberately covers only the ambiguous residue, so with the
    codemod off there is simply no action that can migrate a plain
    `requests.get`. The model declines rather than improvising, and most files
    are left unmigrated. The pass is load-bearing for correctness -- the
    opposite of how a cost 'pre-filter' is usually described.
    """
    settings = dataclasses.replace(DEFAULT, deterministic_pass=False)
    without = MigrationAgent(settings=settings).run(corpus)
    with_it = MigrationAgent().run(corpus)

    assert without.total_tokens > with_it.total_tokens * 1.8
    assert with_it.unverified == []
    assert len(without.unverified) > 50


def test_disabling_the_codemod_also_breaks_correctness(corpus):
    """The mechanism behind the failure, asserted directly.

    Files are left unmigrated because no action in the closed vocabulary can
    handle a mechanical site -- not because the model errored.
    """
    settings = dataclasses.replace(DEFAULT, deterministic_pass=False)
    report = MigrationAgent(settings=settings).run(corpus)
    assert report.unverified, "expected unmigrated files without the codemod"
    assert all("still references" in entry for entry in report.unverified)


def test_clustering_is_the_largest_token_lever(corpus):
    """Guards the headline ablation claim against silent drift.

    Losing clustering means one question per hunk, and the static prefix is
    then paid once per request rather than amortised across a packed batch.
    """
    settings = dataclasses.replace(DEFAULT, clustering=False)
    without = MigrationAgent(settings=settings).run(corpus)
    with_it = MigrationAgent().run(corpus)
    assert without.total_tokens > with_it.total_tokens * 8
    assert without.clusters == with_it.hunks


def test_escalation_recovers_from_a_bad_cheap_model(corpus):
    """The escalation path, exercised for real rather than asserted.

    The cheap model returns nothing usable, so verification fails; the strong
    model is then asked about the implicated clusters and the run recovers.
    """
    small = build_corpus(file_count=20, seed=11)
    silent = model_mod.ScriptedModel(responses=[""] * 20)
    agent = MigrationAgent(model=silent, strong_model=model_mod.OfflineModel())

    report = agent.run(small)
    assert report.escalations > 0
    assert report.unverified == []
    assert silent.calls > 0


def test_escalation_declines_cleanly_with_no_strong_model():
    """Retrying the same model on the same input is not a recovery strategy."""
    small = build_corpus(file_count=20, seed=11)
    silent = model_mod.ScriptedModel(responses=[""] * 20)
    report = MigrationAgent(model=silent).run(small)

    assert report.unverified
    assert report.escalations == 0
    assert any("no strong model" in w for w in report.warnings)


def test_run_is_deterministic(corpus):
    """Same corpus, same answer -- otherwise the benchmark means nothing."""
    first = MigrationAgent().run(corpus)
    second = MigrationAgent().run(corpus)
    assert first.output == second.output
    assert first.total_tokens == second.total_tokens
    assert first.transforms_by_cluster == second.transforms_by_cluster


def test_empty_input_does_not_crash():
    report = MigrationAgent().run([])
    assert report.files_total == 0
    assert report.requests == 0
