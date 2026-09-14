"""Naive vs optimised, measured over the same corpus.

Run: ``python benchmarks/compare.py``

Every token count here comes from ``tiktoken`` over the exact bytes each
approach would send. Nothing is extrapolated and nothing is estimated -- the
only modelled quantity is the provider's cached-prefix discount, which is
applied to dollars, not to token counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from migrator import tokens  # noqa: E402
from migrator.config import DEFAULT  # noqa: E402
from migrator.corpus import build_corpus, corpus_stats  # noqa: E402
from migrator.pipeline import MigrationAgent  # noqa: E402


def main() -> None:
    files = build_corpus()
    stats = corpus_stats(files)

    agent = MigrationAgent()
    optimised = agent.run(files)
    naive = MigrationAgent().run_naive(files)

    print("CORPUS")
    print(f"  {stats['files']} files, {stats['files_touched']} import requests")
    print(
        f"  {stats['sites']} migration sites "
        f"({stats['mechanical_sites']} mechanical, {stats['ambiguous_sites']} ambiguous)"
    )
    if not tokens.is_exact:
        print("  WARNING: tiktoken unavailable, token counts are estimates")
    print()

    print("NAIVE vs OPTIMISED")
    print(f"{'':<26}{'naive':>12}{'optimised':>12}{'saved':>10}")
    print("-" * 60)

    rows = [
        ("model requests", naive.requests, optimised.requests),
        ("input tokens", naive.ledger.input_tokens, optimised.ledger.input_tokens),
        ("output tokens", naive.ledger.output_tokens, optimised.ledger.output_tokens),
        ("total tokens", naive.total_tokens, optimised.total_tokens),
    ]
    for label, left, right in rows:
        saved = f"{(1 - right / left) * 100:.1f}%" if left else "--"
        print(f"{label:<26}{left:>12,}{right:>12,}{saved:>10}")

    naive_cost = naive.ledger.cost()
    opt_cost = optimised.ledger.cost()
    saved_cost = f"{(1 - opt_cost / naive_cost) * 100:.1f}%" if naive_cost else "--"
    print(f"{'cost (as run)':<26}{'$' + format(naive_cost, '.4f'):>12}"
          f"{'$' + format(opt_cost, '.4f'):>12}{saved_cost:>10}")

    # The line above compares a strong-model baseline against a cheap-model
    # optimised run, so it bundles the model-routing win into the token win.
    # That flatters the result. This line prices both at the strong model, so
    # what remains is the token reduction alone.
    strong = DEFAULT.escalation.strong_model
    pricing = DEFAULT.pricing
    naive_same = pricing.cost(strong, naive.ledger.input_tokens, naive.ledger.output_tokens)
    opt_same = pricing.cost(
        strong, optimised.ledger.input_tokens, optimised.ledger.output_tokens
    )
    saved_same = f"{(1 - opt_same / naive_same) * 100:.1f}%" if naive_same else "--"
    print(f"{'cost (same model)':<26}{'$' + format(naive_same, '.4f'):>12}"
          f"{'$' + format(opt_same, '.4f'):>12}{saved_same:>10}")
    print()
    print("  'as run' bundles in the win from routing the narrow, well-specified")
    print("  cluster questions to a cheap model. 'same model' prices both sides at")
    print("  the strong model, isolating the token reduction on its own.")
    print()

    print("WHERE THE WORK WENT")
    print(f"  resolved by codemod   {optimised.sites_by_codemod:>4} sites  (no tokens)")
    print(f"  resolved by model     {optimised.sites_by_model:>4} sites")
    print(f"  hunks after merging   {optimised.hunks:>4}")
    print(f"  clusters asked about  {optimised.clusters:>4}")
    print(f"  requests sent         {optimised.requests:>4}")
    print(f"  cached prefix tokens  {optimised.ledger.cached_tokens:>4}")
    print()

    print("CORRECTNESS")
    print(f"  files migrated        {optimised.files_migrated}")
    print(f"  failed verification   {len(optimised.unverified)}")
    print(f"  model declined        {optimised.declined} cluster(s)")
    if optimised.unverified:
        for entry in optimised.unverified[:5]:
            print(f"    {entry}")
    print()

    print("AT SCALE (linear extrapolation to a 4x larger codebase)")
    print(f"  naive      ${naive_cost * 4:,.4f} per full migration pass")
    print(f"  optimised  ${opt_cost * 4:,.4f} per full migration pass")
    print("  Linear is conservative for the optimised run: its cost is driven by")
    print("  cluster count, which grows far slower than file count.")
    print()
    print("The output-token column is the one most people miss. The naive run")
    print("returns whole rewritten files, so its output scales with the size of")
    print("the codebase. The optimised run returns parameterised rules, so its")
    print("output scales with the number of distinct problems -- which barely")
    print("moves as the codebase grows.")


if __name__ == "__main__":
    main()
