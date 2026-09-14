"""Tests for hunk windowing.

The window is the quality/cost dial. These tests pin the two things that
silently degrade answers: losing the context the model needs to judge, and
targeting more lines than the finding actually occupies.
"""

from __future__ import annotations

from conftest import AMBIGUOUS_SOURCE

from migrator import codemod, hunks, rules
from migrator.config import HunkBudget


def _hunks(source, budget=None):
    budget = budget or HunkBudget()
    result = codemod.run("x.py", source)
    return hunks.extract("x.py", result.text, result.unresolved, budget)


def test_windowing_reduces_bytes_sent(corpus):
    """The headline claim of this stage, measured rather than asserted.

    Two ratios matter and they tell different stories.

    Against the files that actually reach the model, windowing saves ~28%.
    That is real but unspectacular, and the reason is worth knowing: these are
    small modules with several sites each, so a +/-6 line window around every
    site covers much of the file anyway. On a codebase of 400-line modules
    with one site apiece the same code saves far more. This is why the
    ablation prices windowing well below the deterministic pass -- the lever's
    value is a property of the corpus, not a constant.

    Against every file a naive run would send, the funnel is already at ~50%
    by this stage, and the deterministic pass has not finished paying out yet.
    """
    from migrator import tokens

    model_bound_tokens = 0
    hunk_tokens = 0
    all_touched_tokens = 0

    for source in corpus:
        if "requests" in source.text:
            all_touched_tokens += tokens.count(source.text)
        result = codemod.run(source.path, source.text)
        if not result.unresolved:
            continue
        model_bound_tokens += tokens.count(result.text)
        for hunk in hunks.extract(
            source.path, result.text, result.unresolved, HunkBudget()
        ):
            hunk_tokens += hunk.token_cost

    assert hunk_tokens < model_bound_tokens * 0.80
    assert hunk_tokens < all_touched_tokens * 0.60


def test_windows_do_not_merge_across_function_boundaries():
    """Regression: merging on line proximity alone is a correctness bug.

    These two calls are three lines apart, so a proximity-only merge joins
    them into one hunk. They then land in one cluster and receive one answer.
    But one is a nightly export and the other serves a page view, and their
    correct timeouts differ by more than 10x -- so whichever answer came back
    would be wrong for one of them, silently, in a file that still parses and
    still passes review.
    """
    source = (
        "import requests\n"
        "def export_ledger(w):\n"
        "    return requests.get('u', params={'w': w})\n"
        "def load_for_request(i):\n"
        "    return requests.get(f'u/{i}')\n"
    )
    extracted = _hunks(source)
    assert len(extracted) == 2

    markers = [
        {s for s in hunk.signature if s.startswith("fn:")} for hunk in extracted
    ]
    assert all(len(m) == 1 for m in markers), "a hunk must not mix path kinds"
    assert {next(iter(m)) for m in markers} == {"fn:batch", "fn:interactive"}


def test_sites_within_one_function_still_merge():
    """The guard must not over-fire and defeat merging entirely."""
    source = (
        "import requests\n"
        "def f(a, b):\n"
        "    x = requests.get(a)\n"
        "    y = requests.get(b)\n"
        "    return x, y\n"
    )
    extracted = _hunks(source)
    assert len(extracted) == 1
    assert extracted[0].site_lines == (3, 4)


def test_overlapping_windows_are_merged():
    """Two findings three lines apart must not ship the same source twice.

    Beyond the duplicated payload, two overlapping questions can come back
    with contradictory answers for the same region.
    """
    source = (
        "import requests\n"
        "def f(a, b):\n"
        "    x = requests.get(a)\n"
        "    y = requests.get(b)\n"
        "    return x, y\n"
    )
    extracted = _hunks(source)
    assert len(extracted) == 1
    assert extracted[0].site_lines == (3, 4)


def test_distant_findings_stay_separate():
    body = "\n".join(f"    v{i} = {i}" for i in range(40))
    source = (
        "import requests\n"
        "def f(a, b):\n"
        "    x = requests.get(a)\n"
        f"{body}\n"
        "    y = requests.get(b)\n"
    )
    assert len(_hunks(source)) == 2


def test_enclosing_def_is_included_when_outside_the_window():
    """Without it the model cannot tell a batch path from a request path.

    That is precisely the judgement the ambiguous-timeout shape exists to
    force, so dropping this line turns the decision into a coin flip.
    """
    padding = "\n".join(f"    step_{i} = {i}" for i in range(20))
    source = (
        "import requests\n"
        "def export_ledger(window):\n"
        f"{padding}\n"
        "    return requests.get('u')\n"
    )
    extracted = _hunks(source)
    assert len(extracted) == 1
    assert extracted[0].enclosing_def == "def export_ledger(...)"
    assert "export_ledger" in extracted[0].render()


def test_enclosing_def_not_duplicated_when_already_visible():
    source = "import requests\ndef f(a):\n    return requests.get(a)\n"
    extracted = _hunks(source)
    assert extracted[0].enclosing_def is None
    assert extracted[0].render().count("def f") == 1


def test_site_lines_are_exact_not_the_window():
    """Transforms target these lines, so padding must not leak into them.

    If the window were used as the target range, a rule aimed at one call
    could also hit an unrelated call that merely sat within six lines of it.
    """
    source = (
        "import requests\n"
        "def f(a):\n"
        "    before = 1\n"
        "    x = requests.get(a)\n"
        "    after = 2\n"
        "    return x\n"
    )
    hunk = _hunks(source)[0]
    assert hunk.site_lines == (4,)
    assert hunk.start_line < 4 < hunk.end_line


def test_rendered_hunk_is_line_numbered():
    """Line numbers are what make patch-shaped responses addressable."""
    rendered = _hunks(AMBIGUOUS_SOURCE)[0].render()
    assert " | " in rendered
    assert rendered.startswith("--- x.py:")


def test_oversized_hunk_is_flagged_not_truncated():
    """Truncating would send a question the model cannot answer correctly.

    Flagging routes it to a human instead; truncating produces a confident
    guess that then gets applied.
    """
    filler = "\n".join(f"    v{i} = {i}" for i in range(200))
    source = f"import requests\ndef f(a):\n{filler}\n    return requests.get(a)\n"
    tight = HunkBudget(context_before=300, context_after=300, max_hunk_tokens=50)
    extracted = _hunks(source, tight)
    assert extracted[0].oversized
    # The text is intact; nothing was silently dropped.
    assert "v199" in extracted[0].text


def test_no_findings_produces_no_hunks():
    assert hunks.extract("x.py", "x = 1\n", [], HunkBudget()) == []
