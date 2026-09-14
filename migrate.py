#!/usr/bin/env python
"""Command-line entry point.

    python migrate.py                     # run against the synthetic corpus
    python migrate.py --explain           # show the funnel stage by stage
    python migrate.py --show src/...py    # print one migrated file
    python migrate.py --path ./myrepo     # run against real files on disk
    python migrate.py --path ./myrepo --write   # actually modify them

``--write`` is the only flag that touches your files, and it refuses to run
unless every file passes verification. A migration tool that writes a partial
result is worse than one that writes nothing, because the half-migrated state
is the expensive one to unpick.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from migrator import cluster as cluster_mod  # noqa: E402
from migrator.corpus import SourceFile, build_corpus, corpus_stats  # noqa: E402
from migrator.pipeline import MigrationAgent  # noqa: E402


def load_from_disk(root: Path) -> list:
    files = []
    for path in sorted(root.rglob("*.py")):
        try:
            files.append(
                SourceFile(
                    path=str(path.relative_to(root)),
                    text=path.read_text(encoding="utf-8"),
                    shapes={},
                )
            )
        except (OSError, UnicodeDecodeError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="requests -> httpx migration agent")
    parser.add_argument("--path", type=Path, help="migrate real files under this directory")
    parser.add_argument("--write", action="store_true", help="write changes to disk")
    parser.add_argument("--explain", action="store_true", help="show the funnel")
    parser.add_argument("--show", metavar="FILE", help="print one migrated file")
    args = parser.parse_args()

    if args.path:
        files = load_from_disk(args.path)
        if not files:
            print(f"no .py files under {args.path}", file=sys.stderr)
            return 1
    else:
        files = build_corpus()

    agent = MigrationAgent()
    report = agent.run(files)

    if args.show:
        text = report.output.get(args.show)
        if text is None:
            print(f"no such file in the run: {args.show}", file=sys.stderr)
            print("try one of:", file=sys.stderr)
            for path in list(report.output)[:10]:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(text)
        return 0

    if args.explain:
        explain(files, report)
    else:
        summary = report.summary()
        width = max(len(k) for k in summary)
        for key, value in summary.items():
            shown = f"{value:,}" if isinstance(value, int) else value
            print(f"{key:<{width}}  {shown}")

    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for warning in report.warnings[:10]:
            print(f"  {warning}")

    if report.unverified:
        print(f"\n{len(report.unverified)} file(s) failed verification:")
        for entry in report.unverified[:10]:
            print(f"  {entry}")

    if args.write:
        if not args.path:
            print("\n--write requires --path", file=sys.stderr)
            return 1
        if report.unverified:
            print("\nrefusing to write: verification failed", file=sys.stderr)
            return 1
        written = 0
        for relative, text in report.output.items():
            target = args.path / relative
            if target.read_text(encoding="utf-8") != text:
                target.write_text(text, encoding="utf-8")
                written += 1
        print(f"\nwrote {written} file(s)")

    return 1 if report.unverified else 0


def explain(files, report) -> None:
    stats = corpus_stats(files)
    print("THE FUNNEL\n")
    steps = [
        ("files scanned", report.files_total, ""),
        ("files importing requests", report.files_touched, ""),
        ("migration sites found", report.sites_total, ""),
        (
            "resolved by codemod",
            report.sites_by_codemod,
            "no model, no tokens",
        ),
        ("left for the model", report.sites_by_model, ""),
        ("windowed hunks", report.hunks, "after merging overlaps"),
        ("clusters", report.clusters, "distinct change shapes"),
        ("requests sent", report.requests, "after packing"),
    ]
    width = max(len(label) for label, _, _ in steps)
    for label, value, note in steps:
        suffix = f"   ({note})" if note else ""
        print(f"  {label:<{width}}  {value:>6,}{suffix}")

    print("\nRULES THE MODEL RETURNED\n")
    for fingerprint, described in sorted(report.transforms_by_cluster.items()):
        print(f"  {fingerprint}")
        for item in described:
            print(f"      {item}")

    print("\nTOKENS\n")
    for key, value in report.ledger.summary().items():
        shown = f"{value:,}" if isinstance(value, int) else value
        print(f"  {key:<22}  {shown}")

    if stats["sites"]:
        share = report.sites_by_codemod / stats["sites"] * 100
        print(
            f"\n  {share:.0f}% of sites never reached the model. "
            "That is the whole game."
        )


if __name__ == "__main__":
    raise SystemExit(main())
