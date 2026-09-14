"""The migration agent: orchestrates the funnel and accounts for every token.

The shape of the run, and where the money goes:

    files            -> deterministic codemod  (no model, no tokens)
    unresolved sites -> windowed hunks         (bytes sent shrink ~20x)
    hunks            -> clusters               (questions shrink ~10x)
    clusters         -> packed requests        (prefix paid ~3x not ~20x)
    answers          -> parameterised rules    (output is O(clusters))
    rules            -> applied + verified     (failures escalate, not merge)

Each stage is individually switchable via ``Settings`` so the ablation
benchmark can price the levers separately rather than asserting their value.

One structural choice worth flagging: transforms are collected per file and
applied in a single pass at the end, rather than per cluster as answers
arrive. A file can hold sites belonging to three different clusters, and
applying three separate edit passes against positions computed from three
different versions of the text is the offset-drift bug from ``rules.py`` all
over again, one level higher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import (
    cluster as cluster_mod,
    codemod,
    exemplars as exemplar_mod,
    hunks as hunks_mod,
    model as model_mod,
    packer,
    patch,
    prompts,
    rules,
    tokens,
    transforms as transform_mod,
    verify as verify_mod,
)
from .config import DEFAULT, Settings
from .corpus import SourceFile


@dataclass
class MigrationReport:
    """Everything a run produces, including the numbers that justify it."""

    files_total: int = 0
    files_touched: int = 0
    files_migrated: int = 0
    sites_total: int = 0
    sites_by_codemod: int = 0
    sites_by_model: int = 0
    hunks: int = 0
    clusters: int = 0
    requests: int = 0
    #: Clusters the model itself declined with an ESCALATE block.
    declined: int = 0
    #: Clusters re-asked against the strong model after a verification failure.
    escalations: int = 0
    unverified: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ledger: model_mod.Ledger = field(default_factory=model_mod.Ledger)
    output: Dict[str, str] = field(default_factory=dict)
    transforms_by_cluster: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.ledger.total_tokens

    @property
    def verified(self) -> bool:
        return not self.unverified

    def summary(self) -> dict:
        data = {
            "files_total": self.files_total,
            "files_touched": self.files_touched,
            "files_migrated": self.files_migrated,
            "sites_total": self.sites_total,
            "sites_by_codemod": self.sites_by_codemod,
            "sites_by_model": self.sites_by_model,
            "hunks": self.hunks,
            "clusters": self.clusters,
            "declined": self.declined,
            "escalations": self.escalations,
            "unverified": len(self.unverified),
        }
        data.update(self.ledger.summary())
        return data


class MigrationAgent:
    def __init__(
        self,
        model: Optional[model_mod.Model] = None,
        settings: Settings = DEFAULT,
        strong_model: Optional[model_mod.Model] = None,
    ) -> None:
        self.model = model or model_mod.OfflineModel()
        self.settings = settings
        #: Used only for clusters whose result failed verification. Left None
        #: when there is nothing better to escalate to, in which case failures
        #: are reported rather than silently retried against the same model.
        self.strong_model = strong_model

    # -- the optimised path ------------------------------------------------

    def run(self, files: Sequence[SourceFile]) -> MigrationReport:
        report = MigrationReport(files_total=len(files))
        report.ledger = model_mod.Ledger(pricing=self.settings.pricing)

        results = self._codemod_pass(files, report)
        all_hunks = self._hunk_pass(results, report)
        clusters = self._cluster_pass(all_hunks, report)

        answers = self._ask(clusters, report)
        self._apply(results, clusters, answers, report)
        self._verify(results, report)

        if report.unverified and self.settings.escalation_enabled:
            self._escalate(results, clusters, answers, report)

        report.requests = report.ledger.requests
        return report

    def _escalate(
        self,
        results: Sequence[codemod.CodemodResult],
        clusters: Sequence[cluster_mod.Cluster],
        answers: Dict[str, patch.ClusterAnswer],
        report: MigrationReport,
    ) -> None:
        """Re-ask a stronger model about the clusters that produced bad output.

        Escalation is scoped to the *clusters implicated in a failure*, not to
        the failing files. A file fails because some rule applied to it was
        wrong, and that rule is wrong everywhere it landed -- including in
        files that happened to still parse. Re-asking per file would fix the
        symptom in one place and leave the same bad rule applied elsewhere.

        Without a ``strong_model`` this records why it declined instead of
        retrying the same model and hoping for a different answer.
        """
        failed_paths = {entry.split(":", 1)[0] for entry in report.unverified}
        implicated = [
            c
            for c in clusters
            if any(hunk.path in failed_paths for hunk in c.members)
        ]
        if not implicated:
            return

        if self.strong_model is None:
            report.warnings.append(
                f"{len(failed_paths)} file(s) failed verification and "
                f"{len(implicated)} cluster(s) are implicated, but no strong "
                "model is configured to escalate to"
            )
            return

        for attempt in range(self.settings.escalation.strong_attempts):
            items = [self._pack_item(c) for c in implicated]
            requests = (
                packer.pack(items, self.settings.packing)
                if self.settings.packing_enabled
                else [packer.Request(items=[item]) for item in items]
            )
            for request in requests:
                system, user = prompts.build_request(
                    request.clusters_block(), request.exemplars()
                )
                response = self.strong_model.complete(system, user)
                report.ledger.record(response.usage)
                parsed = patch.parse(response.text, request.fingerprints)
                report.warnings.extend(parsed.warnings)
                answers.update(parsed.answers)
                report.escalations += len(parsed.answers)

            for fingerprint, answer in answers.items():
                report.transforms_by_cluster[fingerprint] = [
                    t.describe() for t in answer.transforms
                ]

            # Re-apply from the post-codemod text, never from the already
            # patched output -- the bad transform is still in there.
            report.output.clear()
            report.files_migrated = 0
            report.unverified.clear()
            self._apply(results, clusters, answers, report)
            self._verify(results, report)
            if not report.unverified:
                return

    def _codemod_pass(
        self, files: Sequence[SourceFile], report: MigrationReport
    ) -> List[codemod.CodemodResult]:
        """Resolve everything mechanical, or nothing if the lever is off."""
        results: List[codemod.CodemodResult] = []
        for source in files:
            if self.settings.deterministic_pass:
                result = codemod.run(source.path, source.text)
            else:
                # Lever off: every site becomes model work, which is exactly
                # what the ablation needs to price.
                try:
                    findings = rules.scan(source.text)
                except SyntaxError as exc:
                    findings = []
                    results.append(
                        codemod.CodemodResult(
                            path=source.path,
                            original=source.text,
                            text=source.text,
                            error=str(exc),
                        )
                    )
                    continue
                result = codemod.CodemodResult(
                    path=source.path,
                    original=source.text,
                    text=source.text,
                    unresolved=findings,
                )
            results.append(result)

        report.files_touched = sum(1 for r in results if r.resolved or r.unresolved)
        report.sites_by_codemod = sum(len(r.resolved) for r in results)
        report.sites_total = sum(len(r.resolved) + len(r.unresolved) for r in results)
        report.sites_by_model = sum(len(r.unresolved) for r in results)
        for result in results:
            if result.error:
                report.warnings.append(f"{result.path}: {result.error}")
        return results

    def _hunk_pass(
        self, results: Sequence[codemod.CodemodResult], report: MigrationReport
    ) -> List[hunks_mod.Hunk]:
        collected: List[hunks_mod.Hunk] = []
        for result in results:
            if not result.unresolved:
                continue
            if self.settings.windowing:
                collected.extend(
                    hunks_mod.extract(
                        result.path, result.text, result.unresolved, self.settings.hunks
                    )
                )
            else:
                # Lever off: one hunk per file, covering the whole file. This
                # is what "just send the file" costs once you already have the
                # rest of the funnel, and it isolates windowing from the
                # deterministic pass -- two levers that are easy to conflate
                # because both reduce bytes sent.
                lines = result.text.splitlines()
                collected.append(
                    hunks_mod.Hunk(
                        path=result.path,
                        start_line=1,
                        end_line=max(1, len(lines)),
                        text=result.text,
                        shapes=tuple(f.shape for f in result.unresolved),
                        reasons=tuple(
                            dict.fromkeys(f.reason for f in result.unresolved if f.reason)
                        ),
                        signature=tuple(
                            part for f in result.unresolved for part in f.signature
                        ),
                        site_lines=tuple(sorted(f.line for f in result.unresolved)),
                    )
                )
        for hunk in collected:
            if hunk.oversized:
                report.warnings.append(
                    f"{hunk.path}:{hunk.start_line} oversized hunk, flagged for review"
                )
        report.hunks = len(collected)
        return collected

    def _cluster_pass(
        self, all_hunks: Sequence[hunks_mod.Hunk], report: MigrationReport
    ) -> List[cluster_mod.Cluster]:
        if self.settings.clustering:
            clusters = cluster_mod.build(all_hunks, self.settings.clusters)
        else:
            # Lever off: one cluster per hunk, so every site is asked about
            # individually. This is the single most expensive lever to lose.
            clusters = [
                cluster_mod.Cluster(
                    fingerprint=f"solo-{index:04d}",
                    shapes=tuple(dict.fromkeys(hunk.shapes)),
                    members=[hunk],
                    singleton=True,
                )
                for index, hunk in enumerate(all_hunks)
            ]
        report.clusters = len(clusters)
        return clusters

    def _ask(
        self, clusters: Sequence[cluster_mod.Cluster], report: MigrationReport
    ) -> Dict[str, patch.ClusterAnswer]:
        items = [self._pack_item(c) for c in clusters]

        if self.settings.packing_enabled:
            requests = packer.pack(items, self.settings.packing)
        else:
            requests = [packer.Request(items=[item]) for item in items]

        answers: Dict[str, patch.ClusterAnswer] = {}
        for request in requests:
            system, user = prompts.build_request(
                request.clusters_block(), request.exemplars()
            )
            response = self.model.complete(system, user)
            report.ledger.record(response.usage)

            parsed = patch.parse(response.text, request.fingerprints)
            report.warnings.extend(parsed.warnings)
            answers.update(parsed.answers)

        report.requests = report.ledger.requests
        report.declined = sum(1 for a in answers.values() if a.is_escalation)
        for fingerprint, answer in answers.items():
            report.transforms_by_cluster[fingerprint] = [
                t.describe() for t in answer.transforms
            ]
        return answers

    def _pack_item(self, cluster: cluster_mod.Cluster) -> packer.PackItem:
        representatives = cluster.representatives(self.settings.clusters)
        keys = []
        for hunk in cluster.members:
            keys.extend(hunk.signature)
            keys.extend(shape.value for shape in hunk.shapes)

        chosen: List[str] = []
        if self.settings.exemplars_enabled:
            chosen = [
                exemplar.render()
                for exemplar in exemplar_mod.select(keys, self.settings.exemplars)
            ]

        reasons: List[str] = []
        for hunk in representatives:
            for reason in hunk.reasons:
                if reason not in reasons:
                    reasons.append(reason)

        body = prompts.render_cluster(
            fingerprint=cluster.fingerprint,
            shapes=[shape.value for shape in cluster.shapes],
            site_count=cluster.size,
            reasons=reasons,
            representatives=[hunk.render() for hunk in representatives],
        )
        return packer.PackItem(
            fingerprint=cluster.fingerprint, body=body, exemplars=chosen
        )

    def _apply(
        self,
        results: Sequence[codemod.CodemodResult],
        clusters: Sequence[cluster_mod.Cluster],
        answers: Dict[str, patch.ClusterAnswer],
        report: MigrationReport,
    ) -> None:
        """Fan the per-cluster rules back out to every member site."""
        by_path: Dict[str, List[Tuple[transform_mod.Transform, Tuple[int, ...]]]] = {}

        for cluster in clusters:
            answer = answers.get(cluster.fingerprint)
            if answer is None or answer.is_escalation or not answer.transforms:
                continue
            for hunk in cluster.followers:
                bucket = by_path.setdefault(hunk.path, [])
                for transform in answer.transforms:
                    bucket.append((transform, hunk.site_lines))

        for result in results:
            pending = by_path.get(result.path)
            text = result.text
            if pending:
                # Group by target lines so each distinct line set is applied
                # once with its full transform list, preserving order.
                grouped: Dict[Tuple[int, ...], List[transform_mod.Transform]] = {}
                for transform, lines in pending:
                    grouped.setdefault(lines, []).append(transform)
                for lines, batch in grouped.items():
                    applied = transform_mod.apply(text, batch, line_filter=lines)
                    if applied.errors:
                        report.warnings.extend(
                            f"{result.path}: {err}" for err in applied.errors
                        )
                        continue
                    text = applied.text
                text = codemod.finalize(text)

            report.output[result.path] = text
            if text != result.original:
                report.files_migrated += 1

    def _verify(
        self, results: Sequence[codemod.CodemodResult], report: MigrationReport
    ) -> None:
        for result in results:
            text = report.output.get(result.path, result.text)
            if text == result.original and not result.resolved and not result.unresolved:
                continue  # untouched file, nothing to verify
            outcome = verify_mod.verify(result.path, text)
            if not outcome.ok:
                report.unverified.append(f"{result.path}: {'; '.join(outcome.failures)}")

    # -- the baseline ------------------------------------------------------

    def run_naive(self, files: Sequence[SourceFile]) -> MigrationReport:
        """Whole file in, whole file out, full rulebook every time.

        Deliberately implemented as the honest version of the obvious
        approach, not a straw man. It uses a substring test to decide which
        files are relevant -- which is what you write before you own an AST
        pass -- and it pays the full rulebook on every request because it has
        no notion of a cluster to amortise it over.

        The output token count is measured against the *actually migrated*
        text produced by the optimised run where available, so the baseline is
        not penalised for hypothetically returning something longer.
        """
        report = MigrationReport(files_total=len(files))
        report.ledger = model_mod.Ledger(pricing=self.settings.pricing)
        system = prompts.NAIVE_SYSTEM_PROMPT

        for source in files:
            if "requests" not in source.text:
                continue
            report.files_touched += 1

            user = (
                f"File: {source.path}\n\n"
                f"{source.text}\n\n"
                "Return the complete migrated file."
            )
            usage = model_mod.Usage(
                input_tokens=tokens.count(system) + tokens.count(user),
                output_tokens=tokens.count(source.text),
                prefix_cached_tokens=0,
                model=self.settings.escalation.strong_model,
            )
            report.ledger.record(usage)
            report.output[source.path] = source.text

        report.requests = report.ledger.requests
        report.files_migrated = report.files_touched
        return report
