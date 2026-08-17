"""Per-question typed diagnostic (experiment-matched-confidence.md):

"sample n answers (e.g. n=10 at temperature 1), cluster by semantic
equivalence, and record the spread over reading-level clusters. The
simplest statistics are the entropy over clusters or the size of the
second-largest cluster."

Open item resolved here (documented default, see pre_registration.md):
the clustering rule is alias-list membership against the item's own known
readings, not free-form string clustering -- this is judge-free and
reproducible. Set A items have two reading buckets; Set B items have one
(there is nothing to be ambiguous about, so low spread is the predicted
sanity check, not a measurement gap). Samples matching neither reading's
alias list fall into an "other" bucket and are recorded, not discarded.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .grading import contains_alias


@dataclass
class DiagnosticResult:
    n_samples: int
    cluster_counts: dict  # bucket name -> count
    entropy: float
    second_largest_frac: float


def _cluster_bucket(sample: str, buckets: dict) -> str:
    for name, aliases in buckets.items():
        if contains_alias(sample, aliases):
            return name
    return "other"


def run_diagnostic(client, question: str, buckets: dict, n: int = 10, temperature: float = 1.0) -> DiagnosticResult:
    """buckets: e.g. {"reading_a": [...aliases], "reading_b": [...aliases]}."""
    system = "Answer the question as concisely as possible: a short phrase, no explanation."
    samples = [client.complete(system, question, temperature=temperature) for _ in range(n)]
    counts = Counter(_cluster_bucket(s, buckets) for s in samples)

    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)

    sorted_counts = sorted(counts.values(), reverse=True)
    second_largest = sorted_counts[1] if len(sorted_counts) > 1 else 0
    second_largest_frac = second_largest / total if total else 0.0

    return DiagnosticResult(
        n_samples=total,
        cluster_counts=dict(counts),
        entropy=entropy,
        second_largest_frac=second_largest_frac,
    )
