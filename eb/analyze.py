"""The three predictions from experiment-matched-confidence.md:

1. Gain from asking is large on Set A, near zero on Set B.
2. Pre-ask confidence does not predict the gain (Corollary 1 in observable form).
3. The cluster diagnostic does predict the gain.

Usage: python -m eb.analyze results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

CORRECT = {"correct": 1.0, "wrong": 0.0, "hedged": 0.0}  # strict grading


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(records: list[dict], condition: str) -> float:
    if not records:
        return float("nan")
    return sum(CORRECT[r[condition]["grade"]] for r in records) / len(records)


def per_item_gain(records: list[dict], ask_condition: str = "self_ask") -> list[float]:
    return [CORRECT[r[ask_condition]["grade"]] - CORRECT[r["answer_now"]["grade"]] for r in records]


def pearson(xs: list[float], ys: list[float]) -> float:
    from scipy.stats import pearsonr

    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return pearsonr(xs, ys)[0]


def analyze(records: list[dict]) -> dict:
    setA = [r for r in records if r["set"] == "A"]
    setB = [r for r in records if r["set"] == "B"]

    result = {
        "n_setA": len(setA),
        "n_setB": len(setB),
        "setA_accuracy": {
            "answer_now": accuracy(setA, "answer_now"),
            "oracle_clarify": accuracy(setA, "oracle_clarify"),
            "self_ask": accuracy(setA, "self_ask"),
        },
        "setB_accuracy": {
            "answer_now": accuracy(setB, "answer_now"),
            "self_ask": accuracy(setB, "self_ask"),
        },
    }

    # Prediction 1: gain by set. Oracle gain is the primary/ceiling measure
    # (experiment-ask-protocol.md); self-ask gain is the realized competence.
    result["prediction_1_gain"] = {
        "setA_oracle_gain": result["setA_accuracy"]["oracle_clarify"] - result["setA_accuracy"]["answer_now"],
        "setA_selfask_gain": result["setA_accuracy"]["self_ask"] - result["setA_accuracy"]["answer_now"],
        "setB_selfask_gain": result["setB_accuracy"]["self_ask"] - result["setB_accuracy"]["answer_now"],
    }

    # Predictions 2 and 3 pool both sets, using self-ask gain per item
    # (oracle-clarify has no Set B analogue by construction).
    pooled = setA + setB
    gains = per_item_gain(pooled, "self_ask")
    confidences = [r["confidence"] for r in pooled]
    diag_entropy = [r["diagnostic"]["entropy"] for r in pooled]
    diag_second = [r["diagnostic"]["second_largest_frac"] for r in pooled]

    result["prediction_2_confidence_vs_gain_corr"] = pearson(confidences, gains)
    result["prediction_3_entropy_vs_gain_corr"] = pearson(diag_entropy, gains)
    result["prediction_3_second_largest_vs_gain_corr"] = pearson(diag_second, gains)

    # Hedge rate, recorded per experiment-ask-protocol.md's "third response category".
    hedge_count = sum(1 for r in setA if r["answer_now"]["grade"] == "hedged")
    result["setA_answer_now_hedge_rate"] = hedge_count / len(setA) if setA else float("nan")

    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    args = parser.parse_args(argv)
    records = load_records(args.results_path)
    result = analyze(records)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
