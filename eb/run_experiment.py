"""Orchestrates E-B: screening -> matched sets -> four conditions -> diagnostic -> JSONL.

Usage:
    python -m eb.run_experiment --mock --out results.jsonl
    python -m eb.run_experiment --model Qwen/Qwen2.5-0.5B-Instruct --n-per-set 25 --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from . import conditions as C
from .data_ambigqa import load_ambigqa_candidates
from .data_setb import load_setb_candidates
from .diagnostic import run_diagnostic
from .grading import grade_ambiguous, grade_single
from .model_client import LocalHFClient, MockClient
from .screening import build_matched_sets


def run_set_a_item(client, item, diagnostic_n: int, diagnostic_temp: float) -> dict:
    intended_reading = item.reading_a if item.intended == "a" else item.reading_b
    other_reading = item.reading_b if item.intended == "a" else item.reading_a

    answer_now_resp = item.answer_now_response  # reused from screening, see screening.py's module docstring
    oracle_resp = C.oracle_clarify(client, intended_reading.question)
    self_ask_res = C.self_ask_set_a(client, item.ambiguous_question, intended_reading.question, intended_reading.answers)
    free_choice_res = C.free_choice_set_a(client, item.ambiguous_question, intended_reading.question, intended_reading.answers)
    diag = run_diagnostic(
        client,
        item.ambiguous_question,
        {"reading_a": item.reading_a.answers, "reading_b": item.reading_b.answers},
        n=diagnostic_n,
        temperature=diagnostic_temp,
    )

    def g(resp: str) -> str:
        return grade_ambiguous(resp, intended_reading.answers, other_reading.answers)

    return {
        "set": "A",
        "id": item.id,
        "ambiguous_question": item.ambiguous_question,
        "intended": item.intended,
        "confidence": item.confidence,
        "answer_now": {"response": answer_now_resp, "grade": g(answer_now_resp)},
        "oracle_clarify": {"response": oracle_resp, "grade": g(oracle_resp)},
        "self_ask": {
            "clarifying_question": self_ask_res.clarifying_question,
            "simulator_reply": self_ask_res.simulator_reply,
            "leaked": self_ask_res.leaked,
            "final_answer": self_ask_res.final_answer,
            "grade": g(self_ask_res.final_answer),
        },
        "free_choice": {
            "asked": free_choice_res.asked,
            "clarifying_question": free_choice_res.clarifying_question,
            "simulator_reply": free_choice_res.simulator_reply,
            "leaked": free_choice_res.leaked,
            "final_answer": free_choice_res.final_answer,
            "grade": g(free_choice_res.final_answer),
        },
        "diagnostic": diag.__dict__,
    }


def run_set_b_item(client, item, diagnostic_n: int, diagnostic_temp: float) -> dict:
    answer_now_resp = item.answer_now_response  # reused from screening, see screening.py's module docstring
    self_ask_res = C.self_ask_set_b(client, item.question)
    free_choice_res = C.free_choice_set_b(client, item.question)
    diag = run_diagnostic(client, item.question, {"reading": item.answers}, n=diagnostic_n, temperature=diagnostic_temp)

    def g(resp: str) -> str:
        return grade_single(resp, item.answers)

    return {
        "set": "B",
        "id": item.id,
        "question": item.question,
        "confidence": item.confidence,
        "answer_now": {"response": answer_now_resp, "grade": g(answer_now_resp)},
        "self_ask": {
            "clarifying_question": self_ask_res.clarifying_question,
            "simulator_reply": self_ask_res.simulator_reply,
            "leaked": False,
            "final_answer": self_ask_res.final_answer,
            "grade": g(self_ask_res.final_answer),
        },
        "free_choice": {
            "asked": free_choice_res.asked,
            "clarifying_question": free_choice_res.clarifying_question,
            "simulator_reply": free_choice_res.simulator_reply,
            "leaked": False,
            "final_answer": free_choice_res.final_answer,
            "grade": g(free_choice_res.final_answer),
        },
        "diagnostic": diag.__dict__,
    }


def build_client(args):
    if args.mock:
        from tests.fixtures import make_mock_client

        return make_mock_client(args.mock)
    return LocalHFClient(model_name=args.model, device_map=args.device_map)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", choices=["flat", "typed"], default=None, help="use MockClient persona instead of a real model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device-map", default=None, help='e.g. "auto" to shard a large model across multiple visible GPUs')
    parser.add_argument("--n-per-set", type=int, default=25)
    parser.add_argument("--band-low", type=float, default=50.0)
    parser.add_argument("--band-high", type=float, default=60.0)
    parser.add_argument("--diagnostic-n", type=int, default=10)
    parser.add_argument("--diagnostic-temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-limit", type=int, default=None, help="cap both pools (for quick smoke tests); overridden by --setA-pool-limit/--setB-pool-limit if given")
    parser.add_argument("--setA-pool-limit", type=int, default=None)
    parser.add_argument("--setB-pool-limit", type=int, default=None)
    parser.add_argument("--setA-all-splits", action="store_true", help="combine AmbigQA train+validation (2,956 usable two-reading items total, vs. 587 in validation alone)")
    parser.add_argument("--setA-full-scan", action="store_true", help="ignore --n-per-set for Set A: screen the entire Set A pool (no early stop) and keep every in-band match, whatever that count turns out to be")
    parser.add_argument("--out", default="results.jsonl")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    client = build_client(args)

    print("Loading candidate pools...", file=sys.stderr)
    if args.setA_all_splits:
        setA_pool = load_ambigqa_candidates(split="validation") + load_ambigqa_candidates(split="train")
    else:
        setA_pool = load_ambigqa_candidates()
    setB_pool = load_setb_candidates()

    a_limit = args.setA_pool_limit or args.pool_limit
    b_limit = args.setB_pool_limit or args.pool_limit
    if a_limit:
        setA_pool = setA_pool[:a_limit]
    if b_limit:
        setB_pool = setB_pool[:b_limit]

    setA_target = len(setA_pool) if args.setA_full_scan else None
    print(f"Set A pool: {len(setA_pool)}, Set B pool: {len(setB_pool)}. Screening...", file=sys.stderr)
    if args.setA_full_scan:
        print(f"Set A: full-scan mode, no early stop (target = pool size = {setA_target})", file=sys.stderr)
    setA, setB = build_matched_sets(
        client, setA_pool, setB_pool, target_per_set=args.n_per_set, band=(args.band_low, args.band_high), rng=rng,
        setA_target=setA_target,
    )
    print(f"Matched Set A: {len(setA)}, Set B: {len(setB)}", file=sys.stderr)

    n_leaked = 0
    n_total_selfask = 0
    with open(args.out, "w") as f:
        for item in setA:
            record = run_set_a_item(client, item, args.diagnostic_n, args.diagnostic_temp)
            n_total_selfask += 1
            if record["self_ask"]["leaked"]:
                n_leaked += 1
            f.write(json.dumps(record) + "\n")
        for item in setB:
            record = run_set_b_item(client, item, args.diagnostic_n, args.diagnostic_temp)
            f.write(json.dumps(record) + "\n")

    drop_rate = n_leaked / n_total_selfask if n_total_selfask else 0.0
    print(f"Leak audit: {n_leaked}/{n_total_selfask} self-ask items leaked ({drop_rate:.1%}).", file=sys.stderr)
    if drop_rate > 0.02:
        print("WARNING: leak drop rate exceeds 2% -- rewrite set needs inspection before this run counts (per experiment-ask-protocol.md).", file=sys.stderr)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
