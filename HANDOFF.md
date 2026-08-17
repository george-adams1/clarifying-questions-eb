# Handoff: E-B harness, for whoever (whichever Claude) picks this up next

You're likely running on more powerful hardware (an H100 cluster) than this
was originally built and run on (a Vulcan/Alliance Canada cluster with L40S
GPUs, 48GB VRAM each). This document is written for you specifically —
what this is, what's already been verified, what's still in progress
elsewhere, the mistakes already made so you don't repeat them, and what's
worth doing differently now that you have better hardware.

**Read `experiments-run-with-results.md` first** for the full precise
account of what was tested and what was found. This document is a faster
orientation and a list of gotchas; that one is the source of truth.

## What this is, in one paragraph

E-B is an experiment testing whether a language model's verbalized
confidence score can predict when asking a clarifying question will help.
Two question sets are matched on confidence (both land in the 50-60% band):
Set A is genuinely ambiguous (AmbigQA, two valid readings per question),
Set B is just hard trivia (TriviaQA, one reading). If confidence really
can't distinguish "I'm unsure which reading you mean" from "I don't know
this fact," then asking should help a lot on Set A and do ~nothing on Set
B — a pattern invisible to the confidence number itself. Full design in
`experiment-matched-confidence.md` and `experiment-ask-protocol.md`.

## Quick start

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v     # 13 tests, offline, no GPU/API needed, ~1s
```

Real run:
```
.venv/bin/python -m eb.run_experiment \
  --model <hf-model-name> [--device-map auto for multi-GPU sharding] \
  --setA-all-splits --setB-pool-limit <N> \
  --n-per-set 20 --diagnostic-n 8 --seed 0 \
  --out results.jsonl
.venv/bin/python -m eb.analyze results.jsonl
```

## Gotchas already discovered — read before "fixing" anything

1. **Blind pre-answer confidence elicitation is broken for capable models.**
   `experiment-ask-protocol.md` specifies eliciting confidence via a bare
   "state your confidence as a number" prompt, before the model answers.
   This was tried first, exactly as specified, and failed completely: both
   Qwen3-8B and (less severely) Qwen2.5-72B-Instruct collapse to reporting
   ~85-95% confidence on nearly everything, regardless of actual difficulty
   — verified even at sampling temperature 1.0. The harness now elicits
   confidence *after* an attempted answer instead (model answers, then
   rates confidence in that specific answer) — see `eb/screening.py`'s
   module docstring and `pre_registration.md` for the full account. If you
   revert this "to match the spec," you will likely reproduce the same
   failure (0 matched items) on any similarly capable model. If you're
   testing a very different kind of model and it doesn't show this
   collapse, reverting might be worth it — but check first, don't assume.

2. **Set A has a hard ceiling: 2,956 usable candidates, period.** This
   comes from AmbigQA — only items with exactly two disambiguated readings
   are used (`eb/data_ambigqa.py`), and there are only 2,956 of those across
   both the `light` config's validation (587) and train (2,369) splits
   combined. No amount of compute changes this number. At the observed
   ~0.9-1% confidence-band hit rate, the realistic ceiling on *matched*
   Set A items is roughly 26-30, not more, however many H100s you have. Use
   `--setA-all-splits --setA-full-scan` to scan the entire pool (no early
   stop) and get however many that turns out to be — that's already running
   as a separate check (see "Concurrently in progress" below).

3. **Set B's confidence distribution is model-dependent and can be sharply
   bimodal.** For Qwen3-8B specifically, a 60-item deep-dive showed the
   in-band (50-60%) hit rate was close to 0% — the model was either very
   confident (90-100%) or very unconfident (0%) on hard trivia, almost
   nothing in between. Qwen2.5-72B-Instruct had a healthier but still low
   rate (~0.25%). TriviaQA has ~99,000 candidates available in total
   (11,313 validation + 87,622 train) if you need a much bigger pool to find
   enough matches — but check the actual hit rate on your model with a
   small sample (`eb/screening.py::elicit_confidence_for_answer`) before
   committing to a huge pool; don't assume the rate transfers across models.

4. **Qwen3 (and similar hybrid-reasoning models) default to a long
   chain-of-thought "thinking" mode.** Already handled —
   `eb/model_client.py::LocalHFClient.complete()` passes
   `enable_thinking=False` to the chat template when supported, with a
   `TypeError` fallback for models that don't have the kwarg. If you add a
   new model family with its own "thinking" toggle, you may need to extend
   this.

5. **Multi-GPU sharding (`--device-map auto`) works but is inefficient.**
   It uses `transformers`/`accelerate`'s naive layer-pipeline sharding —
   only one GPU computes at a time as activations flow through the pipeline
   of shards, the rest idle. Verified working (Qwen2.5-72B-Instruct sharded
   cleanly across 4× L40S, ~36GB each), but throughput was only ~0.9s/call
   versus ~0.14s/call for Qwen3-8B on a single GPU — a ~6.5x slowdown for a
   ~9x parameter increase, reasonable but not great. **On an H100 cluster,
   strongly consider swapping `LocalHFClient` for a real inference server
   with tensor parallelism** (vLLM is the obvious choice — `pip install
   vllm`, serve the model, hit it over HTTP or the Python client) instead
   of raw `transformers.generate()`. This should give much better multi-GPU
   utilization and let you run bigger models or bigger samples in the same
   wall-clock budget. This wasn't done here for time reasons, not because
   it's a bad idea — it's probably the single highest-leverage change you
   can make given better hardware. If you do this, keep the `ModelClient`
   interface (`eb/model_client.py`) — everything else in the harness talks
   to it, not to `transformers` directly, so a `VLLMClient` implementing
   `complete(system, user, temperature) -> str` should drop in cleanly.

6. **The leak audit is real, not decorative.** Every self-ask reply is
   checked post-hoc for whether it happens to contain the intended answer
   — this caught a genuine leak in one early run (a weak model's
   user-simulator independently regenerated the correct answer from its own
   knowledge of the rewritten question, not from anything in its prompt).
   Don't skip or weaken this if you extend the harness.

## Concurrently in progress (as of this handoff, on the Vulcan cluster)

Two runs were launched to find Set A's true match ceiling (see gotcha #2)
and were still running when this repo was created — their results are
**not** reflected in `experiments-run-with-results.md` yet:
- Qwen3-8B, `--setA-full-scan` (fast, ~20-25 min)
- Qwen2.5-72B-Instruct, `--setA-full-scan` (slow, likely several hours
  including a multi-hour queue wait)

If you're picking this up to run something new rather than continue that
specific check, you can ignore this — just know the numbers in the results
doc for Set A's ceiling are estimates (26-30), not yet confirmed exactly.

## What's genuinely worth doing with better hardware

- **A frontier-scale model.** Every run so far used open-weight models
  chosen to be cheap to validate the harness against (largest so far:
  Qwen2.5-72B-Instruct), not the frontier model the paper's real numbers
  should come from (`pre_registration.md` says this explicitly). If your
  H100 cluster can run something meaningfully larger or you have API access
  to a frontier model, that's the most valuable next step for the paper
  itself, not just a bigger version of what's already been done.
- **Bigger Set B samples**, given Set B has ~99,000 candidates available and
  you've likely got far more compute headroom than 4×L40S.
- **The vLLM swap** (gotcha #5) if you want everything faster rather than
  just bigger.
- **Prediction 2/3 correlations need more data.** Both real runs so far had
  n≈20 per set (39-40 pooled items) — enough to see the right direction but
  not enough to trust the correlation magnitudes. This is the most direct
  case for "just run more."

## Files in this repo

```
eb/                          the harness itself
tests/                       13 offline tests against a mock model
requirements.txt
README.md                    general usage instructions
pre_registration.md          frozen experimental parameters + documented deviations
experiments-run.md           precise methodology, no results
experiments-run-with-results.md   same, plus actual numbers from the two completed runs
experiment-matched-confidence.md  original design doc (primary spec)
experiment-ask-protocol.md        original design doc (mechanics spec)
```

Not included (by request, when this repo was created): the paper draft
itself and `paper2_plan.md` (unpublished, co-authored, explicitly flagged
in `paper2_plan.md` as needing to stay private pending a companion paper's
arXiv posting), and raw `results_*.jsonl` output files from prior runs.
