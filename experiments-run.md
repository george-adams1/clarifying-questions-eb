# E-B: what was actually run

This document describes, precisely, the experiment implemented and executed
for E-B (the matched-confidence ask-or-answer test), as distinct from what
the design docs (`experiment-matched-confidence.md`,
`experiment-ask-protocol.md`) originally specified. Where the executed
version deviates from the spec, that's called out explicitly with the reason.
No results are in this document — see `experiments-run-with-results.md` for
the same content plus the actual numbers.

## What the experiment tests

The paper's Corollary 1 claims that no function of a model's collapsed
confidence score can decide whether asking a clarifying question is worth
it, because confidence can be low for two structurally different reasons:
genuine ambiguity about which reading of a question is meant (resolvable by
asking), or ordinary not-knowing a fact (not resolvable by asking). E-B
tests this by building two question sets matched on confidence and checking
whether the *gain* from asking differs sharply between them even though the
confidence score can't.

- **Set A (ambiguous).** Questions with exactly two valid readings, each
  with its own known answer, sourced from AmbigQA.
- **Set B (unambiguous, hard).** Single-reading trivia questions, screened
  to land in the same confidence band as Set A, sourced from TriviaQA.

Both sets are run through four conditions (answer-now, oracle-clarify,
self-ask, free-choice) and a typed diagnostic (repeated sampling, clustered
by reading). Full mechanics in `experiment-ask-protocol.md`; harness
implementation in `eb/`.

## Deviations from the design docs, and why

### 1. Confidence elicitation timing

**Spec** (`experiment-ask-protocol.md`, "Elicitation order"): confidence
elicited in a stateless call on the bare question, before any condition
runs — a blind pre-answer guess.

**What was run instead:** the model answers the question first, then rates
its confidence in that specific answer (`eb/screening.py::CONFIDENCE_SYSTEM`).

**Why:** the blind pre-answer prompt was tried first, exactly as specified,
against Qwen3-8B. It failed completely — the model reported ~95% confidence
on essentially every question regardless of actual difficulty, verified
even at sampling temperature 1.0 (the per-token probability mass on "95"
was large enough that sampling rarely escaped it). Across 500 candidates
per set under this elicitation, zero landed in the 50-60% band. Switching
to post-answer elicitation (the standard, generally better-calibrated
pattern) produced a real, usable distribution. As a consequence, condition
1 (answer-now) reuses the answer generated during screening rather than
asking the model again independently, both to save a call and because a
second independently-sampled answer could disagree with the one confidence
was actually rated against.

### 2. Diagnostic sample size

**Spec**: n=10 samples at temperature 1 for the typed diagnostic.

**What was run:** n=8, in every real run performed. No specific reason
beyond a minor, undocumented reduction when the CLI default was set;
noted here for the record since exact reproduction should use n=8 to match,
not n=10.

### 3. Set A candidate pool

AmbigQA items can have more than two disambiguated readings. The harness
only uses items with *exactly* two (`eb/data_ambigqa.py`), matching the
paper's two-reading framing. This leaves 2,956 usable candidates total
across AmbigQA's `light` config validation (587) and train (2,369) splits
combined — the harness's final runs use both splits together
(`--setA-all-splits`), not the single-split default.

### 4. Set B source

Not specified by the design docs beyond "obscure trivia screened to land
in the same confidence band." TriviaQA (`mandarjoshi/trivia_qa`,
`unfiltered.nocontext` config, validation split — 11,313 candidates
available) was used as a documented default.

## Harness mechanics (brief; see `eb/*.py` docstrings for full detail)

- **Grading**: normalized exact match (lowercase, strip punctuation and
  articles, alias-list containment) against the pre-fixed intended reading's
  answer aliases only. Three outcomes — correct / wrong / hedged — with
  hedge scored as wrong under the strict grading used for headline numbers.
  Hedge detection is mechanical: an answer counts as a hedge iff it contains
  alias hits from *both* readings.
- **Leak firewall**: the self-ask condition's user-simulator is given the
  ambiguous question, the disambiguated rewrite, and the model's own
  clarifying question — never the answer field. Every self-ask reply is
  additionally audited post-hoc for whether it happens to contain the
  answer anyway (a model that already knows the rewritten question's answer
  can regenerate it independently of what's in its prompt); the audit
  result is recorded per-item and summarized as a leak rate.
- **Diagnostic clustering**: each of the n sampled answers is bucketed by
  alias-list containment against the item's own known readings (`reading_a`
  / `reading_b` for Set A, a single `reading` bucket for Set B), not by a
  semantic-equivalence judge. Samples matching neither bucket go to `other`
  and are recorded, not dropped.
- **Model client**: a pluggable interface (`eb/model_client.py`) — a
  `MockClient` for free offline testing (used in the 13-test pytest suite),
  and `LocalHFClient` for real models via `transformers`, with optional
  `device_map="auto"` sharding across multiple GPUs for models too large
  for one card. Qwen3 and similar hybrid-reasoning models default to a long
  chain-of-thought "thinking" mode before answering; this is explicitly
  disabled (`enable_thinking=False`) since the harness's prompts want a
  short direct phrase.

## Infrastructure

All real-model runs executed on **Vulcan** (a SLURM cluster, Digital
Research Alliance of Canada), account `aip-irina`, user `georgea`. GPU
nodes carry 4× NVIDIA L40S (48GB VRAM each). Model weights cached under
`/scratch/georgea/hf_cache`. No API cost — everything ran on local
inference against downloaded open-weight models, not a paid API.

## Final experimental configuration (the two runs that count)

Both use band 50-60% inclusive, target 20 matched items per set, seed 0,
diagnostic n=8 at temperature 1.0, strict grading. Confidence is elicited
per the deviation above (post-answer).

**Run 1 — Qwen3-8B**, single GPU:
```
python -m eb.run_experiment --model Qwen/Qwen3-8B \
  --setA-all-splits --setB-pool-limit 8000 \
  --n-per-set 20 --diagnostic-n 8 --seed 0 \
  --out results_qwen3_8b_v3.jsonl
```
SLURM job 479685, ran 2026-08-13 15:26:51 to 15:47:10 (20m19s).

**Run 2 — Qwen2.5-72B-Instruct**, sharded across 4× L40S GPUs
(`--device-map auto`, ~145GB weights in bf16 spread across the 4 cards):
```
python -m eb.run_experiment --model Qwen/Qwen2.5-72B-Instruct \
  --device-map auto --setA-all-splits --setB-pool-limit 10000 \
  --n-per-set 20 --diagnostic-n 8 --seed 0 \
  --out results_qwen2.5_72b_full.jsonl
```
Submitted as SLURM batch job 480455 (`sbatch`, 8-hour budget, so it would
survive independent of any interactive session), landed on partition
`gpubase_bygpu_b2`. Ran 2026-08-13 19:03:01 to 2026-08-14 01:24:01
(6h21m00s), COMPLETED, exit code 0.

## Development and iteration log

The two runs above didn't happen on the first try. In order:

1. **Qwen2.5-0.5B-Instruct**, single GPU, pool-limit 150, n-per-set 10 —
   pure mechanics smoke test. Matched 10/10 both sets, but accuracy was
   near-zero across every condition (a 0.5B model doesn't know most
   AmbigQA/TriviaQA answers). Confirmed the leak audit works for real: it
   caught a genuine leak where the user-simulator, playing along with a
   badly-formed clarifying question, independently regenerated the correct
   answer from its own knowledge of the rewritten question.
2. **Qwen2.5-7B-Instruct**, single GPU — load/generation sanity check only,
   never run as a full experiment.
3. **Qwen3-8B**, blind pre-answer confidence (per original spec), pool-limit
   250 both sets — 0/0 matched. This is what surfaced the confidence-collapse
   problem described in deviation #1 above.
4. **Qwen3-8B**, post-answer confidence, pool 500/500 — 5/0 matched (Set B's
   confidence turned out to be sharply bimodal: a 60-item deep-dive showed
   0/60 landing in-band, essentially all mass at 0 or 90-100).
5. **Qwen3-8B**, full pools (2,956 / 8,000) — **20/20 matched, the reported
   run**.
6. **Qwen2.5-72B-Instruct**, 4-GPU smoke test — verified sharded loading
   (~35-37GB per GPU) and sampled a 40-item confidence distribution (2/40
   in-band, healthier spread than Qwen3-8B).
7. **Qwen2.5-72B-Instruct**, pools 800/1200 — 7/3 matched, too small to be
   informative on its own; used to estimate real hit rates (~0.9% Set A,
   ~0.25% Set B) for sizing the final run.
8. **Qwen2.5-72B-Instruct**, full pools (2,956 / 10,000), submitted as a
   proper batch job — **20/19 matched, the reported run**.

Throughput differed sharply by model size: roughly 0.14s/generation call for
Qwen3-8B on one GPU, versus roughly 0.9s/call for Qwen2.5-72B-Instruct
sharded across 4 GPUs (only one GPU actively computes at a time as
activations flow through the naive layer-pipeline split; this is not tensor
parallelism).

## Known limitations of this run

- n=20 (or 19) per set is small. Correlation numbers in particular (used
  for predictions 2 and 3) should not be read as statistically robust.
- Neither model is the frontier model the paper's real reported numbers
  should come from (see `pre_registration.md`) — both were chosen to make
  this harness cheap and fast to validate end-to-end, not to produce
  publication-quality results.
- The post-answer confidence elicitation (deviation #1) is a real change to
  the tested protocol, not a bug fix. If a future model doesn't show the
  same blind-elicitation collapse, reverting to the literal spec is worth
  reconsidering.
- Leaked self-ask items are flagged (`"leaked": true`) but not auto-dropped
  from the output; both final runs happened to have zero leaks, so this
  didn't matter here, but it would need handling if a future leak rate
  exceeds the ~2% threshold `experiment-ask-protocol.md` specifies.
