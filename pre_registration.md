# Pre-registration: E-B matched-confidence ask-or-answer

Status: **draft, not frozen.** Written alongside the implementation, not
after it -- review and edit before treating any run's results as
confirmatory. Once you're happy with it, timestamp a frozen copy before
the real run (per experiment-ask-protocol.md: "Freeze them in writing so
a null on prediction 2 is a finding").

## Source documents

- `experiment-matched-confidence.md` -- primary design (Oberman, prepared 2026-07-27).
- `experiment-ask-protocol.md` -- mechanics of the clarifying exchange (2026-08-13).

## Deviation from experiment-ask-protocol.md: post-answer confidence elicitation

The protocol doc specifies confidence elicited by "a stateless call on the
ambiguous question" -- a blind pre-answer guess -- "before any of the four
conditions run." That was the original implementation, and it broke on
Qwen3-8B: with a bare "state your confidence as a number" prompt, the model
reports ~95% on essentially every question regardless of difficulty, verified
even at temperature 1.0 (the per-token probability mass on "95" is
overwhelming enough that sampling rarely escapes it) -- so the 50-60% band
never filled, out of 250 candidates per set, on two separate runs.

Confidence is now elicited *after* an attempted answer: the model answers
the question, then rates its confidence in that specific answer
(`eb/screening.py::CONFIDENCE_SYSTEM`). This is the standard, generally
better-calibrated verbalized-confidence pattern. It preserves the spirit of
"stateless call before any condition observes the answer" for conditions
2-4 and the intended-reading fix, but condition 1 (answer-now) now reuses
the attempted answer generated during screening rather than re-asking
independently (`MatchedSetAItem.answer_now_response` /
`MatchedSetBItem.answer_now_response`) -- both to save a call and because
re-asking risked a second, independently sampled answer disagreeing with
the one confidence was actually rated against.

**If you rerun against a different model and it doesn't show this collapse,**
consider reverting to the doc's literal pre-answer spec (`git log` /
`eb/screening.py`'s prior version) -- this deviation was a response to a
specific model's behavior, not a judgment that the documented protocol is
wrong in general.

## Frozen parameters

- **Confidence band:** 50-60% inclusive, verbalized as an integer percentage
  rating the model's own attempted answer (see deviation above),
  elicited before conditions 2-4 run (`eb/screening.py`).
- **Target set sizes:** 20-30 items per set after matching (default 25 in
  `run_experiment.py --n-per-set`).
- **Grading rule:** normalized exact match (lowercase, strip punctuation and
  articles, alias containment) against the pre-fixed intended reading's alias
  list only. Three outcomes: correct, wrong, hedged. Hedge scores as wrong
  under the strict grading used for the headline numbers (`eb/grading.py`).
- **Intended reading:** fixed per Set A item, uniformly at random, before any
  condition call is made (`eb/screening.py::build_matched_sets`).
- **Model under test:** default `Qwen/Qwen2.5-0.5B-Instruct` for correctness
  testing of this harness (see README for why). **This is not the model the
  paper's E-B numbers should be run against** -- swap `--model` for whatever
  frontier model the actual pre-registered run targets, and record the exact
  model name/version here before that run, per paper2_plan.md's "What George
  needs: a statement of which model and which version, recorded at run time."

## Open items from experiment-ask-protocol.md, resolved here as defaults

The protocol doc lists three things as needing a decision "before a
pre-registration." Defaults chosen for this implementation, documented so
they're easy to override rather than silently baked in:

1. **Clustering rule for the typed diagnostic.** Not a semantic-equivalence
   judge. Each of the n=10 temperature-1 samples is bucketed by alias-list
   containment against the item's own known readings (`eb/diagnostic.py`):
   Set A has two buckets (`reading_a`, `reading_b`), Set B has one
   (`reading`); anything matching neither falls into `other` and is recorded,
   not dropped. This is mechanical and reproducible, at the cost of only
   working because Set A's readings come with ground-truth alias lists
   (true for AmbigQA, not true in general).
2. **Condition 4 sampling.** Runs on the same items as conditions 1-3, not a
   held-out slice. Justification: every condition call is stateless and
   independent (no shared conversation across conditions -- see
   `eb/conditions.py` module docstring), so there is no memory-based
   contamination to worry about, only ordinary sampling variance across
   repeated calls on the same item. Matches the ~700-call budget in the
   design doc.
3. **Hedge-detection rule.** Mechanical: an answer is a hedge iff it contains
   alias-list hits from *both* readings (`eb/grading.py::grade_ambiguous`).
   No judge model.

## Set B source (also left open by the design docs)

`mandarjoshi/trivia_qa`, `unfiltered.nocontext` config, via HuggingFace
`datasets` (`eb/data_setb.py`). Chosen because the screening step -- not the
source dataset -- is what selects for "hard" (band-filtering on elicited
confidence), so any large diverse short-answer trivia pool works; TriviaQA is
the standard public choice with alias lists in the same format Set A's
grading already expects.

## Set A reading-count restriction

AmbigQA items can carry more than two disambiguated readings; this harness
only uses items with *exactly* two (`eb/data_ambigqa.py`), matching the
paper's toy two-reading framing and the protocol's turn structure. Items with
3+ readings are skipped, not merged or truncated.

## Leak audit threshold

Per experiment-ask-protocol.md: "A rate above one or two percent means the
rewrite set is contaminated and needs inspection before the run counts."
`run_experiment.py` computes and prints this rate and warns above 2%; it does
not currently auto-drop leaked items from the output file (they're flagged
via `"leaked": true` in the JSONL record) -- filter them out in `analyze.py`
before treating results as final if the audit fires.

## What this pre-registration does *not* cover

- The real paid run against a frontier model -- this document and the
  current harness were built to validate the harness's mechanics against a
  small local model, not to produce the paper's reported numbers.
- E-A (freeze test) and E-C (variance estimator) -- out of scope for this
  implementation; see the top-level plan.
