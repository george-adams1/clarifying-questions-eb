# Matched-Confidence Experiment: Can a Collapsed Confidence Score Decide Ask-or-Answer?

Design note for the draft *Clarifying Questions as Experimental Design* (Oberman, July 2026). Prepared July 27, 2026. This is a stated design; no run has been performed.

## Claim under test

The paper models a system's state on an ambiguous question as a typed belief: a model channel $w$ over readings of the question, and a parameter channel over answers within each reading. Reporting one confidence number collapses this belief to a single predictive distribution. Corollary 1 of the paper states that no function of the collapsed belief can decide the ask-or-answer choice: two belief states with the same collapsed report can assign very different values to a clarifying question.

The experiment tests the observable consequence. A model's stated confidence should have no power to predict which questions benefit from a clarifying exchange, while a cheap diagnostic that looks at the spread across readings should predict it well.

## Design

Build two question sets matched on the model's stated confidence (target band roughly 50-60%):

- **Set A (ambiguous).** Questions with two readings, each reading having a known answer ("What is the capital of Georgia?"). AmbigQA (Min et al., EMNLP 2020) supplies these ready-made, with readings and per-reading answers annotated.
- **Set B (difficult, unambiguous).** One clear reading, but the model half-knows the answer: obscure trivia screened to land in the same confidence band.

Matching procedure: elicit verbalized confidence over a candidate pool and keep only items in the band, so the two sets are indistinguishable to the collapsed score by construction. Expect to screen a pool 3-4 times larger than the final sets.

Conditions per question, specified in full in `experiment-ask-protocol.md`:

1. **Answer now.** The model answers immediately.
2. **Oracle-clarify.** The intended reading's AmbigQA rewrite is substituted for the ambiguous question, with no dialogue. This is the ceiling gain and the primary gain measure, since it removes the model's question-writing skill from the quantity being estimated. Set B has no reading to substitute, so its ceiling is zero by construction.
3. **Self-ask.** The model writes its own clarifying question and a simulated user replies. The simulator is never given the answer, only the rewritten question, so a leak is impossible by construction rather than by instruction. For Set B the reply adds no disambiguating content; it controls for the extra turn.
4. **Free-choice.** The model decides whether to ask or answer. This is the condition that tests Corollary 1 directly rather than testing its premise: the prediction is that asking rate tracks stated confidence and not realized gain.

Grading for Set A: fix the intended reading per item in advance, uniformly at random between the two readings, and grade both conditions against the intended reading's answer. Under strict grading (only the intended answer counts), answer-now succeeds only by committing to the correct reading, which is the decision problem the paper analyzes. Fix one grading rule before running and keep it.

Per-question typed diagnostic: sample $n$ answers (e.g. $n = 10$ at temperature 1), cluster by semantic equivalence, and record the spread over reading-level clusters. The simplest statistics are the entropy over clusters or the size of the second-largest cluster.

## Predictions

1. The gain from asking (ask-first accuracy minus answer-now accuracy) is large on Set A and near zero on Set B. With two equiprobable readings and strict grading, answer-now on Set A should sit near 50% of the model's within-reading accuracy, and ask-first near the within-reading accuracy itself.
2. Pre-ask confidence does not predict the gain: within the matched band, the correlation across the pooled sets is near zero. This is Corollary 1 in observable form.
3. The cluster diagnostic does predict the gain: the correlation is clearly positive, and ranking items by the diagnostic separates Set A from Set B.

Prediction 3 is part of the design, since without the positive contrast a null correlation between confidence and gain could be read as noise.

The scale of the predicted gap has a closed-form anchor in the paper: on the two-coin example, the same one-flip observation is worth $0.1$ under an identification utility and $0.0004$ under a one-step Brier forecasting utility (Proposition 5, verified in `code/voi_check.py`). The value of asking concentrates in the model channel, which is present in Set A and absent in Set B.

## What would count against the paper

If stated confidence does separate the sets, for example if the model verbalizes lower or qualitatively different confidence on ambiguous items, then the verbalized channel carries more than the collapsed predictive. That would be a substantive finding against the modeling assumption of Section 4, which treats verbalized confidence as a function of the collapsed predictive and flags that treatment as an idealization. Either outcome is informative.

## Cost

- 40-60 questions after matching (20-30 per set), plus the screening pool.
- One model; on the order of $(2 + n) \times 60 \approx 700$ short API calls plus screening.
- No training; no human annotation beyond assembling Set B and spot-checking AmbigQA readings.
- Analysis: two correlations and one bar chart. A day of work end to end.

## Secondary designs (Section 9 of the paper, heavier)

1. **Question-selection comparison.** Select clarifying questions by the discrimination objective ($J$ or value of information under a stated utility) against selection by answer-entropy reduction, on ambiguity benchmarks; the predicted signature is that discriminating questions move cross-category forecasts with the sign the mixing law's off-diagonal dictates.
2. **Freeze test.** A system restricted to its collapsed confidence should show no update on clarifying replies that move a typed baseline; the martingale tests of Falck et al. (2024) give the measurement template.
3. **Cost threshold.** With a stated utility, the value-of-information rule predicts the asking rate as a function of the asking cost, falsifiable against human judgments of when a question was worth its turn.

## Reference

Draft: `LLM UQ latex/clarifying_questions_design.pdf`, in particular Corollary 1 (no collapsed score decides ask-or-answer), Proposition 5 (closed forms), and Section 9 (Toward an Experiment).
