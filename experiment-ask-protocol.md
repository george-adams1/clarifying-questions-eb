# The ask protocol for E-B: how the model actually asks a clarifying question

Companion to `experiment-matched-confidence.md`, which designs the matched-confidence
ask-or-answer experiment but specifies the ask itself only as "the model receives one
clarifying exchange." This document fixes the mechanics of that exchange. Nothing here has
been run.

## AmbigQA supplies the readings, so nothing has to be invented

Each ambiguous item in AmbigQA (Min et al., EMNLP 2020) ships with its readings already
written out as *disambiguated rewrites*, each carrying its own answer set:

- ambiguous: "What is the capital of Georgia?"
- reading 1: "What is the capital of the country Georgia?" → Tbilisi
- reading 2: "What is the capital of the U.S. state Georgia?" → Atlanta

The rewrite is the ground-truth clarification. That removes the two hardest pieces of a
clarification experiment: inventing the readings, and judging whether a clarification
succeeded.

Structurally, an item carries an annotation of type `multipleQAs` holding a list of
question-answer pairs, each with a rewritten question and a list of answer aliases. Check
the exact field names against the loaded split before writing the loader.

## Four conditions

The original note's two conditions conflate three different quantities. Splitting them is
what makes the result interpretable.

**1. Answer-now.** The ambiguous question, answered immediately. Baseline.

**2. Oracle-clarify.** The intended reading's rewrite is substituted for the ambiguous
question. No dialogue at all. This is the ceiling: the maximum gain available from
resolving the model channel, with the model's question-writing skill removed from the
measurement. It is the cleanest estimate of the value-of-information quantity Proposition 3
concerns, and it is the primary gain measure.

**3. Self-ask.** The model writes its own clarifying question, a simulated user replies,
and then it answers. The gain here divided by the gain in condition 2 is a competence
measure: how much of the available value the model captures.

**4. Free-choice.** The model is told it may either answer or ask one question, and
decides. This tests Corollary 1 directly rather than testing its premise. Conditions 1
through 3 measure whether asking has value; condition 4 measures whether the model's
decision to ask tracks that value.

The sharpest prediction lives in condition 4: asking rate correlates with stated confidence,
which a collapsed score can support, and not with realized gain, which requires the model
channel.

## Turn structure in condition 3

Turn 1, to the model under test:

```
You will be asked a question. Before answering, you may ask exactly one
clarifying question. Ask it now, as a single sentence, and nothing else.

Question: What is the capital of Georgia?
```

Turn 2, to a separate user-simulator call:

```
You are the person who asked: "What is the capital of Georgia?"
What you meant was: "What is the capital of the country Georgia?"

The assistant has asked you: "Do you mean the country or the U.S. state?"

Reply in one short sentence. Say only which of the two you meant.
```

Turn 3: the model under test receives its own question and that reply, and answers.

## The leak firewall

**The simulator is never given the answer.** If its context contains "Tbilisi," no
instruction reliably prevents a leak, and a leak makes the gain from asking trivially
large. The experiment would then be measuring the leak rather than the model channel.

So the simulator's context holds the ambiguous question, the rewritten question, and the
model's clarifying question, and never the answer field. The rewrite disambiguates without
stating the answer, which is the property AmbigQA's annotation guarantees. This is leak
prevention by construction rather than by instruction, and it is the difference between a
result and an artifact.

Back it with an automated audit regardless: check each simulator reply against the intended
answer and its aliases, drop any item that hits, and report the drop rate. A rate above one
or two percent means the rewrite set is contaminated and needs inspection before the run
counts.

## Set B's control is turn-matched, not clarification-matched

Set B is unambiguous, so there is nothing to disambiguate and condition 2 does not exist
for it. Its ceiling gain is zero by construction, which is the prediction.

Condition 3 still runs on Set B. The model asks whatever it asks, and the simulator replies
from a context carrying no disambiguating information, along the lines of "I mean it exactly
as asked; there is no ambiguity." That absorbs the extra turn, the extra tokens, and the
extra opportunity to reason, so any gain on Set A above Set B is attributable to resolving
the reading rather than to the turn itself.

## Grading, and a third response category

The intended reading is fixed in advance, uniformly at random, before any call is made. Both
conditions are graded against that reading's answer set only, by normalized exact match
over the alias list.

Record three outcomes rather than two: correct, wrong, and **hedged**, where a hedge names
both readings ("Tbilisi or Atlanta, depending"). Under strict grading a hedge scores wrong,
and that has to be pre-registered, since it is the most contestable grading call in the
design.

The hedge rate is worth recording in its own right. A hedge is the barycenter report that
Section 1 of `llm-uncertainty-via-design.md` describes: a model that hedges often is
visibly averaging over the model channel rather than collapsing onto one reading. That rate
is a direct observable of the framing, whichever way the main predictions land.

## Elicitation order

Confidence for the matching step is elicited in its own stateless call on the ambiguous
question, before any of the four conditions run, so the elicitation cannot contaminate the
answer and the matched band is fixed before any condition is observed.

## What still needs deciding before a pre-registration

- The clustering rule for the per-question typed diagnostic (prediction 3 of the original
  note). Normalized exact match over short entity answers covers most AmbigQA items; the
  residual needs either an equivalence judge or a documented drop rule.
- Whether condition 4 is run on the same items as conditions 1 through 3, which risks order
  effects across conditions, or on a held-out slice, which costs sample size.
- The hedge-detection rule, which has to be mechanical rather than judged.
