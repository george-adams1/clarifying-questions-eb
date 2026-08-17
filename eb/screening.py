"""Confidence-band screening and matched-set construction.

experiment-matched-confidence.md: "Matching procedure: elicit verbalized
confidence over a candidate pool and keep only items in the band, so the
two sets are indistinguishable to the collapsed score by construction."

experiment-ask-protocol.md, "Elicitation order": "Confidence for the
matching step is elicited in its own stateless call on the ambiguous
question, before any of the four conditions run, so the elicitation
cannot contaminate the answer and the matched band is fixed before any
condition is observed." -- and: "The intended reading is fixed in advance,
uniformly at random, before any call is made."

DEVIATION FROM THE PROTOCOL DOC, recorded here and in pre_registration.md:
a blind pre-answer "state a number" elicitation (the doc's literal spec)
turned out to be degenerate on Qwen3-8B -- it reports ~95% confidence on
essentially every question regardless of difficulty, verified even at
temperature 1.0 (the per-token probability mass on "95" is overwhelming),
so the 50-60% band never fills. Confidence is instead elicited *after* an
attempted answer, asking the model to rate that specific answer. This is
the standard better-calibrated verbalized-confidence pattern, and it still
happens in its own call, still before any of conditions 2-4 run, and still
before the intended reading is fixed -- so the "cannot contaminate the
answer" property is preserved for everything except condition 1 itself,
which now simply reuses the attempted answer instead of re-asking (see
`answer_now_response` below), rather than risking a second, independently
sampled answer disagreeing with the one confidence was rated against.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .conditions import answer_now

CONFIDENCE_SYSTEM = (
    "You will be shown a question and a proposed answer to it. State only your "
    "confidence that the proposed answer is correct, as a single integer "
    "percentage from 0 to 100. Respond with only the number, nothing else."
)


def _parse_confidence(response: str) -> float | None:
    match = re.search(r"\d{1,3}", response)
    if not match:
        return None
    value = float(match.group())
    return value if 0 <= value <= 100 else None


def elicit_confidence_for_answer(client, question: str, answer: str) -> float | None:
    user = f"Question: {question}\nProposed answer: {answer}"
    response = client.complete(CONFIDENCE_SYSTEM, user, temperature=0.0)
    return _parse_confidence(response)


@dataclass
class MatchedSetAItem:
    id: str
    ambiguous_question: str
    reading_a: object  # data_ambigqa.Reading
    reading_b: object
    intended: str  # "a" or "b"
    confidence: float
    answer_now_response: str


@dataclass
class MatchedSetBItem:
    id: str
    question: str
    answers: list[str]
    confidence: float
    answer_now_response: str


def screen_to_band(client, candidates, question_fn, band: tuple[float, float] = (50.0, 60.0), limit: int | None = None):
    """Returns [(candidate, answer, confidence), ...] for candidates whose
    elicited confidence -- in the model's own attempted answer -- falls in
    `band` (inclusive), in candidate order, up to `limit`.
    """
    kept = []
    for c in candidates:
        question = question_fn(c)
        answer = answer_now(client, question)
        conf = elicit_confidence_for_answer(client, question, answer)
        if conf is None:
            continue
        if band[0] <= conf <= band[1]:
            kept.append((c, answer, conf))
            if limit is not None and len(kept) >= limit:
                break
    return kept


def build_matched_sets(
    client,
    setA_pool: list,
    setB_pool: list,
    target_per_set: int = 25,
    band: tuple[float, float] = (50.0, 60.0),
    rng: random.Random | None = None,
    setA_target: int | None = None,
    setB_target: int | None = None,
) -> tuple[list[MatchedSetAItem], list[MatchedSetBItem]]:
    """setA_target/setB_target override target_per_set for one set only when
    given; pass a very large number (e.g. len(pool)) for "no early stop,
    scan the whole pool" -- screen_to_band's `limit=None` already means
    unlimited, but that's reserved here for "use target_per_set" so a
    caller can't accidentally get unlimited on both sets from one typo.
    """
    rng = rng or random.Random()

    a_limit = setA_target if setA_target is not None else target_per_set
    b_limit = setB_target if setB_target is not None else target_per_set
    a_kept = screen_to_band(client, setA_pool, lambda c: c.ambiguous_question, band, limit=a_limit)
    b_kept = screen_to_band(client, setB_pool, lambda c: c.question, band, limit=b_limit)

    setA = []
    for cand, answer, conf in a_kept:
        intended = rng.choice(["a", "b"])
        setA.append(
            MatchedSetAItem(
                id=cand.id,
                ambiguous_question=cand.ambiguous_question,
                reading_a=cand.reading_a,
                reading_b=cand.reading_b,
                intended=intended,
                confidence=conf,
                answer_now_response=answer,
            )
        )

    setB = [
        MatchedSetBItem(id=cand.id, question=cand.question, answers=cand.answers, confidence=conf, answer_now_response=answer)
        for cand, answer, conf in b_kept
    ]

    return setA, setB
