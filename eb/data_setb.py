"""Set B (difficult, unambiguous trivia) candidate loader.

experiment-matched-confidence.md leaves the Set B source unspecified
("obscure trivia screened to land in the same confidence band") --
paper2_plan.md lists "a source for the Set B trivia pool" as something
George still needs. TriviaQA (unfiltered.nocontext config, via
`mandarjoshi/trivia_qa` on HuggingFace) is used here as a documented
default: a large public pool of short-answer trivia with alias lists in
the same format Set A's grading already expects, no context passages
needed since the screening step (not the source dataset) is what selects
for "hard." Swap this loader for a different pool without touching
anything downstream.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SetBItem:
    id: str
    question: str
    answers: list[str]  # alias list


def load_setb_candidates(split: str = "validation") -> list[SetBItem]:
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split=split)
    items: list[SetBItem] = []
    for ex in ds:
        aliases = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        if not ex["question"] or not aliases:
            continue
        items.append(SetBItem(id=ex["question_id"], question=ex["question"], answers=list(aliases)))
    return items
