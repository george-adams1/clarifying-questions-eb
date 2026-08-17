"""Set A (ambiguous) candidate loader.

experiment-ask-protocol.md: "Structurally, an item carries an annotation of
type `multipleQAs` holding a list of question-answer pairs, each with a
rewritten question and a list of answer aliases. Check the exact field
names against the loaded split before writing the loader." Verified against
the real `sewon/ambig_qa` (config "light") validation split:

    annotations = {
        "type": ["multipleQAs", ...],       # one entry per annotator
        "answer": [[...], ...],             # only populated for singleAnswer
        "qaPairs": [
            {"question": [str, str, ...], "answer": [[alias, ...], ...]},
            ...
        ],
    }

A real item can carry more than two disambiguated readings (some AmbigQA
questions have 3-4). The paper's toy framing and the protocol's turn
structure assume exactly two readings ("the country or the U.S. state?"),
so we filter to qaPairs groups with exactly two sub-questions -- documented
in pre_registration.md as a scope-narrowing choice, not silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Reading:
    question: str
    answers: list[str]  # alias list, ground truth for this reading


@dataclass
class SetAItem:
    id: str
    ambiguous_question: str
    reading_a: Reading
    reading_b: Reading


def load_ambigqa_candidates(split: str = "validation", config: str = "light") -> list[SetAItem]:
    from datasets import load_dataset

    ds = load_dataset("sewon/ambig_qa", config, split=split)
    items: list[SetAItem] = []
    for ex in ds:
        ann = ex["annotations"]
        for i, t in enumerate(ann["type"]):
            if t != "multipleQAs":
                continue
            qa = ann["qaPairs"][i]
            questions = qa["question"]
            answers = qa["answer"]
            if len(questions) != 2:
                continue
            if not answers[0] or not answers[1]:
                # an empty alias list means nothing to grade against
                continue
            items.append(
                SetAItem(
                    id=ex["id"],
                    ambiguous_question=ex["question"],
                    reading_a=Reading(question=questions[0], answers=answers[0]),
                    reading_b=Reading(question=questions[1], answers=answers[1]),
                )
            )
            break  # one two-reading group per item is enough
    return items
