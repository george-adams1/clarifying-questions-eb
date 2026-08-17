"""Normalized exact match grading, per experiment-ask-protocol.md:

"Both conditions are graded against that reading's answer set only, by
normalized exact match over the alias list ... Record three outcomes
rather than two: correct, wrong, and hedged, where a hedge names both
readings ... Under strict grading a hedge scores wrong."

Open item resolved here (documented default, see pre_registration.md):
hedge detection is mechanical -- an answer is a hedge iff it contains an
alias from *both* readings' alias lists. No judge model.
"""
from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def contains_alias(response: str, aliases: list[str]) -> bool:
    norm_response = normalize_text(response)
    for alias in aliases:
        norm_alias = normalize_text(alias)
        if norm_alias and norm_alias in norm_response:
            return True
    return False


def grade_single(response: str, correct_aliases: list[str]) -> str:
    """Set B / oracle-clarify grading: only "correct" or "wrong" apply."""
    return "correct" if contains_alias(response, correct_aliases) else "wrong"


def grade_ambiguous(response: str, intended_aliases: list[str], other_aliases: list[str]) -> str:
    """Set A grading against the pre-fixed intended reading.

    Order matters: check hedge (both readings named) before correct, so an
    answer that happens to also mention the other reading's entity is
    still scored as a hedge, per the protocol's strict-grading rule.
    """
    hit_intended = contains_alias(response, intended_aliases)
    hit_other = contains_alias(response, other_aliases)
    if hit_intended and hit_other:
        return "hedged"
    if hit_intended:
        return "correct"
    return "wrong"
