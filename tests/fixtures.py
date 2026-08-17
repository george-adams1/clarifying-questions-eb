"""Offline fixtures: tiny hand-built Set A / Set B pools and two mock
personas (flat = collapses to one guess regardless of context, typed =
actually resolves ambiguity through the clarifying exchange), mirroring
the two-mock pattern described for the E-A harness (a flat reasoner
scoring slope 0, a typed reasoner scoring slope 1).
"""
from __future__ import annotations

import itertools

from eb.data_ambigqa import Reading, SetAItem
from eb.data_setb import SetBItem
from eb.model_client import MockClient, MockPersona

# ---- Fixture pools ------------------------------------------------------

GEORGIA = SetAItem(
    id="qA1",
    ambiguous_question="What is the capital of Georgia?",
    reading_a=Reading(question="What is the capital of the country Georgia?", answers=["Tbilisi"]),
    reading_b=Reading(question="What is the capital of the U.S. state Georgia?", answers=["Atlanta"]),
)
WASHINGTON = SetAItem(  # confidence deliberately out of band -- tests the screening filter
    id="qA2",
    ambiguous_question="What is the capital of Washington?",
    reading_a=Reading(question="What is the capital of the country of Washington (there is none)?", answers=["N/A"]),
    reading_b=Reading(question="What is the capital of the U.S. state of Washington?", answers=["Olympia"]),
)

CHIPMUNKS = SetBItem(id="qB1", question="Who was the man behind The Chipmunks?", answers=["David Seville"])
BOILING = SetBItem(  # confidence deliberately out of band
    id="qB2", question="What is the boiling point of water at sea level?", answers=["100 degrees Celsius", "212 degrees Fahrenheit"]
)

FIXTURE_SETA_POOL = [GEORGIA, WASHINGTON]
FIXTURE_SETB_POOL = [CHIPMUNKS, BOILING]

_CONFIDENCE = {
    "What is the capital of Georgia?": "55",
    "What is the capital of Washington?": "90",
    "Who was the man behind The Chipmunks?": "55",
    "What is the boiling point of water at sea level?": "95",
}


# ---- Persona construction -------------------------------------------------

def _confidence_handler(system, user, temperature):
    for question, value in _CONFIDENCE.items():
        if question in user:
            return value
    return "50"


def _answer_system_handler(style: str, diag_cycle):
    def handler(system, user, temperature):
        if "You asked:" in user and "Reply:" in user:
            if "Chipmunks" in user:
                return "David Seville"
            if style == "flat":
                # The frozen reasoner (Theorem 1): ignores the reply, which
                # is exactly the forecast-moving observation it cannot use,
                # and keeps its prior guess regardless of what it says.
                return "Atlanta"
            reply = user.split("Reply:", 1)[1]
            if "country" in reply:
                return "Tbilisi"
            return "Atlanta"
        if user == "Question: What is the capital of the country Georgia?":
            return "Tbilisi"
        if user == "Question: What is the capital of the U.S. state Georgia?":
            return "Atlanta"
        if user == "Question: What is the capital of Georgia?":
            return "Tbilisi or Atlanta, depending on which Georgia you mean." if style == "typed" else "Atlanta"
        if user == "Question: Who was the man behind The Chipmunks?":
            return "David Seville"
        if user == "What is the capital of Georgia?":  # diagnostic (no "Question:" prefix)
            return next(diag_cycle) if style == "typed" else "Atlanta"
        if user == "Who was the man behind The Chipmunks?":  # diagnostic
            return "David Seville"
        return "I don't know."

    return handler


def _ask_handler(style: str):
    def handler(system, user, temperature):
        if "Georgia" in user:
            return "Do you mean the country or the U.S. state?" if style == "typed" else "Can you clarify?"
        return "Can you clarify what you're asking?"

    return handler


def _free_choice_handler(style: str):
    def handler(system, user, temperature):
        if "Georgia" in user:
            return "ASK: Do you mean the country or the U.S. state?" if style == "typed" else "Atlanta"
        return "David Seville"  # Set B: answer directly, nothing to clarify

    return handler


def _simulator_a_handler(style: str):
    def handler(system, user, temperature):
        if "capital of the country Georgia" in user:
            return "The country."
        return "The U.S. state."

    return handler


def _simulator_b_handler(style: str):
    def handler(system, user, temperature):
        return "I mean it exactly as asked; there is no ambiguity."

    return handler


def _build_persona(style: str) -> MockPersona:
    diag_cycle = itertools.cycle(["Tbilisi", "Atlanta"])
    rules = [
        (r"single integer percentage", _confidence_handler),
        (r"ASK: <your clarifying question>", _free_choice_handler(style)),
        (r"Ask it now, as a single sentence", _ask_handler(style)),
        (r"about what you meant", _simulator_a_handler(style)),
        (r"You are a helpful user answering a clarifying question\.", _simulator_b_handler(style)),
        (r"Answer the question as concisely", _answer_system_handler(style, diag_cycle)),
    ]
    return MockPersona(name=style, rules=rules)


def make_mock_client(style: str) -> MockClient:
    assert style in ("typed", "flat")
    return MockClient(_build_persona(style))
