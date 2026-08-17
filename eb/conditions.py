"""The four conditions, per experiment-ask-protocol.md.

Each condition is a pure function of (ModelClient, question data) -- every
call is an independent, stateless completion, so there is no shared
conversation state across conditions or across items (resolves the
protocol's open item #2: condition 4 can safely run on the same items as
1-3 because nothing carries over between calls).
"""
from __future__ import annotations

from dataclasses import dataclass

from .grading import contains_alias

ANSWER_SYSTEM = "Answer the question as concisely as possible: a short phrase, no explanation."

ASK_SYSTEM = (
    "You will be asked a question. Before answering, you may ask exactly one "
    "clarifying question. Ask it now, as a single sentence, and nothing else."
)

FREE_CHOICE_SYSTEM = (
    "You will be asked a question. You may either answer it directly, or ask "
    "exactly one clarifying question first if the question seems ambiguous. "
    "If you want to ask a question, respond with exactly:\n"
    "ASK: <your clarifying question>\n"
    "Otherwise, just answer the question directly and concisely, with no preamble."
)


# ---- Condition 1: answer-now -------------------------------------------------

def answer_now(client, question: str, temperature: float = 0.0) -> str:
    return client.complete(ANSWER_SYSTEM, f"Question: {question}", temperature=temperature)


# ---- Condition 2: oracle-clarify (Set A only) --------------------------------

def oracle_clarify(client, rewritten_question: str, temperature: float = 0.0) -> str:
    return client.complete(ANSWER_SYSTEM, f"Question: {rewritten_question}", temperature=temperature)


# ---- Condition 3: self-ask, with the leak firewall ---------------------------

@dataclass
class SelfAskResult:
    clarifying_question: str
    simulator_reply: str
    leaked: bool
    final_answer: str


def _simulate_reply_set_a(client, ambiguous_question: str, rewritten_question: str, clarifying_question: str, temperature: float) -> str:
    # Leak firewall: this prompt never contains an answer field, only the
    # rewritten (disambiguating) question -- see module docstring in
    # data_ambigqa.py and experiment-ask-protocol.md's "leak firewall" section.
    system = "You are a helpful user answering a clarifying question about what you meant."
    user = (
        f'You are the person who asked: "{ambiguous_question}"\n'
        f'What you meant was: "{rewritten_question}"\n\n'
        f'The assistant has asked you: "{clarifying_question}"\n\n'
        "Reply in one short sentence. Say only which of the two you meant."
    )
    return client.complete(system, user, temperature=temperature)


def _simulate_reply_set_b(client, question: str, clarifying_question: str, temperature: float) -> str:
    system = "You are a helpful user answering a clarifying question."
    user = (
        f'You are the person who asked: "{question}"\n\n'
        f'The assistant has asked you: "{clarifying_question}"\n\n'
        "Reply in one short sentence. There is no ambiguity in what you meant -- "
        "the question means exactly what it says. If the assistant's question "
        "can be answered from the original question alone, answer it; otherwise "
        "say there is nothing more to clarify."
    )
    return client.complete(system, user, temperature=temperature)


def self_ask_set_a(client, ambiguous_question: str, rewritten_question: str, intended_aliases: list[str], temperature: float = 0.0) -> SelfAskResult:
    clarifying_question = client.complete(ASK_SYSTEM, f"Question: {ambiguous_question}", temperature=temperature)
    simulator_reply = _simulate_reply_set_a(client, ambiguous_question, rewritten_question, clarifying_question, temperature)
    leaked = contains_alias(simulator_reply, intended_aliases)
    final_answer = client.complete(
        ANSWER_SYSTEM,
        f'Question: {ambiguous_question}\nYou asked: "{clarifying_question}"\nReply: "{simulator_reply}"\nNow answer the original question.',
        temperature=temperature,
    )
    return SelfAskResult(clarifying_question, simulator_reply, leaked, final_answer)


def self_ask_set_b(client, question: str, temperature: float = 0.0) -> SelfAskResult:
    clarifying_question = client.complete(ASK_SYSTEM, f"Question: {question}", temperature=temperature)
    simulator_reply = _simulate_reply_set_b(client, question, clarifying_question, temperature)
    final_answer = client.complete(
        ANSWER_SYSTEM,
        f'Question: {question}\nYou asked: "{clarifying_question}"\nReply: "{simulator_reply}"\nNow answer the original question.',
        temperature=temperature,
    )
    return SelfAskResult(clarifying_question, simulator_reply, leaked=False, final_answer=final_answer)


# ---- Condition 4: free-choice -------------------------------------------------

@dataclass
class FreeChoiceResult:
    asked: bool
    clarifying_question: str | None
    simulator_reply: str | None
    leaked: bool | None
    final_answer: str


def _parse_ask(response: str) -> str | None:
    stripped = response.strip()
    if stripped.upper().startswith("ASK:"):
        return stripped[len("ASK:"):].strip()
    return None


def free_choice_set_a(client, ambiguous_question: str, rewritten_question: str, intended_aliases: list[str], temperature: float = 0.0) -> FreeChoiceResult:
    response = client.complete(FREE_CHOICE_SYSTEM, f"Question: {ambiguous_question}", temperature=temperature)
    clarifying_question = _parse_ask(response)
    if clarifying_question is None:
        return FreeChoiceResult(asked=False, clarifying_question=None, simulator_reply=None, leaked=None, final_answer=response)
    simulator_reply = _simulate_reply_set_a(client, ambiguous_question, rewritten_question, clarifying_question, temperature)
    leaked = contains_alias(simulator_reply, intended_aliases)
    final_answer = client.complete(
        ANSWER_SYSTEM,
        f'Question: {ambiguous_question}\nYou asked: "{clarifying_question}"\nReply: "{simulator_reply}"\nNow answer the original question.',
        temperature=temperature,
    )
    return FreeChoiceResult(asked=True, clarifying_question=clarifying_question, simulator_reply=simulator_reply, leaked=leaked, final_answer=final_answer)


def free_choice_set_b(client, question: str, temperature: float = 0.0) -> FreeChoiceResult:
    response = client.complete(FREE_CHOICE_SYSTEM, f"Question: {question}", temperature=temperature)
    clarifying_question = _parse_ask(response)
    if clarifying_question is None:
        return FreeChoiceResult(asked=False, clarifying_question=None, simulator_reply=None, leaked=None, final_answer=response)
    simulator_reply = _simulate_reply_set_b(client, question, clarifying_question, temperature)
    final_answer = client.complete(
        ANSWER_SYSTEM,
        f'Question: {question}\nYou asked: "{clarifying_question}"\nReply: "{simulator_reply}"\nNow answer the original question.',
        temperature=temperature,
    )
    return FreeChoiceResult(asked=True, clarifying_question=clarifying_question, simulator_reply=simulator_reply, leaked=False, final_answer=final_answer)
