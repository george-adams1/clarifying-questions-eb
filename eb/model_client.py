"""Pluggable model client interface for the E-B harness.

Every condition in conditions.py talks to a model only through the
ModelClient.complete() interface, so the same pipeline runs unchanged
against a MockClient (offline, free, deterministic) or a real model
(LocalHFClient, loading small instruct models via transformers).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ModelClient(ABC):
    """A stateless chat completion call: system + user -> assistant text.

    Each call is independent (no shared conversation state), matching the
    protocol doc's requirement that conditions and the elicitation step
    cannot contaminate each other.
    """

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class LocalHFClient(ModelClient):
    """Runs a small instruct model locally via transformers.

    Designed for cheap correctness testing of the harness against a real
    model, not for the full paid-scale run the design docs describe against
    a frontier model.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-8B", max_new_tokens: int = 200, device: str | None = None, device_map: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device_map:
            # Shard the model across multiple GPUs (e.g. device_map="auto"
            # for a model too large for one card) instead of placing it on
            # a single device -- accelerate handles the placement, and we
            # never call .to() ourselves, since a sharded model has no
            # single device to move to.
            self.device = None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map=device_map
            )
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
        self.model.eval()

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            # Qwen3 and similar hybrid-reasoning models default to a long
            # chain-of-thought "thinking" mode; our prompts want a short
            # direct phrase, so turn it off where the chat template supports
            # the argument. Models without it just ignore the kwarg... except
            # apply_chat_template raises TypeError on an unknown kwarg, so we
            # catch that and fall back to the plain call.
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # self.model.device works for both single-device placement and a
        # device_map-sharded model (returns the entry-layer's device;
        # accelerate's hooks move activations across shards internally).
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@dataclass
class MockPersona:
    """A canned response function keyed by a regex match on the user prompt.

    Lets tests assert on pipeline logic (screening filter, leak firewall,
    grading, diagnostic clustering) without any real model or network call,
    mirroring the two-mock pattern (`flat` vs `typed` reasoner) described
    for the E-A harness in paper2_plan.md.
    """

    name: str
    rules: list = field(default_factory=list)  # list of (pattern, response_fn)
    default: str = "I don't know."

    def respond(self, system: str, user: str, temperature: float) -> str:
        combined = f"{system}␟{user}"
        for pattern, response_fn in self.rules:
            if re.search(pattern, combined, re.DOTALL):
                return response_fn(system, user, temperature)
        return self.default


class MockClient(ModelClient):
    """Wraps a MockPersona behind the ModelClient interface."""

    def __init__(self, persona: MockPersona):
        self.persona = persona
        self.call_log: list[tuple[str, str, float]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.call_log.append((system, user, temperature))
        return self.persona.respond(system, user, temperature)
