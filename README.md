# E-B: matched-confidence ask-or-answer

Implements the experiment designed in `experiment-matched-confidence.md` and
`experiment-ask-protocol.md` (companion planning docs for the paper
*Clarifying Questions as Experimental Design*): can a single collapsed
confidence number predict which questions benefit from a clarifying exchange?
Two matched question sets (ambiguous vs. difficult-but-unambiguous, both
screened to the same 50-60% verbalized-confidence band), four conditions per
item, a diagnostic that samples repeatedly and checks the spread across
readings. See `pre_registration.md` for the frozen parameters and the
defaults chosen for the three items the protocol doc left open.

**Status:** the harness is complete and verified offline (`pytest`, no API
cost) and against a small real model. It has *not* been run at the scale or
against the model the paper's actual E-B numbers should come from --
`pre_registration.md` explains why the default model here is a stand-in.

## Layout

```
eb/
  model_client.py   pluggable ModelClient: LocalHFClient (real), MockClient (offline)
  data_ambigqa.py    Set A loader (AmbigQA, exactly-two-reading items)
  data_setb.py       Set B loader (TriviaQA unfiltered.nocontext)
  screening.py       confidence-band matching, intended-reading fixing
  conditions.py      the 4 conditions + the leak firewall
  diagnostic.py      n=10 @ temp=1 sampling, reading-bucket clustering
  grading.py         normalized exact match, mechanical hedge rule
  run_experiment.py  orchestration -> JSONL
  analyze.py         the 3 predictions
tests/
  fixtures.py        tiny offline Set A/B pools + two mock personas (flat/typed)
  test_pipeline.py   13 tests covering screening, leak audit, grading, diagnostic
pre_registration.md  frozen parameters, defaults for the 3 open protocol items
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the offline tests (no API/model cost)

```
.venv/bin/python -m pytest tests/ -v
```

## Run against a real small model (correctness check, not the paper's numbers)

```
.venv/bin/python -m eb.run_experiment \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --n-per-set 25 --pool-limit 300 \
  --out results.jsonl
.venv/bin/python -m eb.analyze results.jsonl
```

`--pool-limit` caps the candidate pool for a quick smoke run; drop it for a
full screening pass. `--mock typed` / `--mock flat` swap in the offline mock
personas from `tests/fixtures.py` instead of a real model (useful for CLI
smoke tests, but note the mock personas only recognize the two fixture
questions -- everything else falls through to a default "I don't know").

## Cost note

`Qwen/Qwen2.5-0.5B-Instruct` is a ~1GB model chosen to make this harness
cheap to verify end-to-end (CPU-runnable, no API key, no per-token cost). The
design docs' actual ~700-call, "a day of work end to end" cost estimate
assumes a frontier model via a paid API -- update `model_client.py` with an
API-based client (Anthropic/OpenAI) behind the same `ModelClient` interface
before running the real experiment, and record the exact model/version in
`pre_registration.md` per paper2_plan.md's requirement.

## Known limitations

- AmbigQA items with 3+ disambiguated readings are skipped (see
  `pre_registration.md`).
- The typed diagnostic's clustering rule only works because Set A's readings
  ship with ground-truth alias lists; it is not a general semantic-clustering
  solution.
- Leaked self-ask items (per the audit) are flagged in the output, not
  auto-dropped -- filter on `"leaked": true` before treating results as final.
