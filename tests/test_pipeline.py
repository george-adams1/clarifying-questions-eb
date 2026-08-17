import json
import random

import pytest

from eb import conditions as C
from eb.diagnostic import run_diagnostic
from eb.grading import grade_ambiguous, grade_single
from eb.model_client import MockClient, MockPersona
from eb.run_experiment import run_set_a_item, run_set_b_item
from eb.screening import build_matched_sets

from .fixtures import BOILING, CHIPMUNKS, FIXTURE_SETA_POOL, FIXTURE_SETB_POOL, GEORGIA, WASHINGTON, make_mock_client


# ---- grading --------------------------------------------------------------

def test_grade_ambiguous_correct_wrong_hedge():
    intended = ["Tbilisi"]
    other = ["Atlanta"]
    assert grade_ambiguous("Tbilisi", intended, other) == "correct"
    assert grade_ambiguous("It's Atlanta.", intended, other) == "wrong"
    assert grade_ambiguous("Tbilisi or Atlanta, depending.", intended, other) == "hedged"


def test_grade_single():
    assert grade_single("David Seville is the answer.", ["David Seville"]) == "correct"
    assert grade_single("Alvin", ["David Seville"]) == "wrong"


# ---- screening: confidence-band filter and intended-reading fixing --------

def test_screening_filters_to_band_and_fixes_intended_reading():
    client = make_mock_client("typed")
    rng = random.Random(0)
    setA, setB = build_matched_sets(client, FIXTURE_SETA_POOL, FIXTURE_SETB_POOL, target_per_set=10, rng=rng)

    assert [i.id for i in setA] == [GEORGIA.id]  # WASHINGTON's confidence (90) is out of band
    assert [i.id for i in setB] == [CHIPMUNKS.id]  # BOILING's confidence (95) is out of band
    assert setA[0].confidence == 55.0
    assert setA[0].intended in ("a", "b")


# ---- leak firewall ----------------------------------------------------------

def test_self_ask_does_not_leak_answer_by_construction():
    client = make_mock_client("typed")
    result = C.self_ask_set_a(
        client,
        GEORGIA.ambiguous_question,
        GEORGIA.reading_a.question,
        GEORGIA.reading_a.answers,
    )
    assert result.leaked is False
    assert "Tbilisi" not in result.simulator_reply


def test_leak_audit_catches_an_actual_leak():
    # A deliberately broken simulator that leaks the answer, to prove the
    # audit mechanism (grading.contains_alias on the simulator reply) would
    # actually catch a leak rather than trivially always passing.
    leaky_persona = MockPersona(
        name="leaky",
        rules=[
            (r"single integer percentage", lambda s, u, t: "55"),
            (r"Ask it now", lambda s, u, t: "Do you mean the country or the state?"),
            (r"about what you meant", lambda s, u, t: "You meant the country, so the answer is Tbilisi."),
        ],
        default="Tbilisi",
    )
    client = MockClient(leaky_persona)
    result = C.self_ask_set_a(
        client,
        GEORGIA.ambiguous_question,
        GEORGIA.reading_a.question,
        GEORGIA.reading_a.answers,
    )
    assert result.leaked is True


# ---- diagnostic clustering --------------------------------------------------

def test_diagnostic_shows_high_spread_for_typed_on_ambiguous_question():
    client = make_mock_client("typed")
    result = run_diagnostic(
        client, GEORGIA.ambiguous_question, {"reading_a": GEORGIA.reading_a.answers, "reading_b": GEORGIA.reading_b.answers}, n=10, temperature=1.0
    )
    assert result.cluster_counts.get("reading_a", 0) > 0
    assert result.cluster_counts.get("reading_b", 0) > 0
    assert result.entropy > 0.9  # roughly balanced 5/5 split -> entropy near 1 bit


def test_diagnostic_shows_low_spread_for_flat_on_ambiguous_question():
    client = make_mock_client("flat")
    result = run_diagnostic(
        client, GEORGIA.ambiguous_question, {"reading_a": GEORGIA.reading_a.answers, "reading_b": GEORGIA.reading_b.answers}, n=10, temperature=1.0
    )
    assert result.entropy == 0.0
    assert result.second_largest_frac == 0.0


def test_diagnostic_shows_low_spread_on_unambiguous_setb_question():
    client = make_mock_client("typed")
    result = run_diagnostic(client, CHIPMUNKS.question, {"reading": CHIPMUNKS.answers}, n=10, temperature=1.0)
    assert result.entropy == 0.0


# ---- hedge behavior on answer-now -------------------------------------------

def test_typed_persona_hedges_on_immediate_answer():
    client = make_mock_client("typed")
    response = C.answer_now(client, GEORGIA.ambiguous_question)
    graded = grade_ambiguous(response, GEORGIA.reading_a.answers, GEORGIA.reading_b.answers)
    assert graded == "hedged"


# ---- end-to-end: run_set_a_item / run_set_b_item produce valid records -----

def test_run_set_a_item_end_to_end_typed():
    client = make_mock_client("typed")
    rng = random.Random(0)
    setA, _ = build_matched_sets(client, [GEORGIA], [], target_per_set=1, rng=rng)
    record = run_set_a_item(client, setA[0], diagnostic_n=10, diagnostic_temp=1.0)

    assert record["set"] == "A"
    assert record["answer_now"]["grade"] == "hedged"
    assert record["oracle_clarify"]["grade"] == "correct"  # given the reading directly, typed always gets it right
    assert record["self_ask"]["grade"] == "correct"  # typed resolves via the clarifying exchange
    assert record["self_ask"]["leaked"] is False
    assert record["diagnostic"]["entropy"] > 0.9
    json.dumps(record)  # must be JSON-serializable for the JSONL writer


def test_run_set_a_item_end_to_end_flat_shows_no_gain_from_asking():
    client = make_mock_client("flat")
    rng = random.Random(0)
    setA, _ = build_matched_sets(client, [GEORGIA], [], target_per_set=1, rng=rng)
    record = run_set_a_item(client, setA[0], diagnostic_n=10, diagnostic_temp=1.0)

    # The flat persona always answers "Atlanta" whatever it's told (the freeze),
    # so it's only correct when the intended reading happens to be "b".
    expected = "correct" if setA[0].intended == "b" else "wrong"
    assert record["self_ask"]["grade"] == expected
    assert record["diagnostic"]["entropy"] == 0.0


def test_run_set_b_item_end_to_end():
    client = make_mock_client("typed")
    rng = random.Random(0)
    _, setB = build_matched_sets(client, [], [CHIPMUNKS], target_per_set=1, rng=rng)
    record = run_set_b_item(client, setB[0], diagnostic_n=10, diagnostic_temp=1.0)

    assert record["set"] == "B"
    assert record["answer_now"]["grade"] == "correct"
    assert record["self_ask"]["grade"] == "correct"
    assert "oracle_clarify" not in record  # condition 2 does not exist for Set B
    json.dumps(record)


# ---- analyze.py sanity on synthetic records ---------------------------------

def test_analyze_predictions_shape(tmp_path):
    from eb.analyze import analyze

    client_typed = make_mock_client("typed")
    client_flat = make_mock_client("flat")
    rng = random.Random(0)

    setA_typed, setB_typed = build_matched_sets(client_typed, [GEORGIA], [CHIPMUNKS], target_per_set=1, rng=rng)
    records = [
        run_set_a_item(client_typed, setA_typed[0], diagnostic_n=10, diagnostic_temp=1.0),
        run_set_b_item(client_typed, setB_typed[0], diagnostic_n=10, diagnostic_temp=1.0),
    ]

    result = analyze(records)
    assert result["n_setA"] == 1
    assert result["n_setB"] == 1
    assert "prediction_1_gain" in result
    assert "prediction_2_confidence_vs_gain_corr" in result
    assert "prediction_3_entropy_vs_gain_corr" in result
    # Oracle-clarify ceiling gain should be positive on Set A (hedge -> correct).
    assert result["prediction_1_gain"]["setA_oracle_gain"] > 0
    # Set B: answer-now and self-ask are both already correct, so gain is ~0.
    assert result["prediction_1_gain"]["setB_selfask_gain"] == pytest.approx(0.0)
