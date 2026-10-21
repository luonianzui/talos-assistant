from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from talos_assistant.domain import Cell, Effect, GameState, ProbabilityModel, StrategyConfig, state_from_dict, state_to_dict, validate_state
from talos_assistant.storage import Session


def rejected(payload):
    with unittest.TestCase().assertRaises(ValueError):
        state_from_dict(payload)


def test_confirmation_strings_cannot_enable_confirmed_rules_or_effects():
    for value in ("false", "true", 0, 1, None):
        rejected({"rules_confirmed": value})
        rejected({"effects": [{"id": "cagefight", "confirmed": value}]})
        rejected({"probability": {"estimated": value}})


def test_numeric_fields_reject_bool_fraction_and_string():
    for field in ("day", "max_days", "position", "rainbows", "replacements_left", "revision"):
        for value in (True, 1.0, "1"):
            rejected({field: value})
    for field in ("index", "rank"):
        for value in (True, 1.0, "1"):
            data = state_to_dict(GameState())
            data["cells"][1][field] = value
            rejected(data)


def test_malformed_containers_are_clear_value_errors():
    for field in ("cells", "effects", "treasures", "pending_options", "unknown_fields"):
        for value in (None, "day", 1, {}):
            rejected({field: value})
    for value in (None, [], "uncertain"):
        rejected({"probability": value})
        rejected({"effects": [{"id": "saw", "params": value}]})
    rejected({"cells": [None] * 16})
    rejected({"cells": [{}] * 16})


def test_unknown_field_typo_is_not_silently_discarded():
    rejected({"rules_confrimed": True})
    rejected({"effects": [{"id": "rope", "confimed": True}]})
    rejected({"probability": {"specie": {"rat": 1}}})
    cells = state_to_dict(GameState())["cells"]
    cells[1]["mark"] = "scratch"
    rejected({"cells": cells})


def test_effect_params_are_serializable_finite_json_objects():
    for params in ({"chance": float("nan")}, {"chance": float("inf")}, {2: "target"}, {"bad": object()}):
        rejected({"effects": [{"id": "saw", "params": params}]})
    for effect_id in ("", " ", 3):
        rejected({"effects": [{"id": effect_id}]})


def test_probability_models_reject_boolean_values_and_invalid_metadata():
    for probs in ({"rat": True}, {"rat": float("nan")}, {"rat": float("inf")}, {"rat": 10**400}, {"rat": .5}):
        rejected({"probability": {"species": probs, "source": "test"}})
    for sample_size in (-1, True, 1.5, "100"):
        rejected({"probability": {"sample_size": sample_size}})
    rejected({"probability": {"species": {"rat": 1}, "source": ""}})
    rejected({"probability": {"species": {"rat": 1}, "source": {}}})


def test_direct_dataclass_construction_is_validated_too():
    invalid = (
        replace(GameState(), effects=(Effect("rope", confirmed="false"),)),
        replace(GameState(), probability=ProbabilityModel(species={"rat": True}, source="test")),
        replace(GameState(), position=1.0),
        replace(GameState(), unknown_fields="day"),
        replace(GameState(), effects=("rope",)),
    )
    for state in invalid:
        with unittest.TestCase().assertRaises(ValueError):
            validate_state(state)


def test_unknown_information_remains_representable_not_coerced_to_defaults():
    cells = tuple(Cell(i, "office" if i == 0 else "unknown") for i in range(16))
    state = GameState(cells=cells, position=None, rainbows=None, unknown_fields=("position", "rainbows", "marks"))
    result = state_from_dict(json.loads(json.dumps(state_to_dict(state))))
    assert result == state
    assert result.position is None and result.rainbows is None


def test_session_roundtrip_retains_effect_parameters_and_probability_provenance():
    state = GameState(
        effects=(Effect("saw", "锈锯", {"chance": .25}, True),),
        pending_options=(Effect("unknown_reward", "待选珍藏", {"candidates": ["a", "b"]}),),
        probability=ProbabilityModel(species={"lion": .7, "rat": .3}, ranks={"0": 1}, source="人工标记样本", sample_size=25, estimated=True),
        unknown_fields=("patrol_options",),
    )
    session = Session()
    session.record(state, "验证往返")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.json"
        session.save(path, StrategyConfig())
        restored = Session.load(path).state()
    assert restored == state
    assert restored.fingerprint() == state.fingerprint()


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in tuple(globals().items()) if name.startswith("test_") and callable(test))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
