from dataclasses import replace
import json
import pytest

from talos_assistant.domain import GameState, Recommendation, StrategyConfig
from talos_assistant.storage import ResultGate, Session, Settings


def test_old_results_rejected_after_mode_change_and_cancellation():
    gate = ResultGate()
    state = GameState()
    average = StrategyConfig(mode="average")
    ceiling = StrategyConfig(mode="ceiling")
    old = gate.begin(state, average)
    rec = Recommendation(state.fingerprint(), average.fingerprint(), "average")
    assert gate.accepts(old, rec)
    new = gate.begin(state, ceiling)
    assert not gate.accepts(old, rec)
    assert not gate.accepts(new, rec)
    current = Recommendation(state.fingerprint(), ceiling.fingerprint(), "ceiling")
    assert gate.accepts(new, current)
    gate.invalidate()
    assert not gate.accepts(new, current)


def test_observation_roundtrip_does_not_double_count(tmp_path):
    session = Session()
    state = GameState(rainbows=3)
    assert session.record(state, "截图")
    assert not session.record(replace(state, revision=4), "重复截图")
    session.record(replace(state, day=2), "下一天")
    path = tmp_path / "game.json"
    session.save(path, StrategyConfig())
    loaded = Session.load(path)
    assert len(loaded.entries) == 2
    assert loaded.state().rainbows == 3
    assert loaded.state(0).day == 1


def test_invalid_import_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cells": []}), encoding="utf8")
    with pytest.raises(ValueError):
        Session.load(path)


def test_strategy_persists_without_resetting_threshold(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    assert settings.strategy().mode == "ceiling"
    settings.save_strategy(StrategyConfig(mode="average", threshold=.3))
    loaded = Settings(path).strategy()
    assert loaded.mode == "average"
    assert loaded.threshold == .3
