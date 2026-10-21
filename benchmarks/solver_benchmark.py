"""Repeatable local solver checks; every probability below is synthetic.

Run: .venv\\Scripts\\python.exe benchmarks\\solver_benchmark.py
Save JSON: add --output artifacts/solver-benchmark.json
No game screenshots, credentials, network, or third-party algorithms are used.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talos_assistant import solver
from talos_assistant.domain import Action, GameState, Outcome, StrategyConfig


def run_demo(repeats=20):
    reports = []
    for mode in ("average", "ceiling"):
        durations = []
        first_updates = []
        for _ in range(repeats):
            started = time.perf_counter()
            updates = []
            result = solver.solve(solver.demo_state(), StrategyConfig(mode=mode), on_update=lambda update: updates.append(time.perf_counter() - started))
            durations.append(time.perf_counter() - started)
            first_updates.append(updates[0])
            assert result.status == "exact" and result.candidates[0].mean == 2
        reports.append({
            "scenario": "synthetic_verified_final_day",
            "synthetic": True,
            "probability_source": "constructed_test_model_not_game_data",
            "mode": mode,
            "threshold": 0.2,
            "runs": repeats,
            "timing_aggregation": "median_over_runs",
            "elapsed_ms": round(statistics.median(durations) * 1000, 3),
            "first_update_ms": round(statistics.median(first_updates) * 1000, 3),
            "max_ms": round(max(durations) * 1000, 3),
            "max_first_update_ms": round(max(first_updates) * 1000, 3),
            "status": result.status,
            "complete": result.complete,
            "action": result.candidates[0].action.kind,
            "mean": round(result.candidates[0].mean, 4),
            "ceiling": result.candidates[0].ceiling,
            "hit_probability": round(result.candidates[0].hit_probability, 6),
        })
    return reports


def run_branching_model():
    """A four-stage, observable, synthetic decision process."""
    def legal_actions(state):
        return (Action("safe"), Action("risky")) if state.day <= state.max_days else ()

    def transition(state, action):
        if action.kind == "safe":
            return (Outcome(replace(state, day=state.day + 1, rainbows=state.rainbows + 1), 1.0),)
        return (
            Outcome(replace(state, day=state.day + 1), 0.7),
            Outcome(replace(state, day=state.day + 1, rainbows=state.rainbows + 3), 0.3),
        )

    reports = []
    for mode in ("average", "ceiling"):
        updates = []
        started = time.perf_counter()
        with patch.multiple(solver.engine, legal_actions=legal_actions, transition=transition, state_warnings=lambda state: (), is_terminal=lambda state: state.day > state.max_days):
            result = solver.solve(
                GameState(max_days=4, rules_confirmed=True, source="演示：合成四阶段模型"),
                StrategyConfig(mode=mode, quick_seconds=0.1, total_seconds=1.0),
                on_update=lambda update: updates.append(time.perf_counter() - started),
            )
        assert result.candidates and result.candidates[0].distribution
        reports.append({
            "scenario": "synthetic_four_stage_decision_model",
            "synthetic": True,
            "probability_source": "constructed_test_model_not_game_data",
            "mode": mode,
            "threshold": 0.2,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "first_update_ms": round(updates[0] * 1000, 3),
            "status": result.status,
            "complete": result.complete,
            "action": result.candidates[0].action.kind,
            "mean": round(result.candidates[0].mean, 4),
            "ceiling": result.candidates[0].ceiling,
            "hit_probability": round(result.candidates[0].hit_probability, 6),
        })
    return reports


def run_deadline():
    def transition(state, action):
        time.sleep(0.001)
        return (Outcome(replace(state, source=state.source + "x"), 1.0),)

    reports = []
    for mode in ("average", "ceiling"):
        updates = []
        started = time.perf_counter()
        with patch.multiple(solver.engine, legal_actions=lambda state: (Action("next"),), transition=transition, state_warnings=lambda state: (), is_terminal=lambda state: False):
            result = solver.solve(
                GameState(rules_confirmed=True),
                StrategyConfig(mode=mode, quick_seconds=0.01, total_seconds=0.05, max_depth=500),
                on_update=lambda update: updates.append(time.perf_counter() - started),
            )
        elapsed = time.perf_counter() - started
        assert elapsed < 0.2 and not result.complete
        candidate = result.candidates[0] if result.candidates else None
        reports.append({
            "scenario": "synthetic_bounded_search_50ms_budget",
            "synthetic": True,
            "probability_source": "constructed_test_model_not_game_data",
            "mode": mode,
            "threshold": 0.2,
            "elapsed_ms": round(elapsed * 1000, 3),
            "first_update_ms": round(updates[0] * 1000, 3),
            "updates": len(updates),
            "status": result.status,
            "complete": result.complete,
            "action": candidate.action.kind if candidate else None,
            "mean": candidate.mean if candidate else None,
            "ceiling": candidate.ceiling if candidate else None,
            "hit_probability": candidate.hit_probability if candidate else None,
        })
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark synthetic solver models; these are not observed game probabilities.")
    parser.add_argument("--output", type=Path, help="Also save the same JSON report to this path.")
    args = parser.parse_args()
    reports = [*run_demo(), *run_branching_model(), *run_deadline()]
    serialized = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
