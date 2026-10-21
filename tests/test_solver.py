from __future__ import annotations

from dataclasses import replace
import math
import threading
import time
import unittest
from unittest.mock import patch

from talos_assistant import solver
from talos_assistant.domain import Action, GameState, Outcome, ProbabilityModel, StrategyConfig


class GraphGame:
    """Small independent decision graphs, not a copy of the game engine."""

    def __init__(self, graph, terminal_counts, counts=None):
        self.graph = graph
        self.terminal_counts = terminal_counts
        self.counts = counts or {}

    def state(self, name):
        return GameState(
            source=name, rules_confirmed=True,
            day=21 if name in self.terminal_counts else 1,
            rainbows=self.terminal_counts.get(name, self.counts.get(name, 0)),
        )

    def legal_actions(self, state):
        return tuple(Action(name, label=name) for name, _ in self.graph.get(state.source, ()))

    def transition(self, state, action):
        branches = next(branches for name, branches in self.graph[state.source] if name == action.kind)
        return tuple(Outcome(self.state(target), probability) for target, probability in branches)

    def patched(self):
        return patch.multiple(
            solver.engine,
            legal_actions=self.legal_actions,
            transition=self.transition,
            is_terminal=lambda state: state.source in self.terminal_counts,
            state_warnings=lambda state: (),
        )


def config(mode="ceiling", **kwargs):
    return StrategyConfig(mode=mode, quick_seconds=0.05, total_seconds=1.0, **kwargs)


class DistributionTests(unittest.TestCase):
    def test_strategies_choose_different_synthetic_distributions(self):
        self.assertEqual(solver.strategy_comparison(config("average"))[0].action.kind, "synthetic_safe")
        risky = solver.strategy_comparison(config())[0]
        self.assertEqual(risky.action.kind, "synthetic_risky")
        self.assertEqual(risky.ceiling, 5)
        self.assertAlmostEqual(risky.hit_probability, 0.3)
        self.assertAlmostEqual(risky.mean, 1.5)

    def test_extremely_rare_jackpot_does_not_define_ceiling(self):
        safe = solver.evaluate_distribution(Action("safe"), {2: 1.0}, config())
        rare = solver.evaluate_distribution(Action("rare"), {0: 0.99, 10: 0.01}, config())
        self.assertEqual(rare.ceiling, 0)
        self.assertEqual(solver.rank_evaluations((rare, safe), config())[0].action.kind, "safe")

    def test_threshold_is_inclusive_and_can_be_changed(self):
        distribution = {0: 0.8, 5: 0.2}
        self.assertEqual(solver.evaluate_distribution(Action("a"), distribution, config()).ceiling, 5)
        self.assertEqual(solver.evaluate_distribution(Action("a"), distribution, config(threshold=0.21)).ceiling, 0)
        self.assertEqual(solver.evaluate_distribution(Action("a"), {2: 0.5, 3: 0.5}, config(threshold=1)).ceiling, 2)

    def test_mean_uses_steps_only_after_equal_expected_reward(self):
        slow = solver.evaluate_distribution(Action("slow"), {0: 0.5, 4: 0.5}, config("average"), steps=3)
        fast = solver.evaluate_distribution(Action("fast"), {2: 1.0}, config("average"), steps=1)
        self.assertEqual(solver.rank_evaluations((slow, fast), config("average"))[0].action.kind, "fast")

    def test_ceiling_uses_hit_probability_then_mean(self):
        first = solver.evaluate_distribution(Action("first"), {0: 0.7, 5: 0.3}, config())
        second = solver.evaluate_distribution(Action("second"), {0: 0.6, 5: 0.4}, config())
        third = solver.evaluate_distribution(Action("third"), {1: 0.6, 5: 0.4}, config())
        ranked = solver.rank_evaluations((first, second, third), config())
        self.assertEqual([evaluation.action.kind for evaluation in ranked], ["third", "second", "first"])

    def test_rejects_missing_invalid_or_unormalized_probabilities(self):
        for distribution in ({}, {1: 0.5}, {1: math.nan}, {1: -1.0, 2: 2.0}, {-1: 1.0}, {1: None}):
            with self.subTest(distribution=distribution), self.assertRaises(ValueError):
                solver.evaluate_distribution(Action("a"), distribution, config())
        for bad in (StrategyConfig(mode="unknown"), StrategyConfig(threshold=0), StrategyConfig(total_seconds=-1)):
            with self.assertRaises(ValueError):
                solver.rank_evaluations((), bad)


class SearchTests(unittest.TestCase):
    def test_selected_strategy_changes_downstream_policy(self):
        game = GraphGame(
            {"root": [("begin", [("choice", 1.0)])],
             "choice": [("safe", [("two", 1.0)]), ("risk", [("zero", 0.7), ("five", 0.3)])]},
            {"zero": 0, "two": 2, "five": 5},
        )
        with game.patched():
            average = solver.solve(game.state("root"), config("average"))
            ceiling = solver.solve(game.state("root"), config())
        self.assertEqual(average.status, "exact")
        self.assertEqual(ceiling.status, "exact")
        self.assertEqual(average.candidates[0].distribution, {2: 1.0})
        self.assertEqual(ceiling.candidates[0].distribution, {0: 0.7, 5: 0.3})
        self.assertNotEqual(average.strategy_id, ceiling.strategy_id)
        self.assertEqual(average.state_id, ceiling.state_id)

    def test_ceiling_keeps_nonlocal_distribution_frontier(self):
        # Optimizing the child's own 20% ceiling would select five at 30%.
        # It only happens on half the root branches, so the root should instead
        # choose the child's guaranteed two (50% overall versus 15% for five).
        game = GraphGame(
            {"root": [("begin", [("choice", 0.5), ("zero", 0.5)])],
             "choice": [("safe", [("two", 1.0)]), ("risk", [("zero", 0.7), ("five", 0.3)])]},
            {"zero": 0, "two": 2, "five": 5},
        )
        with game.patched():
            result = solver.solve(game.state("root"), config())
        self.assertTrue(result.complete)
        self.assertEqual(result.candidates[0].distribution, {0: 0.5, 2: 0.5})
        self.assertEqual(result.candidates[0].ceiling, 2)

    def test_global_search_can_prefer_delayed_reward(self):
        game = GraphGame(
            {"root": [("quick", [("one", 1.0)]), ("invest", [("later", 1.0)])],
             "later": [("collect", [("three", 1.0)])]},
            {"one": 1, "three": 3},
        )
        with game.patched():
            result = solver.solve(game.state("root"), config("average"))
        self.assertEqual(result.candidates[0].action.kind, "invest")
        self.assertEqual(result.candidates[0].steps, 2)

    def test_unknown_probability_never_gets_equal_weights(self):
        game = GraphGame({"root": [("unknown", [("zero", None), ("five", None)])]}, {"zero": 0, "five": 5})
        with game.patched():
            result = solver.solve(game.state("root"), config())
        self.assertEqual(result.status, "conditional")
        self.assertFalse(result.complete)
        self.assertIsNone(result.candidates[0].mean)
        self.assertEqual(result.candidates[0].distribution, {})
        self.assertTrue(any("不假设等概率" in warning for warning in result.warnings))

    def test_nonterminal_without_actions_is_not_a_terminal_reward(self):
        game = GraphGame(
            {"root": [("gain", [("pending", 1.0)]), ("stop", [("zero", 1.0)])]},
            {"zero": 0}, counts={"pending": 1},
        )
        with game.patched():
            result = solver.solve(game.state("root"), config())
        self.assertEqual(result.status, "conditional")
        # Conditioned suggestions can still prioritize a verified immediate gain.
        self.assertEqual(result.candidates[0].action.kind, "gain")
        self.assertIn("立即增加 1", result.candidates[0].reason)
        self.assertEqual(result.candidates[0].distribution, {})
        self.assertEqual(result.candidates[1].distribution, {0: 1.0})

    def test_missing_chance_branch_does_not_get_renormalized(self):
        game = GraphGame({"root": [("a", [("five", 0.5), ("pending", 0.5)])]}, {"five": 5})
        with game.patched():
            result = solver.solve(game.state("root"), config())
        self.assertEqual(result.candidates[0].distribution, {})
        self.assertIsNone(result.candidates[0].mean)

    def test_unconfirmed_rules_never_claim_exact(self):
        game = GraphGame({"root": [("a", [("one", 1.0)])]}, {"one": 1})
        with game.patched():
            result = solver.solve(replace(game.state("root"), rules_confirmed=False), config())
        self.assertEqual(result.status, "conditional")
        self.assertFalse(result.complete)
        self.assertFalse(result.candidates[0].exact)

    def test_estimated_model_is_labelled_even_after_exhaustive_search(self):
        game = GraphGame({"root": [("a", [("zero", 0.5), ("two", 0.5)])]}, {"zero": 0, "two": 2})
        state = replace(game.state("root"), probability=ProbabilityModel(species={"rat": 1}, source="合成估计", sample_size=100, estimated=True))
        with game.patched():
            result = solver.solve(state, config())
        self.assertTrue(result.complete)
        self.assertEqual(result.status, "estimated")
        self.assertFalse(result.candidates[0].exact)
        self.assertTrue(any("样本量 100" in warning for warning in result.warnings))

    def test_degenerate_estimated_probability_is_not_known_certainty(self):
        game = GraphGame({"root": [("a", [("two", 1.0)])]}, {"two": 2})
        state = replace(game.state("root"), probability=ProbabilityModel(species={"lion": 1.0}, source="合成单样本估计", sample_size=1, estimated=True))
        with game.patched():
            result = solver.solve(state, config())
        self.assertTrue(result.complete)
        self.assertEqual(result.status, "estimated")
        self.assertFalse(result.candidates[0].exact)

    def test_frontier_cap_never_claims_exact_optimum(self):
        game = GraphGame(
            {"root": [("begin", [("choice", 1.0)])],
             "choice": [("safe", [("two", 1.0)]), ("risk", [("zero", 0.7), ("five", 0.3)])]},
            {"zero": 0, "two": 2, "five": 5},
        )
        with game.patched(), patch.object(solver, "_MAX_FRONTIER", 1):
            result = solver.solve(game.state("root"), config())
        self.assertEqual(result.status, "estimated")
        self.assertFalse(result.complete)
        self.assertFalse(result.candidates[0].exact)
        self.assertAlmostEqual(sum(result.candidates[0].distribution.values()), 1.0)
        self.assertTrue(any("策略分布数量" in warning for warning in result.warnings))

    def test_invalid_transition_probability_is_conditioned_not_numeric(self):
        for probability in ("unknown", math.nan, -1, 2, True):
            game = GraphGame({"root": [("a", [("two", probability)])]}, {"two": 2})
            with self.subTest(probability=probability), game.patched():
                result = solver.solve(game.state("root"), config())
            self.assertEqual(result.status, "conditional")
            self.assertFalse(result.candidates[0].distribution)
            self.assertTrue(any("无效概率" in warning for warning in result.warnings))

    def test_zero_probability_unknown_branch_can_be_ignored(self):
        game = GraphGame({"root": [("a", [("one", 1.0), ("pending", 0.0)])]}, {"one": 1})
        with game.patched():
            result = solver.solve(game.state("root"), config())
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.candidates[0].distribution, {1: 1.0})

    def test_cycle_is_bounded_without_claiming_complete(self):
        game = GraphGame({"root": [("loop", [("root", 1.0)]), ("end", [("two", 1.0)])]}, {"two": 2})
        with game.patched():
            result = solver.solve(game.state("root"), config())
        self.assertEqual(result.status, "estimated")
        self.assertFalse(result.complete)
        self.assertEqual(result.candidates[0].action.kind, "end")
        self.assertTrue(any("循环" in warning for warning in result.warnings))

    def test_depth_limit_is_not_terminal_heuristic(self):
        game = GraphGame({"root": [("a", [("later", 1.0)])], "later": [("b", [("one", 1.0)])]}, {"one": 1})
        with game.patched():
            result = solver.solve(game.state("root"), config(max_depth=1))
        self.assertFalse(result.complete)
        self.assertIsNone(result.candidates[0].mean)

    def test_pre_cancelled_job_and_fingerprints(self):
        cancelled = threading.Event()
        cancelled.set()
        state = solver.demo_state()
        settings = config()
        result = solver.solve(state, settings, cancel_event=cancelled)
        self.assertEqual(result.status, "cancelled")
        self.assertFalse(result.complete)
        self.assertEqual(result.state_id, state.fingerprint())
        self.assertEqual(result.strategy_id, settings.fingerprint())
        self.assertNotEqual(result.state_id, replace(state, rainbows=3).fingerprint())

    def test_deadline_and_progress_bound(self):
        updates = []
        started = time.perf_counter()

        def transition(state, action):
            time.sleep(0.002)
            return (Outcome(replace(state, source=state.source + "x"), 1.0),)

        with patch.multiple(solver.engine, legal_actions=lambda state: (Action("next"),), transition=transition, is_terminal=lambda state: False, state_warnings=lambda state: ()):
            result = solver.solve(
                GameState(rules_confirmed=True),
                StrategyConfig(quick_seconds=0.005, total_seconds=0.025, max_depth=200),
                on_update=lambda update: updates.append((time.perf_counter() - started, update)),
            )
        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertLess(updates[0][0], 0.02)
        self.assertGreaterEqual(len(updates), 2)
        self.assertFalse(result.complete)
        self.assertTrue(any("时间预算" in warning for warning in result.warnings))
        self.assertFalse(result.candidates[0].exact)

    def test_cancellation_during_search(self):
        cancelled = threading.Event()

        def transition(state, action):
            time.sleep(0.002)
            return (Outcome(replace(state, source=state.source + "x"), 1.0),)

        timer = threading.Timer(0.015, cancelled.set)
        with patch.multiple(solver.engine, legal_actions=lambda state: (Action("next"),), transition=transition, is_terminal=lambda state: False, state_warnings=lambda state: ()):
            timer.start()
            try:
                result = solver.solve(GameState(rules_confirmed=True), StrategyConfig(total_seconds=0.5, max_depth=200), cancel_event=cancelled)
            finally:
                timer.cancel()
        self.assertEqual(result.status, "cancelled")
        self.assertLess(result.elapsed, 0.2)

    def test_deadline_checked_after_unknown_transition_early_return(self):
        def transition(state, action):
            time.sleep(0.02)
            return (Outcome(state, None),)

        with patch.multiple(solver.engine, legal_actions=lambda state: (Action("unknown"),), transition=transition, is_terminal=lambda state: False, state_warnings=lambda state: ()):
            result = solver.solve(GameState(rules_confirmed=True), StrategyConfig(quick_seconds=0.002, total_seconds=0.005))
        self.assertFalse(result.complete)
        self.assertFalse(result.candidates[0].distribution)
        self.assertTrue(any("时间预算" in warning for warning in result.warnings))

    def test_large_depth_setting_cannot_overflow_python_stack(self):
        def transition(state, action):
            return (Outcome(replace(state, source=state.source + "x"), 1.0),)

        with patch.multiple(solver.engine, legal_actions=lambda state: (Action("next"),), transition=transition, is_terminal=lambda state: False, state_warnings=lambda state: ()):
            result = solver.solve(GameState(rules_confirmed=True), config(max_depth=5000))
        self.assertFalse(result.complete)
        self.assertTrue(any("深度上限" in warning for warning in result.warnings))

    def test_real_engine_demo_has_numeric_exact_result(self):
        result = solver.solve(solver.demo_state(), config())
        self.assertEqual(result.status, "exact")
        self.assertTrue(result.complete)
        self.assertEqual(result.candidates[0].action.kind, "end")
        self.assertEqual(result.candidates[0].distribution, {2: 1.0})

    def test_failing_callback_does_not_drop_final_result(self):
        def callback(update):
            raise RuntimeError("UI already closed")

        result = solver.solve(solver.demo_state(), config(), on_update=callback)
        self.assertEqual(result.status, "exact")
        self.assertTrue(any("进度接收器" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
