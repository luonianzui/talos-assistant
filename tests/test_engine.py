from dataclasses import replace
import math
import unittest

from talos_assistant.domain import Action, Cell, Effect, GameState, ProbabilityModel
from talos_assistant.engine import is_terminal, legal_actions, state_warnings, transition


def board(*inmates, **kwargs):
    cells = [Cell(i, "office" if i == 0 else "closed") for i in range(16)]
    for c in inmates:
        cells[c.index] = c
    return GameState(cells=tuple(cells), position=1, rules_confirmed=True, **kwargs)


def action(state, kind, target=None):
    return next(a for a in legal_actions(state) if a.kind == kind and (target is None or a.target == target))


def test_complete_counter_graph_and_only_current_cell_can_start():
    s = board(Cell(1, "prisoner", "elephant", 2), Cell(2, "prisoner", "lion", 2), Cell(3, "prisoner", "wolf", 2), Cell(4, "prisoner", "rat", 2), Cell(5, "prisoner", "lion", 1))
    eats = [a for a in legal_actions(s) if a.kind == "eat"]
    assert {(a.source, a.target) for a in eats} == {(1, 2), (1, 3)}


def test_unknown_random_species_never_becomes_equal_probability():
    s = board(Cell(1, "prisoner", "lion", 1), Cell(2, "prisoner", "rat", 1))
    outcomes = transition(s, action(s, "eat", 2))
    assert len(outcomes) == 4
    assert all(o.probability is None for o in outcomes)
    assert all(o.state.cells[2].kind == "closed" and o.state.cells[1].rank == 2 for o in outcomes)
    assert all(not any(a.kind == "replace" for a in legal_actions(o.state)) for o in outcomes)
    assert s.cells[2].kind == "prisoner"


def test_explicit_probability_is_respected_and_input_not_mutated():
    p = ProbabilityModel(species={"lion": .8, "rat": .2}, source="test exact", estimated=False)
    s = board(Cell(1, "prisoner", "lion", 1), Cell(2, "prisoner", "rat", 1), probability=p)
    outcomes = transition(s, action(s, "eat", 2))
    assert {o.state.cells[1].species: o.probability for o in outcomes} == {"lion": .8, "rat": .2}
    assert s.daily_action == "none"
    assert all(o.state.revision == 1 for o in outcomes)


def test_replace_blocks_daily_eat_and_conditions_rope_probabilities():
    p = ProbabilityModel(species={"rat": 1}, ranks={"0": .5, "2": .25, "3": .25}, source="test")
    s = board(Cell(1, "prisoner", "wolf", 2), Cell(2, "prisoner", "elephant", 2), probability=p, effects=(Effect("rope", confirmed=True),))
    outcomes = transition(s, action(s, "replace"))
    assert {(o.state.cells[1].rank, o.probability) for o in outcomes} == {(2, .5), (3, .5)}
    assert all(o.state.daily_action == "replace" for o in outcomes)
    assert all([a.kind for a in legal_actions(o.state)] == ["end"] for o in outcomes)


def test_unimplemented_or_unconfirmed_effect_blocks_instead_of_being_ignored():
    base = board(Cell(1, "prisoner", "lion", 2), Cell(2, "prisoner", "rat", 2))
    for effect in (Effect("future_card", "新卡", confirmed=True), Effect("rope", confirmed=False)):
        s = replace(base, effects=(effect,))
        assert legal_actions(s) == ()
        assert state_warnings(s)
        with unittest.TestCase().assertRaises(ValueError):
            transition(s, Action("eat", source=1, target=2))


def test_unconfirmed_profile_is_explicit_condition_not_silently_confirmed():
    s = board(Cell(1, "prisoner", "lion", 2))
    s = replace(s, rules_confirmed=False)
    assert legal_actions(s)
    assert any("provisional_pc" in warning for warning in state_warnings(s))


def test_rainbow_is_removed_and_unknown_reward_stops_rollout():
    s = board(Cell(1, "prisoner", "lion", 4), Cell(2, "prisoner", "rat", 4), probability=ProbabilityModel(species={"lion": 1}, source="test"))
    outcome, = transition(s, action(s, "eat", 2))
    assert outcome.state.rainbows == 1
    assert outcome.state.cells[1].kind == "closed"
    assert outcome.state.pending_options[0].id == "treasure_choice"
    assert legal_actions(outcome.state) == ()
    assert not is_terminal(outcome.state)


def test_rainbow_cap_is_provisional_profile_rule():
    s = board(Cell(1, "prisoner", "lion", 4), Cell(2, "prisoner", "rat", 4), rainbows=10, effects=(Effect("cagefight", confirmed=True),))
    outcome, = transition(s, action(s, "eat", 2))
    assert outcome.state.cells[1].rank == 4
    assert outcome.state.rainbows == 10
    assert not outcome.state.pending_options


def test_patrol_only_exposes_confirmed_visible_destinations():
    s = board(Cell(1, "prisoner", "lion", 2), Cell(4, "prisoner", "wolf", 2), effects=(Effect("patrol_options", params={"targets": [4, 6]}, confirmed=True),))
    assert {(a.kind, a.target) for a in legal_actions(s)} == {("patrol", 4), ("patrol", 6)}
    with unittest.TestCase().assertRaises(ValueError):
        transition(s, Action("patrol", source=1, target=8))
    outcome, = transition(s, action(s, "patrol", 4))
    assert outcome.probability == 1
    assert outcome.state.position == 4
    assert any(a.kind == "replace" for a in legal_actions(outcome.state))


def test_known_closed_rarity_is_not_rerolled_and_unknown_species_remains_unknown():
    s = board(Cell(2, "closed", rank=3, marks=(Effect("reveal", confirmed=True),)), effects=(Effect("patrol_options", params={"targets": [2]}, confirmed=True),))
    outcomes = transition(s, action(s, "patrol", 2))
    assert len(outcomes) == 4
    assert all(o.state.cells[2].rank == 3 and o.probability is None for o in outcomes)


def test_scratch_adds_two_to_base_and_does_not_need_hidden_species():
    s = board(Cell(1, "prisoner", "wolf", 0), Cell(2, "closed", rank=0, marks=(Effect("scratch", confirmed=True),)), effects=(Effect("cagefight", confirmed=True),))
    outcome, = transition(s, action(s, "eat", 2))
    assert outcome.state.cells[1].rank == 3
    assert outcome.state.cells[2].kind == "closed"


def test_day_boundary_matters_for_nest_and_future_routes_are_not_invented():
    nest = Effect("nest", params={"remaining": 1}, confirmed=True)
    s = board(Cell(2, "closed", marks=(nest,)))
    outcome, = transition(s, action(s, "end"))
    assert outcome.state.cells[2] == Cell(2, "prisoner", "rat", 2)
    assert outcome.state.day == 2
    assert "patrol_options" in outcome.state.unknown_fields
    assert not is_terminal(outcome.state)
    assert legal_actions(outcome.state) == ()


def test_prey_promotes_both_from_same_snapshot_and_expires():
    mark = Effect("prey", params={"partner": 2, "remaining": 1}, confirmed=True)
    s = board(Cell(1, "prisoner", "lion", 2, (mark,)), Cell(2, "prisoner", "wolf", 2))
    outcome, = transition(s, action(s, "end"))
    assert [outcome.state.cells[i].rank for i in (1, 2)] == [3, 3]
    assert not outcome.state.cells[1].marks


def test_changed_code_clears_prey_link():
    mark = Effect("prey", params={"partner": 2, "remaining": 3}, confirmed=True)
    s = board(Cell(1, "prisoner", "lion", 1, (mark,)), Cell(2, "prisoner", "wolf", 1), Cell(3, "prisoner", "rat", 1), probability=ProbabilityModel(species={"rat": 1}, source="test"))
    outcome, = transition(s, action(s, "eat", 3))
    assert not outcome.state.cells[1].marks


def test_saw_requires_explicit_chance_and_can_be_combined_with_fixed_species():
    s = board(Cell(1, "prisoner", "lion", 1), Cell(2, "prisoner", "rat", 1), effects=(Effect("saw", params={"chance": .25}, confirmed=True), Effect("cagefight", confirmed=True)))
    outcomes = transition(s, action(s, "eat", 2))
    assert {(o.state.cells[1].rank, o.probability) for o in outcomes} == {(2, .75), (3, .25)}
    assert not legal_actions(replace(s, effects=(Effect("saw", confirmed=True),)))


def test_tusk_is_pending_not_silently_dropped():
    s = board(Cell(1, "prisoner", "elephant", 1, (Effect("tusk", confirmed=True),)), probability=ProbabilityModel(species={"rat": 1}, ranks={"2": 1}, source="test"))
    outcome, = transition(s, action(s, "replace"))
    assert outcome.state.pending_options[0].id == "tusk_resolution"
    assert not legal_actions(outcome.state)


def test_final_day_ends_without_making_up_another_patrol():
    s = board(Cell(1, "prisoner", "lion", 1), day=20)
    outcome, = transition(s, action(s, "end"))
    assert is_terminal(outcome.state)
    assert outcome.probability == 1
    assert not legal_actions(outcome.state)


def test_treasure_consumption_and_multiple_recipient_upgrades():
    s = board(Cell(1, "prisoner", "lion", 1), Cell(2, "prisoner", "wolf", 2), Cell(3, "prisoner", "rat", 3), treasures=(Effect("wine", params={"changes_species": False}, confirmed=True),))
    outcome, = transition(s, action(s, "treasure"))
    assert [outcome.state.cells[i].rank for i in (1, 2, 3)] == [2, 3, 3]
    assert not outcome.state.treasures


def test_one_of_two_identical_treasures_is_consumed():
    wine = Effect("wine", params={"changes_species": False}, confirmed=True)
    s = board(Cell(1, "prisoner", "lion", 0), treasures=(wine, wine))
    outcome, = transition(s, action(s, "treasure"))
    assert outcome.state.treasures == (wine,)


def test_closed_random_open_has_complete_probability_mass():
    p = ProbabilityModel(species={"lion": .6, "rat": .4}, ranks={"0": .7, "4": .3}, source="test exact", estimated=False)
    s = board(probability=p, effects=(Effect("patrol_options", params={"targets": [2]}, confirmed=True),))
    outcomes = transition(s, action(s, "patrol", 2))
    assert len(outcomes) == 4
    assert math.isclose(sum(o.probability for o in outcomes), 1)
    assert {(o.state.cells[2].species, o.state.cells[2].rank) for o in outcomes} == {("lion", 0), ("lion", 4), ("rat", 0), ("rat", 4)}


def test_visit_open_empty_cell_does_not_invent_an_inmate():
    s = board(Cell(2, "empty"), effects=(Effect("patrol_options", params={"targets": [2]}, confirmed=True),))
    outcome, = transition(s, action(s, "patrol", 2))
    assert outcome.state.cells[2].kind == "empty"
    assert outcome.probability == 1


if __name__ == "__main__":
    # Also runnable before optional developer dependencies have been installed.
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in tuple(globals().items()) if name.startswith("test_") and callable(test))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
