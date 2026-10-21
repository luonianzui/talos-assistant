"""Bounded planning over verified game transitions.

This module does not fill in unknown game probabilities.  A numeric distribution
always describes a complete policy that reaches a real terminal state.  The
ceiling objective needs a distribution frontier: greedily picking the locally
highest quantile at every child is not a valid way to optimize a root quantile.

``solve`` is synchronous; desktop callers run it in a worker and enqueue the
``on_update`` snapshots for their UI thread.  No third-party solver is required.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from time import perf_counter
from typing import Callable, Iterable, Mapping

from . import engine
from .domain import Action, ActionEvaluation, Cell, Effect, GameState, ProbabilityModel, Recommendation, StrategyConfig

_EPSILON = 1e-10
_MAX_FRONTIER = 128
_MAX_MEMO = 5000
_MAX_RECURSION_DEPTH = 200


def _validate_config(config: StrategyConfig) -> None:
    if config.mode not in ("average", "ceiling"):
        raise ValueError("策略必须是 average 或 ceiling")
    if not math.isfinite(config.threshold) or not 0 < config.threshold <= 1:
        raise ValueError("目标达成概率门槛必须在 0 到 1 之间，且不能为 0")
    for value in (config.quick_seconds, config.total_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("计算时间预算必须为正数")
    if not isinstance(config.max_depth, int) or config.max_depth < 1:
        raise ValueError("搜索深度必须为正整数")


def _distribution(distribution: Mapping[int, float]) -> dict[int, float]:
    if not distribution:
        raise ValueError("终局分布不能为空")
    cleaned: dict[int, float] = {}
    for count, probability in distribution.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("终局绮彩数量必须是非负整数")
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or probability < 0:
            raise ValueError("终局概率必须是有限非负数")
        if probability > 0:
            cleaned[count] = float(probability)
    total = sum(cleaned.values())
    if not cleaned or abs(total - 1.0) > 1e-6:
        raise ValueError("终局概率总和必须为 1，不能为未知概率补权重")
    return {count: probability / total for count, probability in sorted(cleaned.items())}


def evaluate_distribution(
    action: Action,
    distribution: Mapping[int, float],
    config: StrategyConfig,
    steps: float = 0,
    exact: bool = False,
) -> ActionEvaluation:
    """Evaluate a complete terminal distribution using the selected objective."""
    _validate_config(config)
    probabilities = _distribution(distribution)
    if not math.isfinite(steps) or steps < 0:
        raise ValueError("预计步骤必须为有限非负数")
    mean = sum(count * probability for count, probability in probabilities.items())
    tail = 0.0
    ceiling = min(probabilities)
    hit_probability = 1.0
    for count, probability in sorted(probabilities.items(), reverse=True):
        tail += probability
        if tail + _EPSILON >= config.threshold:
            ceiling, hit_probability = count, min(1.0, tail)
            break
    if config.mode == "average":
        reason = f"预计终局绮彩 {mean:.2f} 个；同收益优先更少步骤。"
    else:
        reason = f"以至少 {config.threshold:.0%} 的达成门槛争取 {ceiling} 个绮彩；该策略达成率 {hit_probability:.1%}。"
    return ActionEvaluation(
        action=action,
        mean=mean,
        ceiling=ceiling,
        hit_probability=hit_probability,
        distribution=probabilities,
        steps=float(steps),
        reason=reason,
        exact=exact,
    )


def _evaluation_key(evaluation: ActionEvaluation, config: StrategyConfig) -> tuple[float, ...]:
    if evaluation.mean is None or evaluation.ceiling is None or evaluation.hit_probability is None:
        return (0.0,)
    if config.mode == "average":
        return (1.0, evaluation.mean, -evaluation.steps)
    return (1.0, float(evaluation.ceiling), evaluation.hit_probability, evaluation.mean, -evaluation.steps)


def rank_evaluations(evals: Iterable[ActionEvaluation], config: StrategyConfig) -> tuple[ActionEvaluation, ...]:
    """Rank complete numeric policies first; preserve input order for true ties.

Conditional candidates remain in the returned list.  Their position is a display
order, not a claim that a known policy outperforms an unknown one.
"""
    _validate_config(config)
    return tuple(sorted(evals, key=lambda evaluation: _evaluation_key(evaluation, config), reverse=True))


@dataclass
class _Policy:
    distribution: dict[int, float]
    steps: float


@dataclass
class _SearchResult:
    policies: list[_Policy]
    exhaustive: bool = True
    conditions: tuple[str, ...] = ()


class _HaltSearch(Exception):
    pass


class _Search:
    def __init__(self, state: GameState, config: StrategyConfig, cancel_event, on_update):
        self.state = state
        self.config = config
        self.cancel_event = cancel_event
        self.on_update = on_update
        self.started = perf_counter()
        self.deadline = self.started + min(10.0, config.total_seconds)
        self.update_interval = min(2.0, config.quick_seconds, config.total_seconds)
        self.next_update = self.started + self.update_interval
        self.state_id = state.fingerprint()
        self.strategy_id = config.fingerprint()
        self.candidates: dict[str, ActionEvaluation] = {}
        self.immediate_summaries: dict[str, str] = {}
        self.immediate_priorities: dict[str, tuple[int, int]] = {}
        self.warnings: dict[str, None] = {}
        self.memo: dict[tuple[str, int], _SearchResult] = {}
        self.cancelled = False
        self.timed_out = False
        self.limited = False
        self.unknown = False
        self.estimated_probability = False
        self.nodes = 0
        self.root_exhaustive = False
        self.root_has_actions = False

    def note(self, text: str) -> None:
        if text:
            self.warnings[text] = None

    def check(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.cancelled = True
            self.note("计算已取消；保留结果仅供查看，当前建议未完成。")
            raise _HaltSearch
        now = perf_counter()
        if now >= self.deadline:
            self.timed_out = True
            self.note("已达到计算时间预算；仅展示当前已找到的完整策略，未证明最优。")
            raise _HaltSearch
        if now >= self.next_update:
            self.publish()
            self.next_update = perf_counter() + self.update_interval

    def snapshot(self, final: bool = False) -> Recommendation:
        numeric = any(candidate.distribution for candidate in self.candidates.values())
        complete = final and self.root_exhaustive and not (self.cancelled or self.timed_out or self.limited or self.unknown)
        if self.cancelled:
            status = "cancelled"
        elif not self.root_has_actions and not engine.is_terminal(self.state) and not self.timed_out:
            status = "blocked"
        elif self.unknown or not numeric and not engine.is_terminal(self.state):
            status = "conditional"
        elif complete and not self.estimated_probability and self.state.rules_confirmed:
            status = "exact"
        else:
            status = "estimated"
        candidates = rank_evaluations(self.candidates.values(), self.config)
        warnings = list(self.warnings)
        if self.unknown:
            # No whole-game ranking is justified while alternatives are unknown.
            # This separate display order only prioritizes already known gains.
            candidates = tuple(sorted(
                candidates,
                key=lambda candidate: self.immediate_priorities.get(candidate.action.key, (0, 0)),
                reverse=True,
            ))
            if candidates:
                warnings.append("部分后续条件未知；当前排列仅参考已知即时增益，不代表终局最优。已有数值只对应可验证的完整策略。")
        return Recommendation(
            state_id=self.state_id,
            strategy_id=self.strategy_id,
            mode=self.config.mode,
            candidates=candidates,
            status=status,
            elapsed=perf_counter() - self.started,
            warnings=tuple(warnings),
            complete=complete,
        )

    def publish(self, final: bool = False) -> None:
        if self.on_update is not None:
            try:
                self.on_update(self.snapshot(final=final))
            except Exception:
                # A disposed UI may reject an update while its worker winds down.
                self.note("进度接收器未能接收更新；计算结果仍可从返回值读取。")

    def blocked(self, *conditions: str) -> _SearchResult:
        self.unknown = True
        for condition in conditions:
            self.note(condition)
        return _SearchResult([], False, tuple(conditions))

    def record_immediate(self, state: GameState, action: Action, outcomes) -> None:
        if state.fingerprint() != self.state_id:
            return
        outcomes = tuple(outcome for outcome in outcomes if outcome.probability is None or outcome.probability > 0)
        if not outcomes:
            return
        descriptions: list[str] = []
        guaranteed_gain = guaranteed_upgrade = 0
        counts = [outcome.state.rainbows for outcome in outcomes]
        if state.rainbows is not None and all(count is not None for count in counts):
            low, high = min(counts) - state.rainbows, max(counts) - state.rainbows
            guaranteed_gain = low
            if high > 0:
                descriptions.append(f"立即增加 {low} 个绮彩" if low == high else f"立即绮彩增量为 {low} 至 {high} 个，触发机会尚须核对")
        before = next((cell for cell in state.cells if cell.index == action.source), None)
        after = [next((cell for cell in outcome.state.cells if cell.index == action.source), None) for outcome in outcomes]
        if before is not None and before.rank is not None and after and all(cell is not None and cell.kind == "prisoner" and cell.rank is not None for cell in after):
            upgrades = [cell.rank - before.rank for cell in after]
            guaranteed_upgrade = max(0, min(upgrades))
            if guaranteed_upgrade > 0:
                descriptions.append(f"当前囚犯至少提升 {guaranteed_upgrade} 阶")
        population = sum(cell.kind == "prisoner" for cell in state.cells)
        reductions = [population - sum(cell.kind == "prisoner" for cell in outcome.state.cells) for outcome in outcomes]
        if reductions and min(reductions) > 0:
            descriptions.append(f"场上囚犯至少减少 {min(reductions)} 名")
        if action.kind == "end":
            descriptions.append("结束本日" if state.day < state.max_days else "结束本局并结算当前已收容绮彩")
        if descriptions:
            self.immediate_summaries[action.key] = "按当前规则，" + "；".join(descriptions) + "。"
        self.immediate_priorities[action.key] = (guaranteed_gain, guaranteed_upgrade)

    def policy_key(self, policy: _Policy) -> tuple[float, ...]:
        self.check()
        # During chance-node convolution all candidates have the same partial
        # probability mass.  Normalize only for the pruning heuristic, never as
        # a substitute for a missing branch in a public evaluation.
        total = sum(policy.distribution.values())
        normalized = {count: probability / total for count, probability in policy.distribution.items()}
        evaluation = evaluate_distribution(Action("policy"), normalized, self.config, steps=policy.steps)
        return _evaluation_key(evaluation, self.config)

    def dominates(self, left: _Policy, right: _Policy) -> bool:
        """First-order stochastic dominance, with steps breaking equal laws."""
        self.check()
        thresholds = sorted(set(left.distribution) | set(right.distribution), reverse=True)
        left_tail = right_tail = 0.0
        strict = False
        for count in thresholds:
            left_tail += left.distribution.get(count, 0.0)
            right_tail += right.distribution.get(count, 0.0)
            if left_tail < right_tail - _EPSILON:
                return False
            if left_tail > right_tail + _EPSILON:
                strict = True
        return strict or left.steps <= right.steps + _EPSILON

    def insert_policy(self, frontier: list[_Policy], policy: _Policy) -> bool:
        """Return False only when a heuristic frontier limit is applied."""
        self.check()
        if self.config.mode == "average":
            if not frontier or self.policy_key(policy) > self.policy_key(frontier[0]):
                frontier[:] = [policy]
            return True
        removed: list[_Policy] = []
        for existing in frontier:
            if self.dominates(existing, policy):
                return True
            if self.dominates(policy, existing):
                removed.append(existing)
        for existing in removed:
            frontier.remove(existing)
        frontier.append(policy)
        if len(frontier) > _MAX_FRONTIER:
            frontier.sort(key=self.policy_key, reverse=True)
            del frontier[_MAX_FRONTIER:]
            self.limited = True
            self.note("策略分布数量超过搜索上限；已保留候选策略，但尚未证明最优。")
            return False
        return True

    def search(self, state: GameState, remaining: int, path: frozenset[str]) -> _SearchResult:
        self.check()
        self.nodes += 1
        if engine.is_terminal(state):
            if state.rainbows is None:
                return self.blocked("终局绮彩累计数量未知，无法计算本局总数。")
            return _SearchResult([_Policy({state.rainbows: 1.0}, 0.0)])
        if remaining <= 0:
            self.limited = True
            condition = "达到搜索深度上限；未将当前棋盘视为终局。"
            self.note(condition)
            return _SearchResult([], False, (condition,))
        state_id = state.fingerprint()
        if state_id in path:
            self.limited = True
            condition = "检测到不推进局面的循环路径；已停止该分支，未宣称全局最优。"
            self.note(condition)
            return _SearchResult([], False, (condition,))
        memo_key = (state_id, remaining)
        if memo_key in self.memo:
            return self.memo[memo_key]
        actions = engine.legal_actions(state)
        self.check()
        if not actions:
            warnings = tuple(engine.state_warnings(state))
            self.check()
            return self.blocked(*(warnings or ("非终局缺少可确认的后续动作，请补全巡监选项或效果描述。",)))
        path = path | {state_id}
        frontier: list[_Policy] = []
        exhaustive = state.rules_confirmed
        conditions: dict[str, None] = {}
        if not state.rules_confirmed:
            self.unknown = True
            conditions["当前规则尚未确认，只能给出条件建议。"] = None
        for action in actions:
            result = self.action_policies(state, action, remaining, path)
            exhaustive = exhaustive and result.exhaustive
            conditions.update(dict.fromkeys(result.conditions))
            for policy in result.policies:
                exhaustive = self.insert_policy(frontier, policy) and exhaustive
        result = _SearchResult(frontier, exhaustive, tuple(conditions))
        # Incomplete results can depend on the current recursion path or budget.
        if exhaustive and len(self.memo) < _MAX_MEMO:
            self.memo[memo_key] = result
        return result

    def action_policies(self, state: GameState, action: Action, remaining: int, path: frozenset[str]) -> _SearchResult:
        self.check()
        try:
            outcomes = engine.transition(state, action)
        except (ValueError, NotImplementedError) as error:
            self.check()
            return self.blocked(f"动作“{action.label or action.kind}”暂不能验证：{error}")
        self.check()
        if not outcomes:
            return self.blocked(f"动作“{action.label or action.kind}”的后续结果尚未定义。")
        if any(outcome.probability is not None and (type(outcome.probability) not in (int, float) or not math.isfinite(outcome.probability) or not 0 <= outcome.probability <= 1) for outcome in outcomes):
            return self.blocked("转移包含无效概率，已停止数值计算。")
        self.record_immediate(state, action, outcomes)
        self.check()
        if any(outcome.probability is None for outcome in outcomes):
            explanations = tuple(dict.fromkeys(outcome.explanation for outcome in outcomes if outcome.explanation))
            return self.blocked("存在未公开或未录入的随机概率；不假设等概率。", *explanations)
        total = sum(outcome.probability for outcome in outcomes)
        if abs(total - 1.0) > 1e-6:
            return self.blocked("转移概率不完整，已停止数值计算。")
        positive_outcomes = [outcome for outcome in outcomes if outcome.probability > 0]
        # The current domain does not record which individual transition used a
        # table.  Conservatively retain the model's estimated label even for a
        # degenerate p=1 table: one represented outcome does not prove certainty.
        if state.probability.estimated and (state.probability.species or state.probability.ranks):
            self.estimated_probability = True
            source = state.probability.source or "已录入模型"
            self.note(f"已录入的概率模型“{source}”为估计值（样本量 {state.probability.sample_size}）；数值保守标为估计，不代表游戏真实概率保证。")
        mixtures = [_Policy({}, 1.0)]
        exhaustive = state.rules_confirmed
        conditions: dict[str, None] = {}
        for outcome in positive_outcomes:
            self.check()
            child = self.search(outcome.state, remaining - 1, path)
            exhaustive = exhaustive and child.exhaustive
            conditions.update(dict.fromkeys(child.conditions))
            if not child.policies:
                # A distribution missing a chance branch is not a terminal law.
                return _SearchResult([], False, tuple(conditions))
            combined: list[_Policy] = []
            weight = outcome.probability / total
            for mixture in mixtures:
                for continuation in child.policies:
                    self.check()
                    distribution = dict(mixture.distribution)
                    for count, probability in continuation.distribution.items():
                        distribution[count] = distribution.get(count, 0.0) + weight * probability
                    policy = _Policy(distribution, mixture.steps + weight * continuation.steps)
                    exhaustive = self.insert_policy(combined, policy) and exhaustive
            mixtures = combined
        return _SearchResult(mixtures, exhaustive, tuple(conditions))

    def run(self) -> Recommendation:
        if not self.state.rules_confirmed:
            self.unknown = True
            self.note("规则尚未由实际玩法确认；本次不宣称精确最优。")
        try:
            self.check()
            for warning in engine.state_warnings(self.state):
                self.note(warning)
            self.check()
            if engine.is_terminal(self.state):
                self.root_exhaustive = self.state.rainbows is not None and self.state.rules_confirmed
                if self.state.rainbows is None:
                    self.unknown = True
                    self.note("本局已结束，但累计绮彩数量未知。")
                else:
                    self.note(f"本局已结束，累计绮彩 {self.state.rainbows} 个，无需继续操作。")
            else:
                actions = engine.legal_actions(self.state)
                self.root_has_actions = bool(actions)
                self.check()
                for action in actions:
                    self.candidates[action.key] = ActionEvaluation(action, reason="正在验证该动作的后续策略。")
                self.publish()
                if not actions:
                    self.unknown = True
                    self.note("当前没有可确认的合法操作，请先补全缺失状态或规则。")
                else:
                    exhaustive = self.state.rules_confirmed
                    for action in actions:
                        self.check()
                        result = self.action_policies(self.state, action, min(self.config.max_depth, _MAX_RECURSION_DEPTH), frozenset({self.state_id}))
                        exhaustive = exhaustive and result.exhaustive
                        if result.policies:
                            best = max(result.policies, key=self.policy_key)
                            evaluation = evaluate_distribution(
                                action, best.distribution, self.config,
                                steps=best.steps,
                                exact=result.exhaustive and not self.estimated_probability and self.state.rules_confirmed,
                            )
                            if not result.exhaustive:
                                evaluation = replace(evaluation, reason=evaluation.reason + " 这是当前找到的完整策略，尚未证明最佳。")
                            self.candidates[action.key] = replace(evaluation, conditions=result.conditions)
                        else:
                            conditions = result.conditions or ("缺少可验证的完整终局路径。",)
                            self.candidates[action.key] = ActionEvaluation(
                                action,
                                reason=self.immediate_summaries.get(action.key, "") + "条件操作：补全后才能比较终局收益；这里只按已知即时增益排列。",
                                conditions=conditions,
                            )
                    self.check()
                    self.root_exhaustive = exhaustive
        except _HaltSearch:
            pass
        finally:
            if self.timed_out or self.cancelled:
                for key, candidate in tuple(self.candidates.items()):
                    if not candidate.distribution and not candidate.conditions:
                        condition = "计算已取消。" if self.cancelled else "该候选尚未在时间预算内完成求解。"
                        self.candidates[key] = replace(candidate, reason=condition, conditions=(condition,))
            self.publish(final=True)
        return self.snapshot(final=True)


def solve(
    state: GameState,
    config: StrategyConfig,
    cancel_event=None,
    on_update: Callable[[Recommendation], None] | None = None,
) -> Recommendation:
    """Return a snapshot-bound recommendation within a bounded search budget.

    There is an immediate legal-action progress snapshot and further updates at
    most every ``quick_seconds`` (capped at 2 seconds).  Work stops at
    ``total_seconds`` (capped at 10 seconds), cancellation, or the configured
    depth (at most 200 transitions to protect the Python stack).  Callbacks must
    be non-blocking.  ``complete`` means exhaustive model
    comparison, not merely that this synchronous call has returned.
    """
    _validate_config(config)
    # frozen dataclasses still contain mutable effect/probability dictionaries.
    return _Search(deepcopy(state), config, cancel_event, on_update).run()


def demo_state() -> GameState:
    """A deliberately synthetic final-day position supported by the engine.

    It avoids unknown treasure choices.  All continuations collect two existing
    rainbows, so the fewest-step tie breaker correctly prefers ending the run.
    Strategy differences are demonstrated separately by strategy_comparison.
    """
    cells = [Cell(index, "office" if index == 0 else "closed") for index in range(16)]
    cells[1] = Cell(1, "prisoner", "lion", 1)
    cells[2] = Cell(2, "prisoner", "wolf", 1)
    return GameState(
        cells=tuple(cells), day=20, position=1, rainbows=2,
        replacements_left=0,
        effects=(Effect("cagefight", "演示：困兽斗", confirmed=True),),
        probability=ProbabilityModel(
            species={"lion": 1.0}, ranks={"1": 1.0},
            source="演示合成概率；仅用于界面和求解验证，不代表实机", estimated=False,
        ),
        rules_confirmed=True,
        source="演示合成局面；非实机截图，规则确认仅限这个人工模型",
    )


def strategy_comparison(config: StrategyConfig) -> tuple[ActionEvaluation, ...]:
    """Synthetic distributions for an explanatory UI, never game inference."""
    candidates = (
        evaluate_distribution(Action("synthetic_safe", label="合成示例：稳定获得 2 个"), {2: 1.0}, config, steps=1, exact=True),
        evaluate_distribution(Action("synthetic_risky", label="合成示例：30% 获得 5 个，其余为 0"), {0: 0.7, 5: 0.3}, config, steps=1, exact=True),
    )
    return rank_evaluations(candidates, config)
