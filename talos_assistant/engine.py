"""Conservative, independently written rules for visible Tartarus positions.

This is not an official game implementation. Unknown odds stay unknown; unknown
effects and the next unseen patrol/reward stop a rollout instead of disappearing.
See data/rules_reference.json for the boundary of the provisional PC profile.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import isfinite

from .domain import Action, Cell, Effect, GameState, Outcome, SPECIES, validate_state

PREY = {"elephant": ("lion", "wolf"), "lion": ("wolf", "rat"), "wolf": ("rat",), "rat": ("elephant",)}
SUPPORTED_TOOLS = {"patrol_options", "rope", "pestle", "rathead", "saw", "muzzle", "cagefight"}
SUPPORTED_MARKS = {"scratch", "nest", "prey", "tusk", "reveal"}
SUPPORTED_TREASURES = {"wine"}
SUPPORTED_EVENTS = {"chain"}


def _cell(s: GameState, index: int | None) -> Cell | None:
    return next((c for c in s.cells if c.index == index), None)


def _effect(s: GameState, name: str) -> Effect | None:
    return next((e for e in s.effects if e.id == name), None)


def _mark(c: Cell, name: str) -> Effect | None:
    return next((m for m in c.marks if m.id == name), None)


def _put(s: GameState, *cells: Cell) -> GameState:
    updates = {c.index: c for c in cells}
    return replace(s, cells=tuple(updates.get(c.index, c) for c in s.cells))


def _pending(s: GameState, name: str, text: str, **params) -> GameState:
    return replace(s, pending_options=s.pending_options + (Effect(name, text, params),))


def _probability(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a * b


def _distribution(s: GameState, which: str, allowed: tuple) -> tuple:
    supplied = s.probability.species if which == "species" else s.probability.ranks
    if not supplied:
        return tuple((v, None) for v in allowed)
    weights = [(v, supplied.get(str(v), 0.0)) for v in allowed]
    total = sum(p for _, p in weights)
    if not total:
        raise ValueError("显式概率模型在当前允许范围内没有概率质量")
    return tuple((v, p / total) for v, p in weights if p > 0)


def _parameter_warnings(e: Effect) -> list[str]:
    p = e.params
    if e.id == "saw":
        chance = p.get("chance")
        if not isinstance(chance, (int, float)) or isinstance(chance, bool) or not isfinite(chance) or not 0 <= chance <= 1:
            return ["锈锯需要已核对的 chance 参数（0 到 1）；不从“随机”推定概率。"]
    if e.id == "muzzle":
        if p.get("timing") not in ("before", "after") or type(p.get("max_population")) is not int or not 0 <= p["max_population"] <= 15:
            return ["剧痛口枷需要确认人数阈值 max_population 与 before/after 判定时点。"]
    if e.id == "patrol_options":
        targets = p.get("targets")
        if not isinstance(targets, list) or not targets or any(type(i) is not int or i not in range(16) for i in targets) or len(set(targets)) != len(targets):
            return ["巡监目的地缺失或无效；必须读取画面给出的选项。"]
    if e.id == "nest" and (type(p.get("remaining")) is not int or p["remaining"] < 1):
        return ["鼠巢缺少剩余日切换次数 remaining。"]
    if e.id == "prey" and (type(p.get("remaining")) is not int or p["remaining"] < 1 or type(p.get("partner")) is not int or p["partner"] not in range(1, 16)):
        return ["猎物标记需要 partner 牢房编号与 remaining 次数。"]
    if e.id == "wine" and p.get("changes_species") is not False:
        return ["赤色佳酿目前只支持已确认不变代号的 changes_species=false 口径。"]
    return []


def _blocking_warnings(s: GameState) -> tuple[str, ...]:
    warnings: list[str] = []
    if s.rules_profile != "provisional_pc":
        warnings.append(f"不支持规则档案：{s.rules_profile}。")
    if s.unknown_fields:
        warnings.append("待补全局面：" + "、".join(s.unknown_fields) + "。")
    if s.position is None or s.rainbows is None:
        warnings.append("需要确认典狱长位置及已收容绮彩数量。")
    if s.pending_options:
        warnings.append("待处理奖励/随机结果：" + "、".join(e.name or e.id for e in s.pending_options) + "；读取实际结果后继续。")
    if sum(c.kind == "office" for c in s.cells) != 1 or _cell(s, 0).kind != "office":
        warnings.append("provisional_pc 要求 0 号为办公室、其余 15 格为牢房。")
    if len({e.id for e in s.effects}) != len(s.effects):
        warnings.append("重复全局效果的叠加规则未确认；请先核对。")
    for group, allowed in ((s.effects, SUPPORTED_TOOLS | SUPPORTED_EVENTS), (s.treasures, SUPPORTED_TREASURES)):
        for e in group:
            if e.id not in allowed:
                warnings.append(f"未实现效果：{e.name or e.id}；为避免漏算，暂停推演。")
            if not e.confirmed:
                warnings.append(f"效果尚未核对：{e.name or e.id}。")
            warnings.extend(_parameter_warnings(e))
    for c in s.cells:
        if c.kind in ("unknown", "event"):
            warnings.append(f"{c.index} 号牢房的内容或事变尚未解析。")
        if c.kind == "prisoner" and (c.species not in SPECIES or c.rank is None or c.rank == 5):
            warnings.append(f"{c.index} 号囚犯信息不完整，或绮彩尚未完成收容。")
        if c.kind in ("closed", "empty") and c.species is not None:
            warnings.append(f"{c.index} 号未开牢房含隐藏代号；请用实际可见信息建立局面。")
        for e in c.marks:
            if e.id not in SUPPORTED_MARKS:
                warnings.append(f"{c.index} 号未知标记：{e.name or e.id}；暂停推演。")
            if not e.confirmed:
                warnings.append(f"{c.index} 号标记尚未核对：{e.name or e.id}。")
            warnings.extend(_parameter_warnings(e))
            if e.id in ("scratch", "nest", "reveal") and c.kind != "closed":
                warnings.append(f"{c.index} 号的 {e.id} 标记要求牢房未开启。")
            if e.id in ("scratch", "reveal") and c.rank is None:
                warnings.append(f"{c.index} 号已探查/划痕标记缺少可见品质。")
            if e.id == "tusk" and c.kind != "prisoner":
                warnings.append(f"{c.index} 号象牙标记没有对应的场上囚犯。")
            if e.id == "prey" and not _parameter_warnings(e):
                partner = _cell(s, e.params["partner"])
                if c.kind != "prisoner" or c.species != "lion" or partner.kind != "prisoner" or partner.index == c.index:
                    warnings.append(f"{c.index} 号猎物标记关联的囚犯已失效。")
    if _effect(s, "rope") and s.probability.ranks and not sum(s.probability.ranks.get(str(i), 0) for i in range(2, 6)):
        warnings.append("绞首绳排除低阶后，概率模型无可用结果。")
    return tuple(dict.fromkeys(warnings))


def state_warnings(state: GameState) -> tuple[str, ...]:
    warnings = list(_blocking_warnings(state))
    if not state.rules_confirmed:
        warnings.append("provisional_pc 尚未用当前游戏核对：办公室布局、所在格操作、吃换互斥、连续吞噬、空格重封与绮彩上限 10；建议仅为条件结果。")
    if not state.probability.species:
        warnings.append("代号随机概率未知；不会默认四种代号等概率。")
    if not state.probability.ranks:
        warnings.append("开门/换人品质概率未知；无法给出可信收益概率。")
    elif state.probability.estimated:
        warnings.append("概率模型为估计值，且当前接口共用开门与换人品质分布；需确认它适用于当前动作。")
    return tuple(dict.fromkeys(warnings))


def is_terminal(state: GameState) -> bool:
    return state.day > state.max_days and not state.pending_options and not state.unknown_fields


def _can_eat(s: GameState, a: Cell, b: Cell) -> bool:
    if a.index == b.index or a.kind != "prisoner" or a.species not in SPECIES or a.rank is None:
        return False
    if b.kind == "closed":
        return a.species == "wolf" and b.rank == a.rank and bool(_mark(b, "scratch"))
    if b.kind != "prisoner" or b.rank is None:
        return False
    rank_ok = b.rank <= a.rank if _effect(s, "cagefight") else b.rank == a.rank
    base_match = rank_ok and b.species in PREY[a.species]
    rat_match = _effect(s, "rathead") and a.species == b.species == "rat" and b.rank == a.rank
    return bool(base_match or rat_match)


def legal_actions(state: GameState) -> tuple[Action, ...]:
    validate_state(state)
    if is_terminal(state) or _blocking_warnings(state):
        return ()
    routes = _effect(state, "patrol_options")
    if routes:
        return tuple(Action("patrol", source=state.position, target=i, label=f"巡监至 {i} 号" if i else "巡监至办公室") for i in routes.params["targets"])
    actions = []
    actor = _cell(state, state.position)
    if actor.kind == "prisoner":
        if state.daily_action in ("none", "eat"):
            actions.extend(Action("eat", source=actor.index, target=c.index, label=f"{actor.index} 号吃 {c.index} 号") for c in state.cells if _can_eat(state, actor, c))
        if state.daily_action in ("none", "replace") and state.replacements_left > 0:
            actions.append(Action("replace", source=actor.index, label=f"换 {actor.index} 号囚犯"))
    unique_treasures = {e.id: e for e in state.treasures}
    actions.extend(Action("treasure", effect_id=e.id, label=f"使用 {e.name or e.id}") for e in unique_treasures.values())
    actions.append(Action("end", label="结束今天"))
    return tuple(actions)


def _settle(s: GameState) -> GameState:
    """Collect in stable board order; stop on reward whose options are not visible."""
    count = s.rainbows or 0
    for c in tuple(s.cells):
        if c.kind != "prisoner" or c.rank is None or c.rank < 5:
            continue
        if count >= 10:
            s = _put(s, replace(c, rank=4))
            continue
        count += 1
        s = _put(s, Cell(c.index, "closed"))
        s = _pending(s, "treasure_choice", "绮彩珍藏选项未知", source_cell=c.index)
    return replace(s, rainbows=count)


def _invalidate_prey(before: GameState, after: GameState, replaced: int | None = None) -> GameState:
    for c in after.cells:
        marks = []
        for mark in c.marks:
            if mark.id != "prey":
                marks.append(mark)
                continue
            partner = mark.params["partner"]
            a0, b0, a1, b1 = _cell(before, c.index), _cell(before, partner), c, _cell(after, partner)
            if replaced in (c.index, partner) or a1.kind != "prisoner" or b1.kind != "prisoner" or a0.species != a1.species or b0.species != b1.species:
                continue
            marks.append(mark)
        if tuple(marks) != c.marks:
            after = _put(after, replace(c, marks=tuple(marks)))
    return after


def _new_inmates(s: GameState, cell: Cell, fixed_species: str | None = None, min_rank: int = 0) -> tuple[tuple[Cell, float | None], ...]:
    species = ((fixed_species, 1.0),) if fixed_species else _distribution(s, "species", SPECIES)
    ranks = ((cell.rank, 1.0),) if cell.kind == "closed" and cell.rank is not None else _distribution(s, "ranks", tuple(range(min_rank, 6)))
    return tuple((Cell(cell.index, "prisoner", code, rank, cell.marks if cell.kind == "prisoner" else ()), _probability(pc, pr)) for (code, pc), (rank, pr) in product(species, ranks))


def _eat(s: GameState, action: Action) -> tuple[Outcome, ...]:
    eater, prey = _cell(s, action.source), _cell(s, action.target)
    gain = 3 if prey.kind == "closed" else 1
    if _effect(s, "rathead") and eater.species == "rat":
        gain += 1
    muzzle = _effect(s, "muzzle")
    if muzzle:
        count = sum(c.kind == "prisoner" for c in s.cells)
        if muzzle.params["timing"] == "after" and prey.kind == "prisoner":
            count -= 1
        if count <= muzzle.params["max_population"]:
            gain += 2
    gains = ((gain, 1.0),)
    if saw := _effect(s, "saw"):
        p = saw.params["chance"]
        gains = tuple((g, q) for g, q in ((gain, 1 - p), (gain + 1, p)) if q > 0)
    codes = ((eater.species, 1.0),) if _effect(s, "cagefight") else _distribution(s, "species", SPECIES)
    outcomes = []
    for (amount, pg), (species, ps) in product(gains, codes):
        current = replace(eater, species=species, rank=min(5, eater.rank + amount))
        after = _put(replace(s, daily_action="eat"), current, Cell(prey.index, "closed"))
        after = _invalidate_prey(s, _settle(after))
        outcomes.append(Outcome(after, _probability(pg, ps), "吞噬升阶后重新核对代号和连锁目标；绮彩珍藏须读取实际选项。"))
    return tuple(outcomes)


def _replace_inmate(s: GameState, action: Action) -> tuple[Outcome, ...]:
    actor = _cell(s, action.source)
    outcomes = []
    for current, probability in _new_inmates(s, actor, actor.species if _effect(s, "pestle") else None, 2 if _effect(s, "rope") else 0):
        after = _put(replace(s, daily_action="replace", replacements_left=s.replacements_left - 1), current)
        if _mark(actor, "tusk"):
            # TLIDB gives four random recipients but not a uniform distribution
            # or replacement inclusion policy. Record this pending explicitly.
            after = _pending(after, "tusk_resolution", "象牙随机升阶对象待读取", count=4, source_cell=actor.index)
        after = _invalidate_prey(s, _settle(after), replaced=actor.index)
        outcomes.append(Outcome(after, probability, "换人后本日不能再发起吞噬。"))
    return tuple(outcomes)


def _end(s: GameState) -> tuple[Outcome, ...]:
    if s.day == s.max_days:
        return (Outcome(replace(s, day=s.day + 1), 1.0, "典狱天数结束；目标计数为已经收容的绮彩。"),)
    after = replace(s, day=s.day + 1, daily_action="none", replacements_left=1)
    promotions: dict[int, int] = {}
    for c in s.cells:
        marks = []
        for m in c.marks:
            if m.id == "nest":
                remaining = m.params["remaining"] - 1
                if remaining == 0:
                    after = _put(after, Cell(c.index, "prisoner", "rat", 2))
                    continue
                marks.append(replace(m, params={**m.params, "remaining": remaining}))
            elif m.id == "prey":
                promotions[c.index] = promotions.get(c.index, 0) + 1
                partner = m.params["partner"]
                promotions[partner] = promotions.get(partner, 0) + 1
                if m.params["remaining"] > 1:
                    marks.append(replace(m, params={**m.params, "remaining": m.params["remaining"] - 1}))
            else:
                marks.append(m)
        current = _cell(after, c.index)
        if current.kind == c.kind:
            after = _put(after, replace(current, marks=tuple(marks)))
    for index, gain in promotions.items():
        c = _cell(after, index)
        after = _put(after, replace(c, rank=min(5, c.rank + gain)))
    after = _invalidate_prey(s, _settle(after))
    after = replace(after, unknown_fields=tuple(dict.fromkeys(after.unknown_fields + ("patrol_options",))))
    return (Outcome(after, 1.0, "日开始标记已结算；下一日巡监选项尚未出现，暂停预测路线。"),)


def _patrol(s: GameState, action: Action) -> tuple[Outcome, ...]:
    target = _cell(s, action.target)
    after = replace(s, position=target.index, effects=tuple(e for e in s.effects if e.id != "patrol_options"))
    if _effect(s, "chain"):
        # Exact route known, but multiple unseen occupants have a joint outcome
        # far larger than one screen action; wait for that visible result.
        after = replace(after, effects=tuple(e for e in after.effects if e.id != "chain"))
        after = _pending(after, "chain_open", "连锁沿途开门结果待读取", source=s.position, target=target.index)
        return (Outcome(after, None, "连锁会改变沿途未开牢房，不把这些变化静默忽略。"),)
    if target.kind == "office":
        after = _pending(after, "office_choice", "办公室煽动/赏赐选项待读取")
        return (Outcome(after, 1.0, "到达办公室，需确认实际选项及作用对象。"),)
    if target.kind == "closed":
        return tuple(Outcome(_settle(_put(after, new)), p, "开门结果未知，按显式模型或未知概率分支列出。") for new, p in _new_inmates(s, target))
    return (Outcome(after, 1.0, "到达已知牢房，进入本日经营。"),)


def transition(state: GameState, action: Action) -> tuple[Outcome, ...]:
    """Return all represented next states; None odds are not equiprobable."""
    if action.key not in {a.key for a in legal_actions(state)}:
        raise ValueError("当前局面不允许该操作，或仍存在未确认的效果/随机结果")
    if action.kind == "eat":
        outcomes = _eat(state, action)
    elif action.kind == "replace":
        outcomes = _replace_inmate(state, action)
    elif action.kind == "patrol":
        outcomes = _patrol(state, action)
    elif action.kind == "end":
        outcomes = _end(state)
    elif action.kind == "treasure":
        treasures = list(state.treasures)
        index = next(i for i, e in enumerate(treasures) if e.id == action.effect_id)
        treasures.pop(index)
        after = replace(state, treasures=tuple(treasures))
        for c in state.cells:
            if c.kind == "prisoner" and c.rank < 3:
                after = _put(after, replace(c, rank=c.rank + 1))
        outcomes = (Outcome(_settle(after), 1.0, "赤色佳酿使低于传奇的场上囚犯升一阶。"),)
    else:
        raise ValueError("未支持的操作")
    # Combining indistinguishable states is safe only if every branch's mass is
    # known. Keeping unknown branches separate must not manufacture probabilities.
    merged: dict[str, Outcome] = {}
    for o in outcomes:
        updated = replace(o.state, revision=state.revision + 1)
        validate_state(updated)
        key = updated.fingerprint()
        existing = merged.get(key)
        probability = o.probability
        if existing:
            probability = None if probability is None or existing.probability is None else probability + existing.probability
        merged[key] = Outcome(updated, probability, o.explanation)
    return tuple(merged.values())
