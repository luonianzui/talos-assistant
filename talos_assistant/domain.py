from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import math
from typing import Any

SPECIES = ("elephant", "lion", "wolf", "rat")
SPECIES_NAMES = {"elephant": "巨象", "lion": "狂狮", "wolf": "头狼", "rat": "硕鼠"}
RANK_NAMES = ("普通", "魔法", "稀有", "传奇", "至臻", "绮彩")
RANK_COLORS = ("#bec9d5", "#57a6f8", "#bd7cf5", "#efa852", "#ef6574", "#69e5cd")


@dataclass(frozen=True)
class Effect:
    id: str
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False


@dataclass(frozen=True)
class Cell:
    index: int
    kind: str = "closed"  # office, closed, empty, prisoner, event, unknown
    species: str | None = None
    rank: int | None = None
    marks: tuple[Effect, ...] = ()


@dataclass(frozen=True)
class ProbabilityModel:
    """Explicit model, never silently inferred from the word 'random'."""
    species: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, float] = field(default_factory=dict)
    source: str = ""
    sample_size: int = 0
    estimated: bool = True


@dataclass(frozen=True)
class GameState:
    cells: tuple[Cell, ...] = field(default_factory=lambda: tuple(Cell(i, "office" if i == 0 else "closed") for i in range(16)))
    day: int = 1
    max_days: int = 20
    position: int | None = 0
    rainbows: int | None = 0
    daily_action: str = "none"
    replacements_left: int = 1
    effects: tuple[Effect, ...] = ()
    treasures: tuple[Effect, ...] = ()
    pending_options: tuple[Effect, ...] = ()
    probability: ProbabilityModel = field(default_factory=ProbabilityModel)
    rules_profile: str = "provisional_pc"
    rules_confirmed: bool = False
    unknown_fields: tuple[str, ...] = ()
    source: str = "manual"
    revision: int = 0

    def fingerprint(self) -> str:
        data = asdict(self)
        data.pop("revision", None)
        return sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Action:
    kind: str
    source: int | None = None
    target: int | None = None
    effect_id: str | None = None
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.source}:{self.target}:{self.effect_id}"


@dataclass(frozen=True)
class Outcome:
    state: GameState
    probability: float | None
    explanation: str = ""


@dataclass(frozen=True)
class StrategyConfig:
    mode: str = "ceiling"  # average / ceiling
    threshold: float = 0.2
    quick_seconds: float = 2.0
    total_seconds: float = 10.0
    seed: int = 42
    max_depth: int = 60

    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ActionEvaluation:
    action: Action
    mean: float | None = None
    ceiling: int | None = None
    hit_probability: float | None = None
    distribution: dict[int, float] = field(default_factory=dict)
    samples: int = 0
    steps: float = 0.0
    reason: str = ""
    conditions: tuple[str, ...] = ()
    exact: bool = False


@dataclass(frozen=True)
class Recommendation:
    state_id: str
    strategy_id: str
    mode: str
    candidates: tuple[ActionEvaluation, ...] = ()
    status: str = "conditional"  # exact, estimated, conditional, blocked, cancelled
    elapsed: float = 0.0
    warnings: tuple[str, ...] = ()
    complete: bool = False


def state_to_dict(state: GameState) -> dict[str, Any]:
    return asdict(state)


def _effect(data: Any) -> Effect:
    _object(data, "效果", {f.name for f in fields(Effect)})
    if "id" not in data:
        raise ValueError("效果必须包含字符串 id")
    effect = Effect(**data)
    _validate_effect(effect)
    return effect


def _object(data: Any, name: str, allowed: set[str]) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name}必须是 JSON 对象")
    if any(not isinstance(key, str) for key in data):
        raise ValueError(f"{name}的字段名称必须是字符串")
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"{name}包含不支持的字段：" + "、".join(sorted(extra)))


def _array(data: Any, name: str) -> None:
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"{name}必须是数组")


def _json_value(value: Any, depth: int = 0) -> None:
    if depth > 20:
        raise ValueError("效果参数嵌套超过 20 层")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("效果参数不能包含非有限数值")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("效果参数的字段名称必须为字符串")
        for item in value.values():
            _json_value(item, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_value(item, depth + 1)
        return
    raise ValueError("效果参数必须使用 JSON 数据类型")


def _validate_effect(effect: Effect) -> None:
    if not isinstance(effect, Effect):
        raise ValueError("效果与标记必须为 Effect")
    if not isinstance(effect.id, str) or not effect.id.strip() or not isinstance(effect.name, str):
        raise ValueError("效果 id 必须是非空字符串，名称必须是字符串")
    if type(effect.confirmed) is not bool:
        raise ValueError("效果 confirmed 必须是 true 或 false，不能使用字符串或数字")
    if not isinstance(effect.params, dict):
        raise ValueError("效果 params 必须是 JSON 对象")
    _json_value(effect.params)


def state_from_dict(data: Any) -> GameState:
    allowed = {f.name for f in fields(GameState)}
    _object(data, "局面", allowed)
    kwargs = dict(data)
    if "cells" in data:
        _array(data["cells"], "棋盘")
        cells = []
        for c in data["cells"]:
            _object(c, "牢房", {f.name for f in fields(Cell)})
            if "index" not in c:
                raise ValueError("牢房缺少 index 编号")
            _array(c.get("marks", []), "牢房标记")
            cells.append(Cell(index=c["index"], kind=c.get("kind", "closed"), species=c.get("species"), rank=c.get("rank"), marks=tuple(_effect(e) for e in c.get("marks", []))))
        kwargs["cells"] = tuple(cells)
    for key in ("effects", "treasures", "pending_options"):
        _array(data.get(key, []), key)
        kwargs[key] = tuple(_effect(e) for e in data.get(key, []))
    _array(data.get("unknown_fields", []), "unknown_fields")
    kwargs["unknown_fields"] = tuple(data.get("unknown_fields", []))
    prob = data.get("probability", {})
    _object(prob, "概率模型", {f.name for f in fields(ProbabilityModel)})
    kwargs["probability"] = ProbabilityModel(**prob)
    state = GameState(**kwargs)
    validate_state(state)
    return state


def validate_state(state: GameState) -> None:
    if not isinstance(state, GameState):
        raise ValueError("局面必须为 GameState")
    if type(state.day) is not int or type(state.max_days) is not int or not (1 <= state.max_days <= 100 and 1 <= state.day <= state.max_days + 1):
        raise ValueError("天数必须在有效范围内")
    _array(state.cells, "棋盘")
    if len(state.cells) != 16 or any(not isinstance(c, Cell) or type(c.index) is not int for c in state.cells) or sorted(c.index for c in state.cells) != list(range(16)):
        raise ValueError("棋盘必须包含编号 0–15 的 16 个不同牢房")
    if state.position is not None and (type(state.position) is not int or state.position not in range(16)):
        raise ValueError("典狱长位置无效")
    if state.rainbows is not None and (type(state.rainbows) is not int or state.rainbows < 0):
        raise ValueError("绮彩数量不能为负")
    if type(state.replacements_left) is not int or state.replacements_left < 0 or state.daily_action not in ("none", "eat", "replace"):
        raise ValueError("本日操作状态无效")
    if type(state.rules_confirmed) is not bool:
        raise ValueError("rules_confirmed 必须是 true 或 false，不能使用字符串或数字")
    if type(state.revision) is not int or state.revision < 0:
        raise ValueError("revision 必须是非负整数")
    if not isinstance(state.source, str) or not isinstance(state.rules_profile, str) or not state.rules_profile.strip():
        raise ValueError("来源与规则档案必须是字符串，规则档案不能为空")
    _array(state.unknown_fields, "unknown_fields")
    if any(not isinstance(value, str) or not value.strip() for value in state.unknown_fields):
        raise ValueError("unknown_fields 必须包含非空字段名称")
    for key in ("effects", "treasures", "pending_options"):
        group = getattr(state, key)
        _array(group, key)
        for effect in group:
            _validate_effect(effect)
    for cell in state.cells:
        if cell.kind not in ("office", "closed", "empty", "prisoner", "event", "unknown"):
            raise ValueError("牢房状态无效")
        if cell.species is not None and cell.species not in SPECIES:
            raise ValueError("未知囚犯代号")
        if cell.rank is not None and (type(cell.rank) is not int or cell.rank not in range(6)):
            raise ValueError("品质必须为 0–5")
        _array(cell.marks, "牢房标记")
        for effect in cell.marks:
            _validate_effect(effect)
    if not isinstance(state.probability, ProbabilityModel):
        raise ValueError("概率模型必须为 ProbabilityModel")
    if type(state.probability.estimated) is not bool:
        raise ValueError("概率模型 estimated 必须是 true 或 false")
    if type(state.probability.sample_size) is not int or state.probability.sample_size < 0:
        raise ValueError("概率模型 sample_size 必须是非负整数")
    if not isinstance(state.probability.source, str):
        raise ValueError("概率模型 source 必须是字符串")
    for probs in (state.probability.species, state.probability.ranks):
        if not isinstance(probs, dict) or any(not isinstance(key, str) for key in probs):
            raise ValueError("概率表必须是字符串键的 JSON 对象")
        if probs:
            if any(type(p) not in (int, float) or not 0 <= p <= 1 or not math.isfinite(p) for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
                raise ValueError("概率必须非负且总和为 1")
    if set(state.probability.species) - set(SPECIES):
        raise ValueError("概率表包含未知代号")
    if set(state.probability.ranks) - {str(i) for i in range(6)}:
        raise ValueError("概率表包含未知品质")
    if (state.probability.species or state.probability.ranks) and not state.probability.source.strip():
        raise ValueError("概率模型必须注明来源")
