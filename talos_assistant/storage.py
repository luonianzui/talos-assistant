from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from .domain import GameState, StrategyConfig, state_from_dict, state_to_dict

SCHEMA = "talos.assistant.session.v1"


def resource_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def user_data_dir() -> Path:
    override = os.environ.get("TALOS_DATA_DIR")
    path = Path(override) if override else ((Path(sys.executable).parent if getattr(sys, "frozen", False) else resource_dir()) / "user_data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    if path.stat().st_size > 10_000_000:
        raise ValueError("文件超过 10 MB，请导入单局记录")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("不允许非有限数值")))


class Settings:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = read_json(path)
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}

    def strategy(self) -> StrategyConfig:
        mode = self.data.get("strategy_mode", "ceiling")
        if mode not in ("average", "ceiling"):
            mode = "ceiling"
        try:
            threshold = max(0.01, min(0.99, float(self.data.get("threshold", .2))))
        except (ValueError, TypeError):
            threshold = .2
        return StrategyConfig(mode=mode, threshold=threshold)

    def save_strategy(self, config: StrategyConfig) -> None:
        self.data.update(strategy_mode=config.mode, threshold=config.threshold)
        write_json(self.path, self.data)


class Session:
    """An observation ledger. Never infer accumulated rewards from repeated frames."""
    def __init__(self):
        self.entries: list[dict[str, Any]] = []

    def record(self, state: GameState, label: str) -> bool:
        if self.entries and state_from_dict(self.entries[-1]["state"]).fingerprint() == state.fingerprint():
            return False
        self.entries.append({"time": datetime.now(timezone.utc).isoformat(), "label": label, "state": state_to_dict(state)})
        return True

    def save(self, path: Path, config: StrategyConfig) -> None:
        write_json(path, {"schema": SCHEMA, "version": 1, "strategy": asdict(config), "entries": self.entries})

    @classmethod
    def load(cls, path: Path) -> "Session":
        data = read_json(path)
        session = cls()
        if isinstance(data, dict) and data.get("schema") == SCHEMA:
            if type(data.get("version")) is not int or data["version"] != 1:
                raise ValueError("不支持该回放版本")
            entries = data.get("entries")
            if not isinstance(entries, list) or not 1 <= len(entries) <= 5000:
                raise ValueError("回放需包含 1–5000 个有效局面")
            for item in entries:
                state = state_from_dict(item["state"])
                session.entries.append({"time": str(item.get("time", "")), "label": str(item.get("label", "导入")), "state": state_to_dict(state)})
        elif isinstance(data, dict) and "cells" in data:
            session.record(state_from_dict(data), "导入局面")
        else:
            raise ValueError("不支持此格式。请使用本工具的局面或回放 JSON；网页模拟器的规则与格式不同，不能直接混用。")
        return session

    def state(self, index: int = -1) -> GameState:
        return state_from_dict(self.entries[index]["state"])


class ResultGate:
    """Reject old results after an observation, mode change, or cancellation."""
    def __init__(self):
        self.generation = 0
        self.state_id = ""
        self.strategy_id = ""

    def begin(self, state: GameState, config: StrategyConfig) -> int:
        self.generation += 1
        self.state_id = state.fingerprint()
        self.strategy_id = config.fingerprint()
        return self.generation

    def invalidate(self) -> None:
        self.generation += 1

    def accepts(self, generation: int, recommendation) -> bool:
        return generation == self.generation and recommendation.state_id == self.state_id and recommendation.strategy_id == self.strategy_id
