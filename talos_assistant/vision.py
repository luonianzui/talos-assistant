"""Calibrated local vision. No built-in game templates and no automatic downloads.

Recognition is an observation, never an event: absolute counters replace their
previous values, and unobserved fields remain explicitly unverified.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np

from .domain import Cell, Effect, GameState, SPECIES

Rect = tuple[float, float, float, float]  # normalized x, y, width, height
RING6X4 = tuple([(x, 0) for x in range(6)] + [(5, y) for y in (1, 2, 3)] +
               [(x, 3) for x in (4, 3, 2, 1, 0)] + [(0, y) for y in (2, 1)])
SPECIES_WORDS = {"elephant": ("巨象", "猛犸", "猛獁"), "lion": ("狂狮", "狂獅", "狮子", "獅子"),
                 "wolf": ("头狼", "頭狼", "狼"), "rat": ("硕鼠", "碩鼠", "巨鼠", "老鼠")}
RANK_WORDS = (("普通",), ("魔法",), ("稀有",), ("传奇", "傳奇"), ("至臻", "至尊"), ("绮彩", "綺彩", "虹彩", "彩色"))


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RecognitionResult:
    state: GameState
    issues: tuple[str, ...]
    confidences: dict[str, float]


def _checked_rect(rect) -> Rect:
    if len(rect) != 4:
        raise ValueError("区域须为归一化 (x, y, width, height)")
    x, y, w, h = map(float, rect)
    if not all(np.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.00001 or y + h > 1.00001:
        raise ValueError("区域必须完全位于截图内，使用 0–1 的归一化坐标。")
    return x, y, w, h


def crop(image: np.ndarray, rect: Rect) -> np.ndarray:
    x, y, w, h = _checked_rect(rect)
    height, width = image.shape[:2]
    x0, y0 = max(0, round(x * width)), max(0, round(y * height))
    x1, y1 = min(width, round((x + w) * width)), min(height, round((y + h) * height))
    result = image[y0:y1, x0:x1]
    if not result.size:
        raise ValueError("所选区域过小。")
    return result


def _signature(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (192, 128), interpolation=cv2.INTER_AREA).astype(np.float32)


class StableFrameGate:
    """Require consecutive similar frames and emit a stable screen only once."""
    def __init__(self, required_frames: int = 2, tolerance: float = 1.5):
        if required_frames < 2:
            raise ValueError("稳定识别至少需要两帧。")
        self.required_frames = required_frames
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self._candidate = None
        self._accepted = None
        self._shape = None
        self._count = 0

    def _similar(self, a: np.ndarray, b: np.ndarray) -> bool:
        diff = np.abs(a - b)
        # The mean alone misses a changed one-digit counter on a large screen.
        # Local differences must be small as well, without requiring byte-for-
        # byte identity through ordinary compression/sampling noise.
        return float(diff.mean()) <= self.tolerance and float(diff.mean(axis=2).max()) <= max(8.0, self.tolerance * 6)

    def accept(self, image: np.ndarray) -> bool:
        if image is None or not image.size:
            return False
        sig = _signature(image)
        shape = image.shape
        same = self._shape == shape and self._candidate is not None and self._similar(sig, self._candidate)
        self._count = self._count + 1 if same else 1
        # Anchor the candidate to the first frame, so slow animation drift is
        # not accepted just because adjacent animation frames look similar.
        if not same:
            self._candidate, self._shape = sig, shape
        if self._count < self.required_frames:
            return False
        if self._accepted is not None and self._accepted[0] == shape and self._similar(sig, self._accepted[1]):
            return False
        self._accepted = shape, sig
        return True


class LocalOCR:
    """Use only explicitly supplied ONNX models; missing files never download."""
    def __init__(self, models_dir: str | Path | None = None):
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        self.models_dir = Path(models_dir) if models_dir is not None else base / "models"
        self.engine = None
        self.error = ""
        self._attempted = False
        self._cache: dict[str, tuple[OCRLine, ...]] = {}

    def _initialize(self) -> bool:
        if self._attempted:
            return self.engine is not None
        self._attempted = True
        names = {"det": "det.onnx", "rec": "rec.onnx", "cls": "cls.onnx", "keys": "keys.txt"}
        manifest = self.models_dir / "models.json"
        try:
            if manifest.is_file():
                override = json.loads(manifest.read_text(encoding="utf-8"))
                names.update({k: override[k] for k in names if k in override})
            paths = {k: (self.models_dir / v).resolve() for k, v in names.items()}
            missing = [p.name for p in paths.values() if not p.is_file() or p.stat().st_size == 0]
            if missing:
                self.error = "离线 OCR 未就绪，缺少本地模型/字典：" + "、".join(missing)
                return False
            # RapidOCR 3.x can resolve a visualization font even when no
            # visualization is requested. Always supply an existing local font
            # to prevent that otherwise surprising background download.
            windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            font_candidates = [self.models_dir / "font.ttf", self.models_dir / "font.ttc",
                               windows_fonts / "msyh.ttc", windows_fonts / "simhei.ttf", windows_fonts / "segoeui.ttf",
                               Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
            font_path = next((p for p in font_candidates if p.is_file()), None)
            if font_path is None:
                self.error = "缺少本地字体，请在模型目录放入 font.ttf；不会自动下载字体。"
                return False
            from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

            self.engine = RapidOCR(params={
                "Det.model_path": str(paths["det"]), "Rec.model_path": str(paths["rec"]),
                "Cls.model_path": str(paths["cls"]), "Rec.rec_keys_path": str(paths["keys"]),
                "Global.use_cls": False, "Global.log_level": "error", "Global.font_path": str(font_path.resolve()),
                # The default "min" resizes a short counter crop to 736 px
                # high, wasting CPU on millions of interpolated pixels.
                "Det.limit_type": "max", "Det.limit_side_len": 960,
                "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.CH, "Cls.lang_type": LangCls.CH,
                "Det.ocr_version": OCRVersion.PPOCRV4, "Rec.ocr_version": OCRVersion.PPOCRV4, "Cls.ocr_version": OCRVersion.PPOCRV4,
                "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE,
                "Det.engine_type": EngineType.ONNXRUNTIME, "Rec.engine_type": EngineType.ONNXRUNTIME, "Cls.engine_type": EngineType.ONNXRUNTIME,
                "EngineConfig.onnxruntime.use_cuda": False, "EngineConfig.onnxruntime.use_dml": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": 2, "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
            return True
        except Exception as exc:
            self.error = f"本地 OCR 初始化失败：{exc}"
            self.engine = None
            return False

    def read(self, image: np.ndarray) -> tuple[OCRLine, ...]:
        if not self._initialize():
            return ()
        key = sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        try:
            output = self.engine(image)
            texts, scores, boxes = getattr(output, "txts", None), getattr(output, "scores", None), getattr(output, "boxes", None)
            lines = []
            if texts is not None and scores is not None:
                for i, (text, score) in enumerate(zip(texts, scores)):
                    box = None
                    if boxes is not None and i < len(boxes):
                        points = np.asarray(boxes[i])
                        box = float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())
                    lines.append(OCRLine(str(text).strip(), float(score), box))
            result = tuple(lines)
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
            return result
        except Exception as exc:
            self.error = f"本地 OCR 运行失败：{exc}"
            return ()

    def warmup(self) -> bool:
        """Load CPU models on a background worker before the first game frame."""
        sample = np.full((96, 320, 3), 255, dtype=np.uint8)
        cv2.putText(sample, "TEST 123", (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        self.read(sample)
        return self.engine is not None and not self.error


def parse_cell_text(text: str) -> tuple[str | None, int | None]:
    species = [s for s, words in SPECIES_WORDS.items() if any(w in text for w in words)]
    ranks = [i for i, words in enumerate(RANK_WORDS) if any(w in text for w in words)]
    return species[0] if len(species) == 1 else None, ranks[0] if len(ranks) == 1 else None


def _feature(image: np.ndarray) -> np.ndarray:
    # A local exemplar represents the whole cell, not a guessed species icon.
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Require both spatial texture and color. Correlation alone calls flat or
    # differently colored rank frames perfect matches.
    af, bf = a.astype(np.float32), b.astype(np.float32)
    color = max(0.0, 1.0 - float(np.mean(np.abs(af - bf))) / 80.0)
    ac, bc = af - af.mean(), bf - bf.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    corr = float(np.sum(ac * bc) / denom) if denom > 1e-5 else (1.0 if np.max(np.abs(af - bf)) < 2 else 0.0)
    return float(np.clip(0.6 * max(0.0, corr) + 0.4 * color, 0, 1))


class Recognizer:
    def __init__(self, profile_path: str | Path, models_dir: str | Path | None = None):
        self.profile_path = Path(profile_path)
        self.profile = {"version": 1, "layout": "ring6x4", "position_base": 0, "board_rect": None, "cell_regions": {}, "regions": {}, "templates": [], "template_threshold": 0.96}
        if self.profile_path.is_file():
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("不支持该识别配置版本。")
            self.profile.update(data)
        self.ocr = LocalOCR(models_dir)
        self._templates: list[tuple[dict, np.ndarray]] = []
        for entry in self.profile["templates"]:
            image = cv2.imdecode(np.frombuffer(base64.b64decode(entry["png"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("识别模板图像损坏，请重新学习。")
            self._templates.append((entry, _feature(image)))

    def _save(self):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_path.with_name(self.profile_path.name + ".tmp")
        temporary.write_text(json.dumps(self.profile, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.profile_path)

    def set_board_rect(self, rect: Rect):
        self.profile["board_rect"] = list(_checked_rect(rect))
        self._save()

    def set_board_layout(self, layout: str):
        if layout not in ("ring6x4", "grid4x4"):
            raise ValueError("布局必须为 ring6x4 或 grid4x4")
        self.profile["layout"] = layout
        self._save()

    def set_region(self, name: str, rect: Rect):
        if name not in ("day", "days_remaining", "rainbows", "position", "effects", "treasures", "daily_action", "replacements_left", "pending_options"):
            raise ValueError("不支持该 OCR 区域名称。")
        self.profile["regions"][name] = list(_checked_rect(rect))
        self._save()

    def set_position_base(self, base: int):
        if type(base) is not int or base not in (0, 1):
            raise ValueError("位置编号起点必须明确为 0 或 1。")
        self.profile["position_base"] = base
        self._save()

    def set_cell_rect(self, index: int, rect: Rect):
        """Optional exact cell calibration when the board is not a regular grid."""
        if index not in range(16):
            raise ValueError("格子编号须为 0–15。")
        self.profile["cell_regions"][str(index)] = list(_checked_rect(rect))
        self._save()

    def board_cell_rect(self, index: int) -> Rect:
        if index not in range(16):
            raise ValueError("格子编号须为 0–15。")
        override = self.profile["cell_regions"].get(str(index))
        if override is not None:
            return _checked_rect(override)
        if not self.profile.get("board_rect"):
            raise ValueError("请先框选完整棋盘区域。")
        x, y, w, h = _checked_rect(self.profile["board_rect"])
        if self.profile["layout"] == "ring6x4":
            col, row = RING6X4[index]
            columns = 6
        else:
            col, row = index % 4, index // 4
            columns = 4
        return x + col * w / columns, y + row * h / 4, w / columns, h / 4

    def stability_view(self, image: np.ndarray) -> np.ndarray:
        """Compact calibrated observations for StableFrameGate, not animation.

        Tile the 16 cell patches (excluding the ring's courtyard/background),
        then append the configured text regions. With no calibration the caller
        gets the original frame and should prompt for region selection.
        """
        strips = []
        if self.profile.get("board_rect"):
            patches = [cv2.resize(crop(image, self.board_cell_rect(i)), (64, 64), interpolation=cv2.INTER_AREA) for i in range(16)]
            for start in range(0, 16, 4):
                strips.append(np.concatenate(patches[start:start + 4], axis=1))
        for name, rect in sorted(self.profile["regions"].items()):
            strips.append(cv2.resize(crop(image, tuple(rect)), (256, 48), interpolation=cv2.INTER_AREA))
        return np.concatenate(strips, axis=0) if strips else image

    def learn_cell(self, image: np.ndarray, index: int, species: str | None = None,
                   rank: int | None = None, kind: str = "prisoner"):
        if kind not in ("office", "closed", "empty", "prisoner", "event"):
            raise ValueError("不能学习未知状态，请明确格子类型。")
        if kind == "prisoner" and (species not in SPECIES or rank not in range(6)):
            raise ValueError("学习囚犯模板需要明确物种与品质。")
        patch = _feature(crop(image, self.board_cell_rect(index)))
        ok, encoded = cv2.imencode(".png", patch)
        if not ok:
            raise ValueError("模板编码失败。")
        entry = {"kind": kind, "species": species if kind == "prisoner" else None,
                 "rank": rank if kind == "prisoner" else None, "png": base64.b64encode(encoded).decode("ascii")}
        if entry not in self.profile["templates"]:
            self.profile["templates"].append(entry)
            self._templates.append((entry, patch))
            self._save()

    def _match_cell(self, image: np.ndarray, index: int) -> tuple[Cell | None, float]:
        sample = _feature(crop(image, self.board_cell_rect(index)))
        groups: dict[tuple, float] = {}
        for entry, template in self._templates:
            label = entry["kind"], entry.get("species"), entry.get("rank")
            groups[label] = max(groups.get(label, 0), _similarity(sample, template))
        ranked = sorted(groups.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, 0.0
        label, score = ranked[0]
        # A tie between labels is unresolved even at a perfect pixel match.
        if score < float(self.profile.get("template_threshold", 0.96)) or (len(ranked) > 1 and score - ranked[1][1] < 0.025):
            return None, score
        return Cell(index, *label), score

    def _text(self, image: np.ndarray) -> tuple[str, float]:
        lines = self.ocr.read(image)
        if not lines:
            return "", 0.0
        return " ".join(line.text for line in lines), min(line.confidence for line in lines)

    def recognize(self, image: np.ndarray, previous: GameState) -> RecognitionResult:
        if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("识别需要非空 BGR 三通道截图。")
        issues: list[str] = []
        unknown = set(previous.unknown_fields)
        confidences: dict[str, float] = {}
        cells: list[Cell] = []
        if self.profile.get("board_rect"):
            unknown.discard("board")
            unmatched = []
            for i in range(16):
                cell, score = self._match_cell(image, i)
                confidences[f"cells.{i}"] = score
                cells.append(cell or Cell(i, "unknown"))
                if cell is None:
                    unmatched.append(i)
            # A single board OCR pass supplements templates only where both a
            # species name and a rank are actually visible in that cell.
            if unmatched:
                board = crop(image, tuple(self.profile["board_rect"]))
                bx, by, bw, bh = self.profile["board_rect"]
                lines = self.ocr.read(board)
                for i in unmatched:
                    cx, cy, cw, ch = self.board_cell_rect(i)
                    selected = []
                    for line in lines:
                        if line.box is None or line.confidence < 0.88:
                            continue
                        lx = bx + ((line.box[0] + line.box[2]) / 2 / board.shape[1]) * bw
                        ly = by + ((line.box[1] + line.box[3]) / 2 / board.shape[0]) * bh
                        if cx <= lx < cx + cw and cy <= ly < cy + ch:
                            selected.append(line)
                    if selected:
                        species, rank = parse_cell_text(" ".join(line.text for line in selected))
                        if species is not None and rank is not None:
                            cells[i] = Cell(i, "prisoner", species, rank)
                            confidences[f"cells.{i}"] = min(line.confidence for line in selected)
        else:
            unknown.add("board")
            issues.append("尚未校准棋盘：请框选完整外环并学习实际格子模板。")
            cells = [Cell(i, "unknown") for i in range(16)]
        unread = []
        previous_cells = {cell.index: cell for cell in previous.cells}
        # Marks are not reconstructed by a species/rank template. Keep the
        # user's records visible, but never treat their continued existence as
        # an observation (a replacement or a consumed mark can change them).
        cells = [replace(cell, marks=previous_cells[cell.index].marks) for cell in cells]
        unknown.discard("recognition_failed")
        unknown.discard("marks")  # Migrate earlier profiles to per-cell review.
        unknown.update(f"marks.{cell.index}" for cell in cells)
        unknown.add("patrol_options")
        for cell in cells:
            field = f"cells.{cell.index}"
            if cell.kind == "unknown":
                unknown.add(field)
                unread.append(str(cell.index))
            else:
                unknown.discard(field)
        if unread:
            issues.append("格子未可靠识别（助手编号 0–15）：" + "、".join(unread) + "；请学习模板或手动确认。")
        issues.append("本帧未解析牢房标记与巡监阶段；旧标记和巡监选项保留供核对，请重新确认标记并录入当前巡监目的地。")
        if not self._templates:
            issues.append("尚无实机模板；OCR 只识别画面中实际出现的中文名称与品质。")

        regions = self.profile["regions"]
        observed: dict[str, tuple[str, float]] = {}
        for name, rect in regions.items():
            observed[name] = self._text(crop(image, tuple(rect)))
            confidences[name] = observed[name][1]

        def numeric(name: str, low: int, high: int) -> int | None:
            text, score = observed.get(name, ("", 0.0))
            # Multiple numbers (e.g. ambiguous day/max-day labels) are not guessed.
            numbers = re.findall(r"(?<!\d)\d+(?!\d)", text)
            # A visible vault fraction has a known denominator; preserve the
            # absolute numerator rather than adding it to the previous count.
            fraction = re.search(r"(?<!\d)(\d+)\s*[/／]\s*10(?!\d)", text) if name == "rainbows" else None
            if fraction and len(numbers) == 2:
                numbers = [fraction.group(1)]
            if score >= 0.88 and len(numbers) == 1 and low <= int(numbers[0]) <= high:
                unknown.discard(name)
                return int(numbers[0])
            unknown.add(name)
            return None

        day = numeric("day", 1, previous.max_days + 1) if "day" in regions else None
        if day is not None:
            unknown.discard("day_convention")
            unknown.discard("days_remaining")
        if day is None and "days_remaining" in regions:
            remaining = numeric("days_remaining", 0, previous.max_days)
            if remaining is not None:
                # UI must confirm this convention against a real game's day label.
                day = previous.max_days - remaining + 1
                unknown.discard("day")
                unknown.add("day_convention")
                issues.append("由剩余天数推算当前天数，请确认当天是否计入剩余天数。")
        if day is None:
            unknown.add("day")
        rainbows = numeric("rainbows", 0, 10)
        position_base = self.profile.get("position_base", 0)
        if type(position_base) is not int or position_base not in (0, 1):
            raise ValueError("识别配置 position_base 必须为 0 或 1。")
        position_display = numeric("position", position_base, position_base + 15)
        position = position_display - position_base if position_display is not None else None
        remaining_replaces = numeric("replacements_left", 0, 20)
        daily_text, daily_score = observed.get("daily_action", ("", 0.0))
        daily_action = previous.daily_action
        daily_matches = [action for action, words in (("none", ("未操作", "未使用")), ("eat", ("已吞噬", "已吞食")), ("replace", ("已替换", "已更换"))) if any(word in daily_text for word in words)]
        if daily_score >= 0.88 and len(daily_matches) == 1:
            daily_action = daily_matches[0]
            unknown.discard("daily_action")
        else:
            unknown.add("daily_action")

        def read_effects(name: str, old: tuple[Effect, ...]) -> tuple[Effect, ...]:
            text, score = observed.get(name, ("", 0.0))
            # The effects ROI is the instrument panel. "No instruments" says
            # nothing about global events, tales or previously unknown effects.
            instruments = {"rope", "pestle", "rathead", "saw", "muzzle"}
            protected = tuple(effect for effect in old if effect.id not in instruments) if name == "effects" else ()
            if score < 0.88 or not text:
                unknown.add(name)
                return old
            empty_words = {"effects": ("无", "无刑具", "没有刑具"),
                           "treasures": ("无", "无珍藏", "没有珍藏"),
                           "pending_options": ("无", "无待选项")}
            if text.strip() in empty_words[name]:
                if any(effect.id != "patrol_options" for effect in protected):
                    unknown.add(name)
                else:
                    unknown.discard(name)
                return protected
            unknown.add(name)
            # This is a visible label, not an inferred effect implementation.
            observed_effect = Effect("unknown:" + sha256(text.encode()).hexdigest()[:12], text, confirmed=False)
            retained = tuple(effect for effect in old if effect.id != observed_effect.id)
            return retained + (observed_effect,)

        effects = read_effects("effects", previous.effects)
        treasures = read_effects("treasures", previous.treasures)
        pending_options = read_effects("pending_options", previous.pending_options)
        missing_core = [name for name in ("day", "rainbows", "position", "effects", "treasures", "daily_action", "replacements_left", "pending_options") if name in unknown]
        if missing_core:
            names = {"day": "当前天数", "rainbows": "绮彩数量", "position": "典狱长位置", "effects": "刑具", "treasures": "珍藏", "daily_action": "本日已用操作", "replacements_left": "剩余替换次数", "pending_options": "待选事件/奖励"}
            issues.append("尚需核验：" + "、".join(names[n] for n in missing_core) + "。缺失区域不会沿用旧值冒充已识别。")
        if self.ocr.error:
            issues.append(self.ocr.error)
        result = replace(previous, cells=tuple(cells), day=day if day is not None else previous.day,
                         position=position, rainbows=rainbows, daily_action=daily_action,
                         replacements_left=remaining_replaces if remaining_replaces is not None else previous.replacements_left,
                         effects=effects, treasures=treasures, pending_options=pending_options, source="screenshot",
                         unknown_fields=tuple(sorted(unknown)))
        if result.fingerprint() != previous.fingerprint():
            result = replace(result, revision=previous.revision + 1)
        return RecognitionResult(result, tuple(dict.fromkeys(issues)), confidences)
