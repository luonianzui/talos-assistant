"""Synthetic pipeline tests; these do not measure real-game recognition accuracy."""
from dataclasses import replace

import cv2
import numpy as np

from talos_assistant.domain import Cell, Effect, GameState
from talos_assistant.capture import load_image
from talos_assistant.vision import LocalOCR, OCRLine, Recognizer, StableFrameGate, parse_cell_text


def synthetic_board():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    rng = np.random.default_rng(22)
    for i in range(16):
        x, y = i % 4 * 100, i // 4 * 100
        # Low-frequency synthetic features survive resampling.
        patch = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        image[y:y + 100, x:x + 100] = cv2.resize(patch, (100, 100), interpolation=cv2.INTER_NEAREST)
    return image


def test_stable_gate_requires_two_and_suppresses_duplicates():
    gate = StableFrameGate()
    a = np.full((40, 50, 3), 30, dtype=np.uint8)
    b = np.full_like(a, 180)
    assert not gate.accept(a)
    assert gate.accept(a)
    assert not gate.accept(a)
    assert not gate.accept(b)
    assert gate.accept(b)
    assert not gate.accept(b)
    gate.reset()
    assert not gate.accept(b)
    assert gate.accept(b)


def test_gate_does_not_accept_slow_drift():
    gate = StableFrameGate(required_frames=3, tolerance=1.1)
    assert not gate.accept(np.full((40, 50, 3), 1, dtype=np.uint8))
    assert not gate.accept(np.full((40, 50, 3), 2, dtype=np.uint8))
    assert not gate.accept(np.full((40, 50, 3), 3, dtype=np.uint8))


def test_gate_notices_a_small_counter_change():
    gate = StableFrameGate()
    a = np.zeros((1080, 1920, 3), dtype=np.uint8)
    b = a.copy()
    b[30:45, 1800:1815] = 255
    assert not gate.accept(a)
    assert gate.accept(a)
    assert not gate.accept(b)
    assert gate.accept(b)


def test_templates_survive_scale_and_profile_reload(tmp_path):
    image = synthetic_board()
    profile = tmp_path / "识别.json"
    rec = Recognizer(profile, tmp_path / "absent")
    rec.set_board_layout("grid4x4")
    rec.set_board_rect((0, 0, 1, 1))
    for i in range(16):
        rec.learn_cell(image, i, ("elephant", "lion", "wolf", "rat")[i % 4], i % 6)
    rec = Recognizer(profile, tmp_path / "absent")
    scaled = cv2.resize(image, (800, 800), interpolation=cv2.INTER_NEAREST)
    result = rec.recognize(scaled, GameState())
    assert all(c.kind == "prisoner" for c in result.state.cells)
    assert result.state.cells[7].species == "rat"
    assert result.state.cells[7].rank == 1
    assert all(result.confidences[f"cells.{i}"] > 0.96 for i in range(16))
    assert "rainbows" in result.state.unknown_fields


def test_missing_configuration_is_unknown_not_a_fake_board(tmp_path):
    rec = Recognizer(tmp_path / "empty.json", tmp_path / "no_models")
    result = rec.recognize(synthetic_board(), GameState())
    assert all(c.kind == "unknown" for c in result.state.cells)
    assert result.state.position is None
    assert result.state.rainbows is None
    assert "board" in result.state.unknown_fields
    assert "day" in result.state.unknown_fields
    assert any("尚未校准" in issue for issue in result.issues)


def test_ambiguous_identical_template_is_not_accepted(tmp_path):
    image = synthetic_board()
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_board_layout("grid4x4")
    rec.set_board_rect((0, 0, 1, 1))
    rec.learn_cell(image, 0, "lion", 2)
    rec.learn_cell(image, 0, "wolf", 2)
    cell, confidence = rec._match_cell(image, 0)
    assert cell is None
    assert confidence > 0.99


def test_absolute_counts_do_not_accumulate(tmp_path):
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("rainbows", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("绮彩 3", 0.99),)
    first = rec.recognize(synthetic_board(), replace(GameState(), rainbows=2))
    second = rec.recognize(synthetic_board(), first.state)
    assert first.state.rainbows == second.state.rainbows == 3
    assert first.state.revision == second.state.revision
    assert first.state.fingerprint() == second.state.fingerprint()


def test_low_confidence_counter_stays_unknown(tmp_path):
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("rainbows", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("3", 0.51),)
    result = rec.recognize(synthetic_board(), GameState())
    assert result.state.rainbows is None
    assert "rainbows" in result.state.unknown_fields
    assert result.confidences["rainbows"] == 0.51


def test_vault_fraction_reads_numerator(tmp_path):
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("rainbows", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("绮彩 3/10", 0.99),)
    result = rec.recognize(synthetic_board(), GameState())
    assert result.state.rainbows == 3
    assert "rainbows" not in result.state.unknown_fields


def test_unknown_instrument_preserved_as_unknown_effect(tmp_path):
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("effects", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("未验证刑具", 0.99),)
    result = rec.recognize(synthetic_board(), GameState())
    assert result.state.effects[0].id.startswith("unknown:")
    assert not result.state.effects[0].confirmed
    assert "effects" in result.state.unknown_fields


def test_marks_and_treasures_are_preserved_but_unverified(tmp_path):
    previous = GameState()
    cells = list(previous.cells)
    cells[1] = Cell(1, "prisoner", "lion", 2, (Effect("prey", "猎物", confirmed=True),))
    treasure = Effect("horn", "号角", confirmed=True)
    previous = replace(previous, cells=tuple(cells), treasures=(treasure,))
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    result = rec.recognize(synthetic_board(), previous)
    assert result.state.cells[1].marks == previous.cells[1].marks
    assert result.state.treasures == (treasure,)
    assert {"marks.1", "treasures", "patrol_options"} <= set(result.state.unknown_fields)
    assert all(f"marks.{i}" in result.state.unknown_fields for i in range(16))
    assert "marks" not in result.state.unknown_fields


def test_instrument_ocr_does_not_erase_events_and_unknown_effects(tmp_path):
    old = (Effect("chain", "连锁", confirmed=True), Effect("cagefight", "困兽之斗", confirmed=True),
           Effect("unknown:global", "待核对全局效果"), Effect("rope", "绳索", confirmed=True))
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("effects", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("无刑具", 0.99),)
    result = rec.recognize(synthetic_board(), replace(GameState(), effects=old))
    assert result.state.effects == old[:3]
    assert "effects" in result.state.unknown_fields
    rec.ocr.read = lambda image: (OCRLine("新刑具", 0.99),)
    nonempty = rec.recognize(synthetic_board(), replace(GameState(), effects=old))
    assert all(effect in nonempty.state.effects for effect in old)


def test_no_instruments_does_not_erase_pending_patrol(tmp_path):
    route = Effect("patrol_options", "巡监", {"targets": [1, 3, 5], "pending": True}, True)
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("effects", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("无刑具", 0.99),)
    result = rec.recognize(synthetic_board(), replace(GameState(), effects=(route,)))
    assert result.state.effects == (route,)
    assert "effects" not in result.state.unknown_fields
    assert "patrol_options" in result.state.unknown_fields


def test_position_number_base_is_explicit_and_range_checked(tmp_path):
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("position", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("位置 1", 0.99),)
    assert rec.recognize(synthetic_board(), GameState()).state.position == 1
    rec.set_position_base(1)
    assert rec.recognize(synthetic_board(), GameState()).state.position == 0
    rec.ocr.read = lambda image: (OCRLine("位置 0", 0.99),)
    assert rec.recognize(synthetic_board(), GameState()).state.position is None


def test_empty_treasure_panel_can_confirm_empty_only(tmp_path):
    rec = Recognizer(tmp_path / "profile.json", tmp_path / "absent")
    rec.set_region("treasures", (0, 0, 0.2, 0.2))
    rec.ocr.read = lambda image: (OCRLine("无珍藏", 0.99),)
    result = rec.recognize(synthetic_board(), replace(GameState(), treasures=(Effect("horn"),)))
    assert result.state.treasures == ()
    assert "treasures" not in result.state.unknown_fields


def test_no_model_never_constructs_rapidocr_or_downloads(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    def forbidden(**kwargs):
        raise AssertionError("missing models must not create a downloader-capable engine")
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=forbidden))
    local = LocalOCR(tmp_path)
    assert local.read(synthetic_board()) == ()
    assert "缺少本地模型" in local.error


def test_ocr_supplies_explicit_local_models_and_cpu_settings(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion
    for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt", "font.ttf"):
        (tmp_path / name).write_bytes(b"synthetic-placeholder")
    recorded = {}
    class FakeOCR:
        def __init__(self, params):
            recorded.update(params)
        def __call__(self, image):
            return SimpleNamespace(txts=("测试",), scores=(0.99,), boxes=np.array([[[0, 0], [20, 0], [20, 10], [0, 10]]]))
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=FakeOCR, EngineType=EngineType,
                        LangCls=LangCls, LangDet=LangDet, LangRec=LangRec, ModelType=ModelType, OCRVersion=OCRVersion))
    local = LocalOCR(tmp_path)
    result = local.read(synthetic_board())
    assert result == (OCRLine("测试", 0.99, (0, 0, 20, 10)),)
    assert recorded["Rec.rec_keys_path"] == str((tmp_path / "keys.txt").resolve())
    assert recorded["Global.font_path"] == str((tmp_path / "font.ttf").resolve())
    assert recorded["Rec.ocr_version"] is OCRVersion.PPOCRV4
    assert recorded["Det.limit_type"] == "max"
    assert recorded["EngineConfig.onnxruntime.use_cuda"] is False
    assert recorded["EngineConfig.onnxruntime.use_dml"] is False


def test_unicode_image_path(tmp_path):
    original = synthetic_board()
    ok, encoded = cv2.imencode(".png", original)
    assert ok
    filename = tmp_path / "中文截图.png"
    filename.write_bytes(encoded.tobytes())
    assert np.array_equal(load_image(filename), original)


def test_chinese_species_rank_parser_rejects_ambiguity():
    assert parse_cell_text("传奇 巨象") == ("elephant", 3)
    assert parse_cell_text("至臻 硕鼠") == ("rat", 4)
    assert parse_cell_text("传奇巨象 稀有狂狮") == (None, None)


def test_ring_layout_is_clockwise_16_cells(tmp_path):
    rec = Recognizer(tmp_path / "profile.json")
    rec.set_board_rect((0, 0, 1, 1))
    assert rec.board_cell_rect(0) == (0, 0, 1 / 6, 1 / 4)
    assert rec.board_cell_rect(5) == (5 / 6, 0, 1 / 6, 1 / 4)
    assert rec.board_cell_rect(15) == (0, 1 / 4, 1 / 6, 1 / 4)


def test_cell_override_supports_nonregular_capture(tmp_path):
    rec = Recognizer(tmp_path / "profile.json")
    rec.set_board_rect((0, 0, 1, 1))
    rec.set_cell_rect(2, (0.3, 0.1, 0.1, 0.2))
    assert rec.board_cell_rect(2) == (0.3, 0.1, 0.1, 0.2)


def test_stability_view_ignores_uncalibrated_animation(tmp_path):
    rec = Recognizer(tmp_path / "profile.json")
    rec.set_region("rainbows", (0, 0, 0.25, 0.25))
    first = np.zeros((400, 400, 3), dtype=np.uint8)
    animated = first.copy()
    animated[200:] = 200
    assert np.array_equal(rec.stability_view(first), rec.stability_view(animated))
    animated[:100, :100] = 200
    assert not np.array_equal(rec.stability_view(first), rec.stability_view(animated))


def test_uncalibrated_stability_view_returns_full_frame(tmp_path):
    rec = Recognizer(tmp_path / "profile.json")
    image = synthetic_board()
    assert rec.stability_view(image) is image
