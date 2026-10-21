import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import time
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from talos_assistant.app import MainWindow
from talos_assistant.domain import Cell, GameState, Recommendation
from talos_assistant.vision import RecognitionResult
from talos_assistant.widgets import CellDialog


def drain_qt_events(app):
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture(scope="session")
def qt_app():
    # Keep one strong reference through the entire test session. Recreating the
    # application between tests can leave queued Qt deletions at interpreter exit.
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app
    drain_qt_events(app)


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("TALOS_DATA_DIR", str(tmp_path))
    ui = MainWindow()
    qt_app.processEvents()
    yield ui
    ui.capture_timer.stop()
    ui.analysis_timer.stop()
    ui.cancel_search()
    for worker in list(ui.workers):
        assert worker.wait(5000), "Solver thread did not stop during test cleanup"
    if ui.ocr_worker is not None:
        assert ui.ocr_worker.wait(5000), "OCR thread did not stop during test cleanup"
    # Dispatch finished signals before deleting their window-owned QThreads.
    drain_qt_events(qt_app)
    ui.close()
    ui.deleteLater()
    drain_qt_events(qt_app)


def wait_for_result(ui, mode):
    until = time.monotonic()+12
    while time.monotonic()<until:
        QApplication.processEvents()
        if ui.last_rec and ui.last_rec.mode == mode and not ui.workers:
            return
        time.sleep(.005)
    pytest.fail("No current strategy result")


def test_mode_change_preserves_board_and_routes_result(window):
    window.load_demo()
    wait_for_result(window, "ceiling")
    board = window.state.fingerprint()
    window.average_button.click()
    wait_for_result(window, "average")
    assert window.state.fingerprint() == board
    assert window.last_rec.strategy_id == window.config.fingerprint()
    assert not window.threshold.isEnabled()


def test_stale_ocr_cannot_replace_manual_edit(window):
    old = window.vision_ticket
    window.set_state(replace(window.state, day=3), "人工修改")
    window.receive_recognition(old, RecognitionResult(replace(window.state, day=9), (), {}))
    assert window.state.day == 3


def test_confirm_does_not_erase_missing_patrol_or_mark(window, monkeypatch):
    window.set_state(replace(window.state, unknown_fields=("day", "patrol_options", "marks.2")), "识别")
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    window.confirm_current_observation()
    assert "day" not in window.state.unknown_fields
    assert "patrol_options" in window.state.unknown_fields
    assert "marks.2" in window.state.unknown_fields


def test_duplicate_observation_does_not_restart_solver(window):
    original = window.gate.generation
    entries = len(window.session.entries)
    window.set_state(replace(window.state, revision=99), "重复")
    assert window.gate.generation == original
    assert len(window.session.entries) == entries


def test_stale_result_is_not_rendered(window):
    window.analyze()
    generation = window.gate.generation
    old = Recommendation(window.state.fingerprint(), window.config.fingerprint(), "ceiling")
    window.cancel_search(clear=True)
    window.receive_recommendation(generation, old)
    assert window.last_rec is None


def test_animation_invalidates_result_before_stability_gate_accepts(window, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    window.ocr_reference = np.zeros_like(image)
    window.ocr_invalidated = False
    window.recognizer = SimpleNamespace(stability_view=lambda frame: frame)
    window.frame_gate = SimpleNamespace(reset=lambda: None, accept=lambda frame: False)
    monkeypatch.setattr("talos_assistant.capture.capture_window", lambda hwnd: image)
    window.last_rec = Recommendation(window.state.fingerprint(), window.config.fingerprint(), "ceiling")
    old_generation = window.gate.generation
    window.capture_tick()
    assert window.last_rec is None
    assert window.gate.generation > old_generation
    assert window.ocr_invalidated


def test_same_recognized_state_restarts_cleared_result(window):
    window.cancel_search(clear=True)
    window.receive_recognition(window.vision_ticket, RecognitionResult(window.state, (), {}))
    assert window.analysis_timer.isActive()


def test_unstable_live_frame_blocks_manual_search(window):
    window.capture_timer.start()
    window.ocr_invalidated = True
    window.analyze()
    assert not window.workers
    assert window.last_rec is None


def test_capture_failure_clears_previous_recommendation(window, monkeypatch):
    def fail(hwnd):
        raise RuntimeError("window unavailable")
    monkeypatch.setattr("talos_assistant.capture.capture_window", fail)
    window.last_rec = Recommendation(window.state.fingerprint(), window.config.fingerprint(), "ceiling")
    window.capture_tick()
    assert window.last_rec is None
    assert "window unavailable" in window.capture_info.text()


def test_failed_manual_ocr_cannot_recommend_from_previous_state(window):
    window.load_demo()
    window.analysis_timer.stop()
    window.recognition_failed(window.vision_ticket, "model unavailable")
    assert "recognition_failed" in window.state.unknown_fields
    assert window.last_rec is None
    assert window.ocr_invalidated
    window.analyze()
    wait_for_result(window, "ceiling")
    assert window.last_rec.status == "blocked"
    assert not window.last_rec.candidates


def test_running_manual_ocr_blocks_old_state_analysis(window):
    from types import SimpleNamespace
    window.ocr_worker = SimpleNamespace(isRunning=lambda: True)
    try:
        window.analyze()
        assert not window.workers
        assert window.last_rec is None
    finally:
        window.ocr_worker = None


def test_invalid_mark_fields_are_reported_before_cell_dialog_closes(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = CellDialog(Cell(1, "closed"), window)
    dialog.marks.setPlainText('[{"id":"nest","params":[],"confirmed":true}]')
    with pytest.raises(ValueError, match="params"):
        dialog.value()
    dialog.show()
    dialog.accept()
    assert warnings and "params" in warnings[0]
    assert dialog.isVisible()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.close()
    dialog.deleteLater()
