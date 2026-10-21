"""Render our Qt application offscreen and validate basic UI/worker integration."""
import os
from pathlib import Path
import sys
import tempfile
import time

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
scratch = tempfile.TemporaryDirectory(prefix="talos-ui-smoke-")
os.environ["TALOS_DATA_DIR"] = scratch.name

from PySide6.QtWidgets import QApplication
from talos_assistant.app import MainWindow
from talos_assistant.theme import STYLE, configure_fonts

app = QApplication([])
app.setStyle("Fusion")
configure_fonts(app)
app.setStyleSheet(STYLE)
window = MainWindow()
window.show()
app.processEvents()
window.load_demo()
deadline = time.monotonic() + 12
while time.monotonic() < deadline:
    app.processEvents()
    if window.last_rec is not None and not window.workers and not window.analysis_timer.isActive():
        break
    time.sleep(.01)
assert window.last_rec is not None, "No solver result reached UI"
assert window.last_rec.state_id == window.state.fingerprint()
assert window.ceiling_button.isChecked()
original = window.state.fingerprint()
window.average_button.click()
deadline = time.monotonic()+12
while time.monotonic() < deadline:
    app.processEvents()
    if window.last_rec and window.last_rec.mode == "average" and not window.workers: break
    time.sleep(.01)
assert window.state.fingerprint() == original
assert window.last_rec.mode == "average"
assert not window.threshold.isEnabled()
artifacts = root / "artifacts"
artifacts.mkdir(exist_ok=True)
window.grab().save(str(artifacts / "desktop-preview.png"))
window.tabs.setCurrentIndex(1)
app.processEvents()
window.grab().save(str(artifacts / "calibration-preview.png"))
window.close()
app.processEvents()
print("Qt smoke passed: render, background result, strategy switch, board preservation, clean close")
