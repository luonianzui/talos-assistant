from __future__ import annotations

from dataclasses import asdict
import json

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QMessageBox, QPlainTextEdit, QVBoxLayout, QWidget)

from .domain import Cell, Effect, GameState, RANK_NAMES, SPECIES_NAMES, validate_state

KIND_NAMES = {"office": "办公室", "closed": "未开启", "empty": "空牢房", "prisoner": "囚犯", "event": "狱中事变", "unknown": "未识别"}


class CellDialog(QDialog):
    def __init__(self, cell: Cell, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"校正牢房 {cell.index:02d}")
        self.resize(420, 390)
        self.cell = cell
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.kind = QComboBox()
        for key, label in KIND_NAMES.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(self.kind.findData(cell.kind))
        self.species = QComboBox()
        self.species.addItem("未知", None)
        for key, label in SPECIES_NAMES.items():
            self.species.addItem(label, key)
        self.species.setCurrentIndex(self.species.findData(cell.species))
        self.rank = QComboBox()
        self.rank.addItem("未知", None)
        for i, label in enumerate(RANK_NAMES):
            self.rank.addItem(label, i)
        self.rank.setCurrentIndex(self.rank.findData(cell.rank))
        form.addRow("牢房状态", self.kind)
        form.addRow("囚犯代号", self.species)
        form.addRow("品质", self.rank)
        layout.addLayout(form)
        layout.addWidget(QLabel("标记（高级，JSON 数组；没有时保持 []）"))
        self.marks = QPlainTextEdit(json.dumps([asdict(e) for e in cell.marks], ensure_ascii=False, indent=2))
        layout.addWidget(self.marks)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self) -> Cell:
        data = json.loads(self.marks.toPlainText())
        if not isinstance(data, list):
            raise ValueError("标记需为 JSON 数组")
        marks = tuple(Effect(**e) for e in data)
        kind = self.kind.currentData()
        result = Cell(self.cell.index, kind, self.species.currentData() if kind in ("prisoner", "closed") else None,
                      self.rank.currentData() if kind in ("prisoner", "closed") else None, marks)
        # Validate the advanced mark fields before closing the editor, so an
        # invalid JSON value is reported here instead of escaping a Qt callback.
        defaults = GameState()
        validate_state(GameState(cells=tuple(result if c.index == result.index else c for c in defaults.cells)))
        return result

    def accept(self):
        try:
            self.value()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "格式有误", str(exc))
            return
        super().accept()


class JsonDialog(QDialog):
    def __init__(self, title, value, explanation, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(660, 540)
        layout = QVBoxLayout(self)
        label = QLabel(explanation)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.editor = QPlainTextEdit(json.dumps(value, ensure_ascii=False, indent=2))
        layout.addWidget(self.editor)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self):
        return json.loads(self.editor.toPlainText())

    def accept(self):
        try:
            self.value()
        except ValueError as exc:
            QMessageBox.warning(self, "JSON 格式有误", str(exc))
            return
        super().accept()


class ImageCanvas(QWidget):
    regionSelected = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 300)
        self.pixmap = None
        self.image_rect = QRectF()
        self.start_point = None
        self.end_point = None
        self.regions: dict[str, tuple] = {}
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, bgr):
        if bgr is None:
            self.pixmap = None
        else:
            rgb = bgr[:, :, ::-1].copy()
            h, w = rgb.shape[:2]
            self.pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a111c"))
        if self.pixmap is None:
            painter.setPen(QColor("#8e9caf"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "选择游戏窗口，或导入截图\n拖拽框选棋盘与计数区域")
            return
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        w, h = self.pixmap.width() * scale, self.pixmap.height() * scale
        self.image_rect = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        painter.drawPixmap(self.image_rect, self.pixmap, QRectF(self.pixmap.rect()))
        for name, rect in self.regions.items():
            x, y, rw, rh = rect
            r = QRectF(self.image_rect.x() + x*w, self.image_rect.y() + y*h, rw*w, rh*h)
            painter.setPen(QPen(QColor("#63dec3"), 2))
            painter.drawRect(r)
            painter.drawText(r.adjusted(4, 2, 0, 0), Qt.AlignmentFlag.AlignTop, name)
        if self.start_point is not None and self.end_point is not None:
            painter.setPen(QPen(QColor("#ffd177"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self.start_point, self.end_point).normalized())

    def mousePressEvent(self, event):
        if self.pixmap and self.image_rect.contains(event.position()):
            self.start_point = event.position()
            self.end_point = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            self.end_point = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start_point is None:
            return
        selection = QRectF(self.start_point, event.position()).normalized().intersected(self.image_rect)
        self.start_point = self.end_point = None
        if selection.width() >= 8 and selection.height() >= 8:
            base = self.image_rect
            self.regionSelected.emit(((selection.x()-base.x())/base.width(), (selection.y()-base.y())/base.height(), selection.width()/base.width(), selection.height()/base.height()))
        self.update()
