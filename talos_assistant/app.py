from __future__ import annotations

from dataclasses import asdict, replace
import html
import json
import os
from pathlib import Path
import sys
import threading
import traceback

# Limit native OCR work; the game should retain most CPU resources.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSlider, QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget, QScrollArea)

from .domain import (Cell, Effect, GameState, ProbabilityModel, RANK_COLORS,
    RANK_NAMES, SPECIES_NAMES, StrategyConfig, state_from_dict, state_to_dict,
    validate_state)
from .storage import ResultGate, Session, Settings, resource_dir, user_data_dir
from .theme import STYLE, configure_fonts
from .widgets import CellDialog, ImageCanvas, JsonDialog, KIND_NAMES


def label(text, name=None):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(callback)
    return widget


class SolveWorker(QThread):
    updateReady = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, generation, state, config, parent=None):
        super().__init__(parent)
        self.generation, self.state, self.config = generation, state, config
        self.cancel = threading.Event()

    def run(self):
        try:
            from .solver import solve
            rec = solve(self.state, self.config, cancel_event=self.cancel,
                        on_update=lambda r: self.updateReady.emit(self.generation, r))
            self.updateReady.emit(self.generation, rec)
        except Exception:
            self.failed.emit(self.generation, traceback.format_exc())


class RecognitionWorker(QThread):
    resultReady = Signal(int, object)
    failed = Signal(str)

    def __init__(self, ticket, recognizer, image, previous, parent=None):
        super().__init__(parent)
        self.ticket, self.recognizer = ticket, recognizer
        self.image, self.previous = image.copy(), previous

    def run(self):
        try:
            self.resultReady.emit(self.ticket, self.recognizer.recognize(self.image, self.previous))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("塔罗斯 · 本地打牌助手")
        self.resize(1180, 850)
        self.setMinimumSize(1020, 740)
        self.directory = user_data_dir()
        self.settings = Settings(self.directory / "settings.json")
        self.config = self.settings.strategy()
        self.state = GameState()
        self.session = Session()
        self.gate = ResultGate()
        self.workers = []
        self.ocr_worker = None
        self.vision_ticket = 0
        self.last_image = None
        self.recognizer = None
        self.frame_gate = None
        self.last_rec = None
        self.ocr_reference = None
        self.ocr_invalidated = False
        self.loading = True
        self.build_ui()
        self.loading = False
        self.sync_state()
        self.session.record(self.state, "新局面")
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(400)
        self.capture_timer.timeout.connect(self.capture_tick)
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.setInterval(300)
        self.analysis_timer.timeout.connect(self.analyze)
        self.refresh_windows()
        QTimer.singleShot(0, self.init_vision)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 19, 24, 13)
        outer.setSpacing(14)
        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(3)
        title_column.addWidget(label("TALOS / LOCAL ASSISTANT", "eyebrow"))
        title_column.addWidget(label("把每一步，走向更多绮彩", "title"))
        header.addLayout(title_column)
        header.addStretch()
        header.addWidget(label("本地运行 · 无需 API", "badge"))
        self.topmost = QCheckBox("置顶")
        self.topmost.toggled.connect(self.set_topmost)
        header.addWidget(self.topmost)
        outer.addLayout(header)

        strategy_panel = QFrame()
        strategy_panel.setObjectName("panel")
        strategy_row = QHBoxLayout(strategy_panel)
        strategy_row.setContentsMargins(16, 10, 16, 10)
        strategy_row.addWidget(label("本局策略", "section"))
        self.mode_group = QButtonGroup(self)
        self.average_button = QPushButton("平均收益")
        self.ceiling_button = QPushButton("冲顶级")
        for btn, ident in ((self.average_button, 0), (self.ceiling_button, 1)):
            btn.setCheckable(True)
            btn.setObjectName("strategy")
            self.mode_group.addButton(btn, ident)
            strategy_row.addWidget(btn)
        (self.average_button if self.config.mode == "average" else self.ceiling_button).setChecked(True)
        self.mode_group.idClicked.connect(self.change_strategy)
        strategy_row.addSpacing(8)
        strategy_row.addWidget(label("冲顶最低达成率"))
        self.threshold = QSpinBox()
        self.threshold.setRange(1, 99)
        self.threshold.setSuffix(" %")
        self.threshold.setValue(round(self.config.threshold * 100))
        self.threshold.valueChanged.connect(self.change_strategy)
        self.threshold.setEnabled(self.config.mode == "ceiling")
        strategy_row.addWidget(self.threshold)
        strategy_row.addStretch()
        strategy_row.addWidget(button("两种策略如何选择？", self.show_strategy_comparison))
        outer.addWidget(strategy_panel)
        self.mode_description = label("", "muted")
        outer.addWidget(self.mode_description)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.build_board_tab()
        self.build_capture_tab()
        self.build_rules_tab()
        footer = QHBoxLayout()
        self.status = label("准备就绪 · 先录入局面或配置画面识别", "muted")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        footer.addWidget(button("新局面", self.new_game))
        footer.addWidget(button("保存 / 导出", self.save_session))
        footer.addWidget(button("导入 / 回放", self.open_session))
        outer.addLayout(footer)
        self.update_mode_description()

    def build_board_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(18)
        left_container = QWidget()
        left = QVBoxLayout(left_container)
        left.setContentsMargins(0, 0, 7, 0)
        heading = QHBoxLayout()
        heading.addWidget(label("当前局面", "section"))
        heading.addStretch()
        heading.addWidget(button("示例局面", self.load_demo))
        left.addLayout(heading)
        self.source_label = label("手动局面 · 点击牢房即可校正", "muted")
        self.source_label.setWordWrap(True)
        left.addWidget(self.source_label)
        grid = QGridLayout()
        grid.setSpacing(7)
        for row in range(4): grid.setRowMinimumHeight(row, 80)
        positions = [(0, i) for i in range(6)] + [(i, 5) for i in range(1, 4)] + [(3, i) for i in range(4, -1, -1)] + [(i, 0) for i in range(2, 0, -1)]
        self.cell_buttons = []
        for index, (r, c) in enumerate(positions):
            btn = QPushButton()
            btn.setObjectName("cell")
            btn.setMinimumSize(77, 77)
            btn.clicked.connect(lambda checked=False, i=index: self.edit_cell(i))
            grid.addWidget(btn, r, c)
            self.cell_buttons.append(btn)
        self.board_center = QLabel()
        self.board_center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.board_center, 1, 1, 2, 4)
        left.addLayout(grid)
        fields = QGridLayout()
        self.day = QSpinBox(); self.day.setRange(1, 21)
        self.rainbows = QSpinBox(); self.rainbows.setRange(0, 999)
        self.rainbows_unknown = QCheckBox("累计未知")
        self.position = QComboBox(); self.position.addItem("位置未知", None)
        for i in range(16): self.position.addItem("办公室 00" if i == 0 else f"牢房 {i:02d}", i)
        self.daily = QComboBox()
        for key, name in (("none", "本日未吃/换"), ("eat", "本日已吃"), ("replace", "本日已换")):
            self.daily.addItem(name, key)
        self.replacements = QSpinBox(); self.replacements.setRange(0, 20)
        for i, (name, widget) in enumerate((("当前天数", self.day), ("已获绮彩", self.rainbows), ("所在位置", self.position), ("本日状态", self.daily), ("剩余换次", self.replacements))):
            fields.addWidget(label(name, "muted"), (i//3)*2, i%3)
            fields.addWidget(widget, (i//3)*2+1, i%3)
        fields.addWidget(self.rainbows_unknown, 3, 2)
        left.addLayout(fields)
        for widget in (self.day, self.rainbows, self.replacements): widget.valueChanged.connect(self.fields_changed)
        self.position.currentIndexChanged.connect(self.fields_changed)
        self.daily.currentIndexChanged.connect(self.fields_changed)
        self.rainbows_unknown.toggled.connect(self.fields_changed)
        row = QHBoxLayout()
        row.addWidget(button("刑具 / 事件 / 珍藏", self.edit_effects))
        row.addWidget(button("巡监候选", self.edit_patrol))
        row.addWidget(button("概率模型", self.edit_probability))
        left.addLayout(row)
        self.confirm_rules = QCheckBox("已核对当前规则配置（详见“规则与数据”）")
        self.confirm_rules.toggled.connect(self.fields_changed)
        left.addWidget(self.confirm_rules)
        self.confirm_observation = button("确认已校正本次识别", self.confirm_current_observation)
        left.addWidget(self.confirm_observation)
        left.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left_container)
        layout.addWidget(scroll, 6)

        right = QVBoxLayout()
        controls = QHBoxLayout()
        self.analyze_button = button("分析当前局面", self.analyze, True)
        controls.addWidget(self.analyze_button)
        controls.addWidget(button("暂停计算", self.pause_search))
        right.addLayout(controls)
        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0); self.progress.setTextVisible(False)
        right.addWidget(self.progress)
        self.result_title = label("等待局面", "section")
        right.addWidget(self.result_title)
        self.results = QTextBrowser()
        self.results.setOpenExternalLinks(False)
        self.results.setHtml(self.empty_result())
        right.addWidget(self.results, 1)
        layout.addLayout(right, 5)
        self.tabs.addTab(page, "局面与建议")

    def build_capture_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        row = QHBoxLayout()
        self.window_combo = QComboBox(); self.window_combo.setMinimumWidth(280)
        self.window_combo.currentIndexChanged.connect(self.capture_source_changed)
        row.addWidget(self.window_combo, 1)
        row.addWidget(button("刷新窗口", self.refresh_windows))
        self.live_button = button("开始自动读取", self.toggle_live, True)
        row.addWidget(self.live_button)
        row.addWidget(button("导入截图", self.import_image))
        row.addWidget(button("识别这一帧", self.recognize_now))
        layout.addLayout(row)
        self.canvas = ImageCanvas()
        self.canvas.regionSelected.connect(self.region_selected)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        row.addWidget(label("拖拽标记"))
        self.region_choice = QComboBox()
        for text, key in (("整个棋盘", "board"), ("指定单格", "cell"), ("当前天数", "day"), ("剩余天数", "days_remaining"), ("绮彩累计数", "rainbows"), ("当前位置编号", "position"), ("刑具 / 效果栏", "effects"), ("珍藏栏", "treasures"), ("本日吃/换状态", "daily_action"), ("剩余换次数", "replacements_left"), ("待选择卡牌", "pending_options")):
            self.region_choice.addItem(text, key)
        row.addWidget(self.region_choice)
        self.layout_choice = QComboBox(); self.layout_choice.addItem("6×4 外环", "ring6x4"); self.layout_choice.addItem("4×4 网格", "grid4x4")
        self.layout_choice.currentIndexChanged.connect(self.layout_changed)
        row.addWidget(self.layout_choice)
        self.position_base = QComboBox(); self.position_base.addItem("画面编号 0–15", 0); self.position_base.addItem("画面编号 1–16", 1)
        self.position_base.currentIndexChanged.connect(self.position_base_changed)
        row.addWidget(self.position_base)
        row.addStretch()
        self.learn_index = QSpinBox(); self.learn_index.setRange(0, 15)
        row.addWidget(label("样本牢房")); row.addWidget(self.learn_index)
        row.addWidget(button("学习该格图标", self.learn_template))
        layout.addLayout(row)
        self.capture_info = label("首次使用：导入实机截图 → 框选棋盘与计数区域 → 标注图标。未识别项会要求校正。", "muted")
        self.capture_info.setWordWrap(True)
        layout.addWidget(self.capture_info)
        layout.addWidget(label("采集可见的游戏客户区；请使用窗口/无边框，并将助手移到棋盘以外。关闭自动读取即停止截图。", "muted"))
        self.tabs.addTab(page, "画面识别与校准")

    def build_rules_tab(self):
        self.rules_view = QTextBrowser()
        self.rules_view.setOpenExternalLinks(True)
        self.tabs.addTab(self.rules_view, "规则与数据")
        self.refresh_rules_text()

    def refresh_rules_text(self):
        path = resource_dir() / "data" / "rules_reference.json"
        reference = path.read_text(encoding="utf-8") if path.exists() else "规则目录正在准备。"
        self.rules_view.setHtml("""<h2>当前规则配置 / provisional_pc</h2>
        <p>这里记录已实现的规则及依据。公开术语、社区观察与本机实测具有不同可信度；勾选核对表示你确认当前局面的规则与此配置一致。</p>
        <p><b>校验重点：</b>所在格发起操作、同阶克制、吞噬后代号、日内吃换限制、巡监候选、标记结算顺序、绮彩与珍藏发放。</p>
        <p>未覆盖效果不会被当作无效果。未知后续概率或奖励会停止终局数值估算，仍可展示已确定的即时变化。</p>
        <h3>使用流程</h3><ol><li>录入或识别棋盘，补全当前位置、天数和本日操作状态。</li>
        <li>在“巡监候选”中填入屏幕实际给出的目的地；没有候选时保持空列表。</li>
        <li>校正识别，选择策略，查看建议。游戏操作后重新读取。</li>
        <li>需要数值推演时，概率模型必须有明确来源。演示模型仅用于测试。</li></ol>
        <h3>数据说明</h3><p>本程序不上传截图、局面或日志，不读取游戏内存，不注入或点击游戏。截图只在内存处理；你主动学习的图标及会话 JSON 保存在程序旁的 user_data 文件夹。</p>
        <h3>规则目录</h3><pre style='white-space:pre-wrap;color:#a9bfd1'>""" + html.escape(reference) + "</pre>")

    def update_mode_description(self):
        if self.config.mode == "average":
            text = "平均收益：优先长期平均绮彩数量；相同时优先更少操作。"
        else:
            text = f"冲顶级：优先至少有 {self.config.threshold:.0%} 机会达到的最高绮彩数量，再比较达成率与平均值。"
        self.mode_description.setText(text + "　·　2 秒初步 / 最多约 10 秒推演")

    def change_strategy(self, *_):
        if self.loading: return
        mode = "average" if self.average_button.isChecked() else "ceiling"
        self.config = replace(self.config, mode=mode, threshold=self.threshold.value()/100)
        self.threshold.setEnabled(mode == "ceiling")
        self.settings.save_strategy(self.config)
        self.update_mode_description()
        self.analyze()

    def set_topmost(self, enabled):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.show()

    def sync_state(self):
        self.loading = True
        self.day.setMaximum(self.state.max_days+1); self.day.setValue(self.state.day)
        self.rainbows.setValue(self.state.rainbows or 0)
        self.rainbows_unknown.setChecked(self.state.rainbows is None)
        self.rainbows.setEnabled(self.state.rainbows is not None)
        self.position.setCurrentIndex(self.position.findData(self.state.position))
        self.daily.setCurrentIndex(self.daily.findData(self.state.daily_action))
        self.replacements.setValue(self.state.replacements_left)
        self.confirm_rules.setChecked(self.state.rules_confirmed)
        for cell in self.state.cells:
            b = self.cell_buttons[cell.index]
            color = RANK_COLORS[cell.rank] if cell.rank is not None else "#718096"
            desc = SPECIES_NAMES.get(cell.species, "代号未知") if cell.kind == "prisoner" else KIND_NAMES[cell.kind]
            rank = RANK_NAMES[cell.rank] if cell.rank is not None else ("待确认" if cell.kind in ("prisoner", "unknown") else "")
            prefix = "● " if self.state.position == cell.index else ""
            b.setText(f"{prefix}{cell.index:02d}\n{desc}\n{rank}")
            border = "#65ddc6" if self.state.position == cell.index else "#304059"
            b.setStyleSheet(f"QPushButton#cell {{color:{color};background:#192536;border:1px solid {border};border-left:3px solid {color};}} QPushButton#cell:hover {{background:#283950;}}")
            b.setToolTip("点击校正" + ("\n标记：" + "、".join(e.name or e.id for e in cell.marks) if cell.marks else ""))
        rainbow_text = "?" if self.state.rainbows is None else str(self.state.rainbows)
        self.board_center.setText(f"<span style='color:#8295ae;font-size:12px'>典 狱 阶 段</span><br><span style='font-size:30px;color:#70e2cb'>第 {self.state.day} 天</span><br><span style='color:#a6b5c8'>已获绮彩　{rainbow_text}　/　剩余 {max(0,self.state.max_days-self.state.day+1)} 天</span>")
        self.source_label.setText(("演示局面（合成数据，不代表实机概率）" if "demo" in self.state.source or "演示" in self.state.source else "当前来源：" + self.state.source) + " · 点击牢房校正")
        self.confirm_observation.setEnabled(bool(self.state.unknown_fields))
        self.loading = False

    def set_state(self, state, description="局面更新", record=True):
        validate_state(state)
        if state.fingerprint() == self.state.fingerprint():
            return
        self.vision_ticket += 1
        self.cancel_search(clear=True)
        self.state = replace(state, revision=self.state.revision+1)
        self.sync_state()
        if record:
            self.session.record(self.state, description)
            try: self.session.save(self.directory / "last_session.json", self.config)
            except OSError as exc: self.status.setText(f"自动保存失败：{exc}")
        self.analysis_timer.start()

    def fields_changed(self, *_):
        if self.loading: return
        state = replace(self.state, day=self.day.value(), rainbows=None if self.rainbows_unknown.isChecked() else self.rainbows.value(), position=self.position.currentData(), daily_action=self.daily.currentData(), replacements_left=self.replacements.value(), rules_confirmed=self.confirm_rules.isChecked())
        self.set_state(state, "人工校正计数")

    def edit_cell(self, index):
        dialog = CellDialog(next(c for c in self.state.cells if c.index == index), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated = dialog.value()
            cells = tuple(updated if c.index == index else c for c in self.state.cells)
            unknown = tuple(k for k in self.state.unknown_fields if k not in (f"cells.{index}", f"marks.{index}"))
            self.set_state(replace(self.state, cells=cells, unknown_fields=unknown), f"校正牢房 {index:02d}")

    def confirm_current_observation(self):
        names = "、".join(self.state.unknown_fields)
        answer = QMessageBox.question(self, "确认局面", "请确认已逐项检查棋盘、天数、位置、吃/换状态、已有刑具和待选卡牌。\n\n尚待确认：" + names + "\n\n确认将采用当前编辑器中的值。未知牢房和未知效果仍保留未知。")
        if answer == QMessageBox.StandardButton.Yes:
            editable = {"day", "day_convention", "rainbows", "position", "daily_action", "replacements_left", "effects", "treasures", "pending_options", "recognition_failed"}
            remaining = tuple(k for k in self.state.unknown_fields if k not in editable)
            if all(c.kind != "unknown" for c in self.state.cells):
                remaining = tuple(k for k in remaining if k != "board")
            self.set_state(replace(self.state, unknown_fields=remaining, source="人工确认的识别局面"), "确认识别")
            if remaining:
                self.status.setText("仍需通过对应编辑器处理：" + "、".join(remaining))

    def edit_effects(self):
        value = {key: [asdict(e) for e in getattr(self.state, key)] for key in ("effects", "treasures", "pending_options")}
        dialog = JsonDialog("刑具、事件、珍藏", value, "从“规则与数据”查找支持的效果 id；params 填写所需参数。未收录效果可用 unknown id 并保留名称，工具会要求补充规则。confirmed 仅表示已核对当前效果。", self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                data = state_to_dict(self.state); data.update(dialog.value())
                self.set_state(state_from_dict(data), "效果更新")
            except Exception as exc: QMessageBox.warning(self, "效果配置无效", str(exc))

    def edit_patrol(self):
        old = next((e for e in self.state.effects if e.id == "patrol_options"), None)
        value = old.params.get("targets", []) if old else []
        dialog = JsonDialog("当前可选巡监目的地", value, "只填写游戏当前显示的可选目的地编号，例如 [3, 7]。当前没有巡监选择时填写 []。不会将所有牢房视为任意可到达。", self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                targets = dialog.value()
                if not isinstance(targets, list) or any(type(i) is not int or i not in range(16) for i in targets): raise ValueError("请输入 0–15 的编号列表")
                effects = tuple(e for e in self.state.effects if e.id != "patrol_options")
                if targets: effects += (Effect("patrol_options", "已观察巡监目的地", {"targets": targets, "pending": True}, True),)
                self.set_state(replace(self.state, effects=effects, unknown_fields=tuple(x for x in self.state.unknown_fields if x != "patrol_options")), "录入巡监选项")
            except Exception as exc: QMessageBox.warning(self, "巡监配置无效", str(exc))

    def edit_probability(self):
        dialog = JsonDialog("概率模型", asdict(self.state.probability), "无实测数据时保持空概率表。species 是代号概率；ranks 是换/开门品质概率（0–5）。每张表总和须为 1，source 必填。估计参数请保留 estimated=true；不应直接套用网页练习概率。", self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                data = state_to_dict(self.state); data["probability"] = dialog.value()
                self.set_state(state_from_dict(data), "概率模型更新")
            except Exception as exc: QMessageBox.warning(self, "概率模型无效", str(exc))

    def analyze(self):
        self.analysis_timer.stop()
        if (self.ocr_worker is not None and self.ocr_worker.isRunning()) or (self.capture_timer.isActive() and self.ocr_invalidated):
            self.cancel_search(clear=True)
            self.result_title.setText("等待稳定画面")
            self.status.setText("画面已变化，等待两帧稳定并完成识别后重新分析。")
            return
        self.cancel_search()
        generation = self.gate.begin(self.state, self.config)
        worker = SolveWorker(generation, self.state, self.config, self)
        worker.updateReady.connect(self.receive_recommendation)
        worker.failed.connect(self.search_failed)
        worker.finished.connect(lambda w=worker, g=generation: self.worker_finished(w, g))
        self.workers.append(worker)
        self.progress.setRange(0, 0)
        self.result_title.setText("正在推演…")
        self.results.setHtml(self.empty_result("正在比较合法操作，等待初步结果…"))
        self.status.setText("计算中 · 当前局面 " + self.state.fingerprint()[:8])
        worker.start()

    def cancel_search(self, clear=False):
        if hasattr(self, "analysis_timer"): self.analysis_timer.stop()
        for worker in self.workers: worker.cancel.set()
        self.gate.invalidate()
        self.progress.setRange(0, 1); self.progress.setValue(0)
        if clear:
            self.last_rec = None
            self.result_title.setText("局面已变化")
            self.results.setHtml(self.empty_result("正在等待重新分析…"))

    def pause_search(self):
        self.cancel_search()
        self.result_title.setText("已暂停计算")
        self.status.setText("计算已暂停；当前内容是暂停前的结果。")

    def worker_finished(self, worker, generation):
        if worker in self.workers: self.workers.remove(worker)
        if generation == self.gate.generation:
            self.progress.setRange(0, 1); self.progress.setValue(1)
        worker.deleteLater()

    def search_failed(self, generation, message):
        if generation != self.gate.generation: return
        self.result_title.setText("计算未完成")
        self.results.setPlainText(message)

    def receive_recommendation(self, generation, rec):
        if not self.gate.accepts(generation, rec): return
        self.last_rec = rec
        status_names = {"exact": "已知模型内最优", "estimated": "当前搜索最佳 · 估算", "conditional": "条件分析 · 信息尚不完整", "blocked": "需要补充局面", "cancelled": "计算已暂停"}
        self.result_title.setText(status_names.get(rec.status, rec.status))
        parts = []
        if self.state.probability.source:
            nature = "估计参数" if self.state.probability.estimated else "显式模型参数"
            parts.append("<p style='color:#93abc3'>概率依据：" + html.escape(self.state.probability.source) + f"（{nature}）</p>")
        if rec.warnings:
            parts.append("<div style='color:#e2b978'><b>需要注意的局面信息</b><ul>" + "".join("<li>"+html.escape(w)+"</li>" for w in rec.warnings) + "</ul></div>")
        for i, candidate in enumerate(rec.candidates[:3]):
            a = candidate.action
            title = a.label or a.kind
            tag = ("首选" if rec.status not in ("conditional", "blocked") else "条件候选") if i == 0 else f"备选 {i}"
            parts.append(f"<h3 style='color:{'#71e2cb' if i==0 else '#ccd8e7'}'>{tag}　{html.escape(title)}</h3>")
            if candidate.mean is not None:
                mean = f"{candidate.mean:.2f}"
                ceiling = str(candidate.ceiling) if candidate.ceiling is not None else "—"
                chance = f"{candidate.hit_probability:.1%}" if candidate.hit_probability is not None else "—"
                parts.append(f"<table width='100%' cellpadding='5'><tr><td>平均绮彩</td><td>冲顶目标</td><td>达成率</td></tr><tr><td><b>{mean}</b></td><td><b>{ceiling}</b></td><td><b>{chance}</b></td></tr></table>")
                qualifier = "完整模型计算" if candidate.exact else "当前搜索估算"
                sample_text = f" · {candidate.samples} 个样本/分支" if candidate.samples else ""
                parts.append(f"<p style='color:#92a6bd'>{qualifier}{sample_text}</p>")
            else:
                parts.append("<p style='color:#92a6bd'>终局收益暂无法可靠估算</p>")
            if candidate.reason: parts.append("<p>"+html.escape(candidate.reason)+"</p>")
            if candidate.conditions: parts.append("<p style='color:#d4b787'>"+"<br>".join(html.escape(s) for s in candidate.conditions)+"</p>")
            parts.append("<hr style='border-color:#2b3b50'>")
        if not rec.candidates:
            parts.append("<p>暂无可推荐操作。请检查待选事件、巡监选项及未确认字段。</p>")
        parts.append("<p style='color:#8497ae;font-size:11px'>概率对应本次规划的后续策略；画面变化后重新评估，旧分布随之失效。数量包含当前已记录绮彩。</p>")
        self.results.setHtml("".join(parts))
        self.status.setText(f"{'平均收益' if self.config.mode=='average' else '冲顶级'} · {rec.elapsed:.2f} 秒 · 局面 {rec.state_id[:8]} · {'完整求解' if rec.complete else '有限搜索 / 条件分析'}")

    @staticmethod
    def empty_result(text="录入棋盘后，选择策略并点击“分析当前局面”。"):
        return f"<h3 style='color:#70dfc6'>下一步，从读懂局面开始</h3><p style='color:#9badc2'>{html.escape(text)}</p><p>可以先载入示例，体验双策略切换和结果展示。</p><p style='color:#8196b0'>未知概率和未覆盖效果会明确显示，不会被当作确定结果。</p>"

    def show_strategy_comparison(self):
        dialog = QDialog(self); dialog.setWindowTitle("平均收益与冲顶级"); dialog.resize(640, 450)
        layout = QVBoxLayout(dialog)
        text = QTextBrowser()
        text.setHtml("""<h2>同一局面，可以有不同目标</h2>
        <p style='color:#e5b46c'>以下为合成教学例子，不是游戏掉率或当前棋盘结果。</p>
        <table border='0' cellpadding='12'><tr><th>方案</th><th>可能结果</th><th>平均数量</th><th>20% 门槛冲顶目标</th></tr>
        <tr><td>A · 稳定路线</td><td>100% 获得 2 个</td><td>2.00</td><td>2 个</td></tr>
        <tr><td>B · 进取路线</td><td>30% 获得 5 个<br>70% 获得 0 个</td><td>1.50</td><td>5 个</td></tr></table>
        <p><b>平均收益选择 A：</b>长期平均数量更高。</p><p><b>冲顶级选择 B：</b>达到 5 个的概率超过 20%，单局上限更高。</p>
        <p>把最低达成率提高到 40% 后，B 不再满足冲顶门槛，A 更合适。未知概率时不会显示此类精确比较。</p>""")
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        dialog.exec()

    def load_demo(self):
        try:
            from .solver import demo_state
            self.set_state(demo_state(), "演示合成局面")
        except ImportError:
            cells = tuple(Cell(i, "office" if i==0 else "empty") for i in range(16))
            self.set_state(GameState(cells=cells, day=20, position=1, rainbows=2, rules_confirmed=True, source="demo：合成校验局面"), "演示局面")

    def new_game(self):
        self.session = Session()
        self.set_state(GameState(), "新局面")

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存局面与回放", str(self.directory / "talos-session.json"), "JSON (*.json)")
        if path:
            try:
                self.session.save(Path(path), self.config)
                self.status.setText("已保存："+path)
            except Exception as exc: QMessageBox.warning(self, "保存失败", str(exc))

    def open_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入局面 / 回放", str(self.directory), "JSON (*.json)")
        if not path: return
        try: session = Session.load(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "无法导入", str(exc)); return
        dialog = QDialog(self); dialog.setWindowTitle("本地局面回放"); dialog.resize(700, 500)
        layout = QVBoxLayout(dialog)
        summary = QLabel(); summary.setWordWrap(True); layout.addWidget(summary)
        preview = QTextBrowser(); layout.addWidget(preview)
        slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0, len(session.entries)-1); slider.setValue(len(session.entries)-1); layout.addWidget(slider)
        def show_entry(index):
            state = session.state(index); entry = session.entries[index]
            summary.setText(f"{index+1} / {len(session.entries)} · {entry['label']} · {entry['time']}")
            preview.setPlainText(json.dumps(state_to_dict(state), ensure_ascii=False, indent=2))
        slider.valueChanged.connect(show_entry); show_entry(slider.value())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Open).setText("采用此局面")
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.session = session
            self.set_state(session.state(slider.value()), "回放载入")

    def init_vision(self):
        try:
            from .vision import Recognizer, StableFrameGate
            self.recognizer = Recognizer(self.directory / "vision-profile.json", models_dir=resource_dir() / "models")
            self.frame_gate = StableFrameGate(required_frames=2)
            self.layout_choice.blockSignals(True)
            self.layout_choice.setCurrentIndex(self.layout_choice.findData(self.recognizer.profile.get("layout", "ring6x4")))
            self.layout_choice.blockSignals(False)
            self.position_base.blockSignals(True)
            self.position_base.setCurrentIndex(self.position_base.findData(self.recognizer.profile.get("position_base", 0)))
            self.position_base.blockSignals(False)
            self.update_regions()
        except Exception as exc: self.capture_info.setText("识别模块未就绪："+str(exc))

    def refresh_windows(self):
        if not hasattr(self, "window_combo"): return
        selected = self.window_combo.currentData()
        self.window_combo.clear()
        try:
            from .capture import list_windows
            for item in list_windows():
                if "本地打牌助手" not in item.title:
                    self.window_combo.addItem(item.title, item.hwnd)
            index = self.window_combo.findData(selected)
            if index >= 0: self.window_combo.setCurrentIndex(index)
            else:
                for i in range(self.window_combo.count()):
                    if any(s in self.window_combo.itemText(i).lower() for s in ("torchlight", "火炬")):
                        self.window_combo.setCurrentIndex(i); break
        except Exception as exc: self.capture_info.setText(str(exc))

    def capture_source_changed(self, *_):
        self.vision_ticket += 1
        if self.frame_gate: self.frame_gate.reset()
        self.ocr_reference = None
        self.ocr_invalidated = True
        if hasattr(self, "capture_timer") and self.capture_timer.isActive():
            self.cancel_search(clear=True)

    def toggle_live(self):
        if self.capture_timer.isActive():
            self.capture_timer.stop(); self.live_button.setText("开始自动读取"); self.vision_ticket += 1
            return
        if self.window_combo.currentData() is None:
            QMessageBox.information(self, "没有可采集窗口", "请先打开游戏，再刷新窗口列表。"); return
        self.capture_source_changed()
        self.cancel_search(clear=True)
        self.capture_timer.start(); self.live_button.setText("停止自动读取")
        self.capture_tick()

    def capture_tick(self):
        try:
            from .capture import capture_window
            image = capture_window(self.window_combo.currentData())
            self.last_image = image
            self.canvas.set_image(image)
            tracked = self.recognizer.stability_view(image) if self.recognizer and hasattr(self.recognizer, "stability_view") else image
            if self.ocr_reference is not None and not self.ocr_invalidated and self.frame_changed(tracked, self.ocr_reference):
                self.vision_ticket += 1
                self.ocr_invalidated = True
                self.cancel_search(clear=True)
                self.result_title.setText("等待稳定画面")
                if self.frame_gate: self.frame_gate.reset()
            if self.ocr_worker is not None and self.ocr_worker.isRunning():
                return
            if self.frame_gate and self.frame_gate.accept(tracked): self.start_recognition(image)
        except Exception as exc:
            self.recognition_failed(self.vision_ticket, "采集已停止：" + str(exc))

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入游戏截图", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                from .capture import load_image
                self.capture_timer.stop(); self.live_button.setText("开始自动读取")
                self.capture_source_changed()
                self.cancel_search(clear=True)
                self.last_image = load_image(path); self.canvas.set_image(self.last_image)
                self.capture_info.setText("截图已载入。可框选区域，再点击“识别这一帧”。")
            except Exception as exc: QMessageBox.warning(self, "读取失败", str(exc))

    def recognize_now(self):
        if self.last_image is None:
            QMessageBox.information(self, "需要画面", "先导入截图或开始自动读取。"); return
        self.start_recognition(self.last_image)

    def start_recognition(self, image):
        if self.recognizer is None: self.init_vision()
        if self.recognizer is None: return
        if self.ocr_worker is not None and self.ocr_worker.isRunning(): return
        self.vision_ticket += 1
        ticket = self.vision_ticket
        self.cancel_search(clear=True)
        self.capture_info.setText("正在本地识别…")
        self.ocr_reference = (self.recognizer.stability_view(image) if hasattr(self.recognizer, "stability_view") else image).copy()
        self.ocr_invalidated = False
        self.ocr_worker = RecognitionWorker(ticket, self.recognizer, image, self.state, self)
        self.ocr_worker.resultReady.connect(self.receive_recognition)
        self.ocr_worker.failed.connect(lambda msg, t=ticket: self.recognition_failed(t, msg))
        self.ocr_worker.finished.connect(lambda w=self.ocr_worker: self.ocr_finished(w))
        self.ocr_worker.start()

    @staticmethod
    def frame_changed(current, previous):
        import cv2
        import numpy as np
        if current.shape != previous.shape: return True
        a = cv2.resize(current, (192, 128)).astype(np.float32)
        b = cv2.resize(previous, (192, 128)).astype(np.float32)
        diff = np.abs(a-b)
        return float(diff.mean()) > 1.5 or float(diff.mean(axis=2).max()) > 9

    def ocr_finished(self, worker):
        if self.ocr_worker is worker: self.ocr_worker = None
        worker.deleteLater()

    def receive_recognition(self, ticket, result):
        if ticket != self.vision_ticket: return
        self.ocr_invalidated = False
        self.capture_info.setText("；".join(result.issues) or "识别完成，请核对关键字段。")
        unchanged = result.state.fingerprint() == self.state.fingerprint()
        self.set_state(result.state, "截图识别")
        if unchanged and self.last_rec is None:
            self.analysis_timer.start()

    def recognition_failed(self, ticket, message):
        if ticket != self.vision_ticket: return
        self.capture_timer.stop()
        self.live_button.setText("开始自动读取")
        if self.frame_gate: self.frame_gate.reset()
        self.ocr_invalidated = True
        unknown = tuple(sorted(set(self.state.unknown_fields) | {"recognition_failed"}))
        self.set_state(replace(self.state, unknown_fields=unknown), "识别失败，保留待核对旧值")
        self.cancel_search(clear=True)
        self.capture_info.setText("采集或识别失败，自动读取已停止，当前旧值仅供校正；请重新读取或完整核对局面：" + message[-500:])

    def region_selected(self, rect):
        if not self.recognizer: self.init_vision()
        if not self.recognizer: return
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.capture_info.setText("请等待本次识别完成后再修改区域。"); return
        key = self.region_choice.currentData()
        try:
            if key == "board": self.recognizer.set_board_rect(rect)
            elif key == "cell": self.recognizer.set_cell_rect(self.learn_index.value(), rect)
            else: self.recognizer.set_region(key, rect)
            self.capture_source_changed()
            self.update_regions()
            self.capture_info.setText("已保存区域："+self.region_choice.currentText())
        except Exception as exc: self.capture_info.setText(str(exc))

    def update_regions(self):
        if not self.recognizer: return
        profile = self.recognizer.profile
        regions = dict(profile.get("regions", {}))
        board = profile.get("board_rect")
        if board: regions["棋盘"] = board
        for key, rect in profile.get("cell_regions", {}).items(): regions[f"格 {key}"] = rect
        self.canvas.regions = regions
        self.canvas.update()

    def layout_changed(self, *_):
        if self.recognizer:
            if self.ocr_worker and self.ocr_worker.isRunning():
                self.layout_choice.blockSignals(True)
                self.layout_choice.setCurrentIndex(self.layout_choice.findData(self.recognizer.profile.get("layout", "ring6x4")))
                self.layout_choice.blockSignals(False)
                self.capture_info.setText("请等待本次识别完成后再修改布局。"); return
            self.recognizer.set_board_layout(self.layout_choice.currentData())
            self.capture_source_changed()
            self.update_regions()

    def position_base_changed(self, *_):
        if self.recognizer:
            if self.ocr_worker and self.ocr_worker.isRunning():
                self.position_base.blockSignals(True)
                self.position_base.setCurrentIndex(self.position_base.findData(self.recognizer.profile.get("position_base", 0)))
                self.position_base.blockSignals(False)
                self.capture_info.setText("请等待本次识别完成后再修改编号方式。"); return
            self.recognizer.set_position_base(self.position_base.currentData())
            self.capture_source_changed()

    def learn_template(self):
        if self.last_image is None or not self.recognizer:
            QMessageBox.information(self, "需要截图", "先载入截图并框选整个棋盘。"); return
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.capture_info.setText("请等待本次识别完成后再学习模板。"); return
        index = self.learn_index.value()
        cell = next(c for c in self.state.cells if c.index == index)
        dialog = CellDialog(cell, self)
        dialog.setWindowTitle(f"标注截图中的牢房 {index:02d}")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            value = dialog.value()
            try:
                self.recognizer.learn_cell(self.last_image, index, value.species, value.rank, value.kind)
                self.capture_source_changed()
                self.capture_info.setText(f"已学习牢房 {index:02d}。可继续标注不同代号、品质、未开启及空牢房。")
            except Exception as exc: QMessageBox.warning(self, "样本保存失败", str(exc))

    def closeEvent(self, event):
        self.capture_timer.stop(); self.analysis_timer.stop(); self.cancel_search()
        for worker in list(self.workers): worker.wait(1500)
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.wait(5000)
        if any(w.isRunning() for w in self.workers) or (self.ocr_worker and self.ocr_worker.isRunning()):
            self.status.setText("正在结束本地计算，请稍后关闭…")
            QTimer.singleShot(1000, self.close)
            event.ignore(); return
        self.settings.save_strategy(self.config)
        event.accept()


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    if "--self-test" in sys.argv:
        from .selfcheck import run
        index = sys.argv.index("--self-test")
        report = sys.argv[index + 1] if len(sys.argv) > index + 1 else str(user_data_dir() / "self-test.json")
        sys.exit(run(report))
    app = QApplication(sys.argv)
    app.setApplicationName("Talos Local Assistant")
    app.setOrganizationName("LocalTools")
    app.setStyle("Fusion")
    configure_fonts(app)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
