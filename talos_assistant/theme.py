def configure_fonts(app):
    """The offscreen Qt platform does not enumerate Windows system fonts."""
    import os
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase
    font_root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf", "seguisb.ttf"):
        path = font_root / name
        if path.exists(): QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont("Microsoft YaHei UI", 10))


STYLE = """
QWidget { background: #101723; color: #dde6ef; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 13px; }
QMainWindow { background: #101723; }
QLabel#eyebrow { color: #65ddc6; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
QLabel#title { font-size: 26px; font-weight: 700; color: #f4f8fc; }
QLabel#muted { color: #91a0b5; font-size: 12px; }
QLabel#section { font-size: 16px; font-weight: 700; color: #eef3f9; }
QLabel#badge { background: #1b3b39; color: #7ae6cd; border-radius: 6px; padding: 7px 12px; }
QFrame#panel { background: #172131; border: 1px solid #2a3648; border-radius: 12px; }
QFrame#panel QLabel { background: transparent; }
QPushButton { background: #202d40; border: 1px solid #33435a; border-radius: 7px; padding: 8px 13px; }
QPushButton:hover { background: #2b3c53; border-color: #65d9c2; }
QPushButton:pressed { background: #38516a; }
QPushButton:disabled { background: #18202c; color: #607087; border-color: #263247; }
QPushButton#primary { background: #65ddc6; color: #092723; border: none; font-weight: 700; }
QPushButton#primary:hover { background: #88efdb; }
QPushButton#strategy { min-height: 31px; font-weight: 600; padding: 7px 17px; }
QPushButton#strategy:checked { color: #78e6cf; border: 1px solid #65ddc6; background: #1b383a; }
QPushButton#cell { text-align: left; padding: 7px; font-weight: 600; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #101a29; border: 1px solid #33435a; border-radius: 5px; padding: 5px 7px; min-height: 22px; }
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox QAbstractItemView { background: #1c2a3e; selection-background-color: #2b5e5b; }
QTabWidget::pane { border: 0; }
QTabBar::tab { padding: 11px 21px; color: #8e9fb5; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #78e6cf; border-bottom: 2px solid #65ddc6; }
QTextBrowser, QPlainTextEdit { background: #111c2a; border: 1px solid #2b394d; border-radius: 7px; padding: 8px; selection-background-color: #265d59; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #52627a; border-radius: 4px; background: #182535; }
QCheckBox::indicator:checked { background: #65ddc6; border-color: #65ddc6; }
QToolTip { background: #23344b; color: #edf4fc; border: 1px solid #526680; padding: 5px; }
QScrollBar:vertical { background: #101723; width: 9px; }
QScrollBar::handle:vertical { background: #35455c; border-radius: 4px; min-height: 28px; }
QProgressBar { border: 0; background: #223146; border-radius: 3px; max-height: 5px; }
QProgressBar::chunk { background: #65ddc6; border-radius: 3px; }
QStatusBar { color: #8d9db2; }
QGroupBox { border: 1px solid #2c3c52; border-radius: 8px; margin-top: 12px; padding-top: 16px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
"""
