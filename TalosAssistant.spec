from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH)
datas = [(str(root / "data"), "data"), (str(root / "models"), "models"),
         (str(root / "README.md"), "."), (str(root / "THIRD_PARTY.md"), ".")]
if (root / "third_party_licenses").exists():
    datas += [(str(root / "third_party_licenses"), "third_party_licenses")]
datas += collect_data_files("rapidocr")
hidden = collect_submodules("rapidocr", filter=lambda name: not any(part in name for part in (".pytorch", ".torch", ".paddle", ".openvino", ".tensorrt", ".mnn")))
a = Analysis([str(root / "run.py")], pathex=[str(root)], binaries=[], datas=datas,
             hiddenimports=hidden, hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=["tkinter", "matplotlib", "scipy", "IPython", "PyQt5", "PyQt6", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"], noarchive=False)
# Qt uses the Windows system ICU API (unversioned exports). A Poppler directory
# on the build machine's PATH can make PyInstaller collect an incompatible ICU
# with suffixed _78 exports under the same DLL filename. Never ship that copy.
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in {"icuuc.dll", "icudt78.dll"}]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TalosAssistant", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TalosAssistant")
