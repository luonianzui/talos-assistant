"""Offline smoke check for the installed executable, using synthetic data only."""
from __future__ import annotations


def run(report_path: str) -> int:
    """Write a JSON report and adjacent preview.png; return 0 only on success.

    Invoke before QApplication is constructed (the executable's --self-test
    entry point). No real game or desktop pixels are captured by this check.
    """
    from dataclasses import replace
    import json
    import os
    from pathlib import Path
    import queue
    import socket
    import tempfile
    import threading
    import time
    import traceback
    from unittest.mock import patch

    started = time.monotonic()
    deadline = started + 28.0  # reserve the remaining two seconds for cleanup/report
    target = Path(report_path).expanduser().resolve()
    preview_path = target.parent / "preview.png"
    attempts: list[str] = []
    details: dict = {}
    report = {
        "passed": False,
        "details": details,
        "elapsed": 0.0,
        "network_attempts": 0,
        "limitations": [
            "仅验证合成演示局面、合成中文文本及离屏界面，不代表游戏规则或实际游戏识别准确率。",
            "不采集游戏、桌面或其他用户窗口，不上传任何数据。",
            "未验证真实游戏画面的字体、动画、遮挡、缩放与连续操作；仍需实机校准和回放验证。",
        ],
    }

    def write_report():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = target.with_name(target.name + ".tmp")
        temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary_report, target)

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("离线自检超过 28 秒工作预算（总预算 30 秒）。")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def forbidden(name):
        def refuse(*_args, **_kwargs):
            attempts.append(name)
            raise OSError("Offline self-check prohibits network connections")
        return refuse

    socket.socket.connect = forbidden("socket.connect")
    socket.socket.connect_ex = forbidden("socket.connect_ex")
    socket.create_connection = forbidden("socket.create_connection")
    old_platform = os.environ.get("QT_QPA_PLATFORM")
    old_directory = os.environ.get("TALOS_DATA_DIR")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    temporary_data = None
    window = None
    app = None
    ocr_thread = None
    try:
        temporary_data = tempfile.TemporaryDirectory(prefix="talos-selfcheck-")
        os.environ["TALOS_DATA_DIR"] = temporary_data.name
        # A false starting report survives an unexpected native-library crash.
        details["phase"] = "starting"
        write_report()
        from PySide6.QtWidgets import QApplication
        from .app import MainWindow
        from .storage import resource_dir
        from .theme import STYLE, configure_fonts

        check_time()
        app = QApplication.instance() or QApplication(["talos-selfcheck"])
        if app.platformName() != "offscreen":
            raise RuntimeError("请在创建 QApplication 之前调用 --self-test。")
        app.setApplicationName("Talos Offline Selfcheck")
        app.setStyle("Fusion")
        configure_fonts(app)
        app.setStyleSheet(STYLE)
        details["qt_platform"] = app.platformName()

        models_dir = resource_dir() / "models"
        model_files = {}
        for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
            path = models_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"打包资源缺少非空模型文件：{name}")
            model_files[name] = path.stat().st_size
        details["model_files"] = model_files
        details["models_present"] = True

        def refuse_capture(*_args, **_kwargs):
            raise AssertionError("Self-check must not capture a real window")

        # MainWindow ordinarily enumerates Windows at startup. Remove that
        # dependency here, so the preview cannot include unrelated window names.
        with patch("talos_assistant.capture.list_windows", return_value=[]), patch("talos_assistant.capture.capture_window", side_effect=refuse_capture):
            check_time()
            window = MainWindow()
            window.config = replace(window.config, quick_seconds=0.2, total_seconds=2.0)
            window.show()  # offscreen backend: renders without a desktop window
            app.processEvents()
            if window.capture_timer.isActive():
                raise AssertionError("自检不应启动自动截图。")
            details["real_capture_disabled"] = True

            def pump_until(predicate):
                while not predicate():
                    check_time()
                    app.processEvents()
                    time.sleep(0.01)
                app.processEvents()

            window.load_demo()
            original_state = window.state.fingerprint()
            window.analyze()
            pump_until(lambda: window.last_rec is not None and not window.workers)
            first = window.last_rec
            if first.state_id != original_state or first.mode != "ceiling" or not first.candidates or first.status in ("blocked", "cancelled"):
                raise AssertionError("合成局面没有得到有效的冲顶策略结果。")
            details["ceiling"] = {
                "status": first.status, "complete": first.complete,
                "candidates": len(first.candidates), "solver_seconds": round(first.elapsed, 4),
                "first_action": first.candidates[0].action.kind,
            }

            window.average_button.click()
            pump_until(lambda: window.last_rec is not None and window.last_rec.mode == "average" and not window.workers)
            second = window.last_rec
            if window.config.mode != "average" or window.state.fingerprint() != original_state or second.state_id != original_state:
                raise AssertionError("切换平均收益策略改变了棋盘或采用了旧局面结果。")
            if not second.candidates or second.status in ("blocked", "cancelled"):
                raise AssertionError("合成局面没有得到有效的平均收益结果。")
            details["average"] = {
                "status": second.status, "complete": second.complete,
                "candidates": len(second.candidates), "solver_seconds": round(second.elapsed, 4),
                "first_action": second.candidates[0].action.kind,
            }
            details["strategy_switch_preserved_state"] = True
            app.processEvents()
            if not window.grab().save(str(preview_path), "PNG") or not preview_path.is_file() or preview_path.stat().st_size < 1000:
                raise RuntimeError("无法保存离屏界面预览。")
            details["preview"] = str(preview_path)

            result_queue: queue.Queue = queue.Queue(maxsize=1)

            def check_ocr():
                try:
                    import numpy as np
                    from PIL import Image, ImageDraw, ImageFont
                    from .vision import LocalOCR

                    ocr = LocalOCR(models_dir)
                    if not ocr._initialize():
                        raise RuntimeError(ocr.error)
                    font_path = str(ocr.engine.cfg.Global.font_path)
                    font = ImageFont.truetype(font_path, 40)
                    expected = "绮彩3传奇巨象"
                    image = Image.new("RGB", (720, 150), "white")
                    ImageDraw.Draw(image).text((25, 35), "绮彩 3 传奇 巨象", font=font, fill="black")
                    ocr_started = time.monotonic()
                    lines = ocr.read(np.asarray(image)[:, :, ::-1].copy())
                    text = "".join(line.text for line in lines).replace(" ", "")
                    if expected not in text:
                        raise AssertionError(f"合成中文识别不符合预期：{text!r}；{ocr.error}")
                    if ocr.engine.cfg.EngineConfig.onnxruntime.use_cuda or ocr.engine.cfg.EngineConfig.onnxruntime.use_dml:
                        raise AssertionError("自检 OCR 未明确采用 CPU 配置。")
                    result_queue.put({"passed": True, "text": text, "cpu_only": True,
                                      "minimum_confidence": round(min(line.confidence for line in lines), 4),
                                      "inference_seconds": round(time.monotonic() - ocr_started, 4)})
                except Exception as exc:
                    result_queue.put({"passed": False, "error": f"{type(exc).__name__}: {exc}"})

            ocr_thread = threading.Thread(target=check_ocr, name="talos-selfcheck-ocr", daemon=True)
            ocr_thread.start()
            pump_until(lambda: not result_queue.empty())
            ocr_result = result_queue.get_nowait()
            ocr_thread.join(timeout=0.1)
            details["ocr"] = ocr_result
            if not ocr_result["passed"]:
                raise RuntimeError(ocr_result["error"])
            if attempts:
                raise AssertionError("离线自检期间发生了被拒绝的网络尝试。")
            check_time()
            details["phase"] = "complete"
            report["passed"] = True
    except Exception as exc:
        details["phase"] = "failed"
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=8)
        report["passed"] = False
    finally:
        if window is not None:
            try:
                window.capture_timer.stop()
                window.analysis_timer.stop()
                window.cancel_search()
                # QThreads are cooperative; all ordinary demo jobs have already
                # completed. Bound waiting here on failure paths.
                for worker in list(window.workers):
                    worker.wait(100)
                if not any(worker.isRunning() for worker in window.workers):
                    window.close()
                else:
                    window.hide()
                    report["passed"] = False
                    details["cleanup_error"] = "求解线程未在清理预算内退出。"
                if app is not None:
                    app.processEvents()
            except Exception as exc:
                report["passed"] = False
                details["cleanup_error"] = str(exc)
        # A timed-out native OCR operation must remain unable to connect while
        # the --self-test process exits; do not restore access underneath it.
        if ocr_thread is None or not ocr_thread.is_alive():
            socket.socket.connect = original_connect
            socket.socket.connect_ex = original_connect_ex
            socket.create_connection = original_create_connection
        else:
            details["network_guard_left_active"] = True
        for name, previous in (("QT_QPA_PLATFORM", old_platform), ("TALOS_DATA_DIR", old_directory)):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        try:
            if temporary_data is not None:
                temporary_data.cleanup()
        except OSError as exc:
            details["temporary_data_cleanup"] = str(exc)
        report["elapsed"] = round(time.monotonic() - started, 4)
        report["network_attempts"] = len(attempts)
        if attempts:
            details["blocked_network_apis"] = attempts
            report["passed"] = False
        if report["elapsed"] > 30:
            report["passed"] = False
            details["time_budget_exceeded"] = True
        try:
            write_report()
        except OSError:
            return 1
    return 0 if report["passed"] else 1
