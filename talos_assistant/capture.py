"""Read visible Windows client pixels; no game memory, hooks or input injection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes
from ctypes import wintypes
import sys

import cv2
import numpy as np


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]  # desktop client-area left, top, right, bottom


class CaptureError(RuntimeError):
    pass


def _user32():
    if sys.platform != "win32":
        raise CaptureError("自动窗口捕获仅支持 Windows；仍可导入截图。")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    # Thread-local awareness works even when Qt has already set process DPI mode.
    if hasattr(user32, "SetThreadDpiAwarenessContext"):
        user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    for name in ("IsWindow", "IsWindowVisible", "IsIconic", "GetWindowTextLengthW"):
        getattr(user32, name).argtypes = [wintypes.HWND]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    return user32


def _client_rect(user32, hwnd: int) -> tuple[int, int, int, int]:
    rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise CaptureError("无法读取窗口客户区；请重新选择游戏窗口。")
    return origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom


def list_windows() -> list[WindowInfo]:
    if sys.platform != "win32":
        return []
    user32 = _user32()
    result: list[WindowInfo] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def collect(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        try:
            rect = _client_rect(user32, hwnd)
            if rect[2] - rect[0] >= 80 and rect[3] - rect[1] >= 80:
                result.append(WindowInfo(int(hwnd), buf.value, rect))
        except CaptureError:
            pass
        return True

    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows(collect, 0)
    return sorted(result, key=lambda w: (not any(t in w.title.lower() for t in ("torchlight", "火炬")), w.title.lower()))


def capture_window(hwnd: int) -> np.ndarray:
    """Capture the visible client rectangle in BGR; occluding windows remain visible.

    The caller must keep the game visible and its own UI outside the selected
    rectangle. This deliberately does not claim to capture a hidden game window.
    """
    user32 = _user32()
    if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
        raise CaptureError("游戏窗口已关闭或不可见，请重新选择。")
    if user32.IsIconic(hwnd):
        raise CaptureError("游戏窗口已最小化，请恢复窗口或无边框模式。")
    left, top, right, bottom = _client_rect(user32, hwnd)
    if right <= left or bottom <= top:
        raise CaptureError("游戏客户区为空，暂时无法捕获。")
    import mss

    with mss.mss() as grabber:
        desktop = grabber.monitors[0]
        if (left < desktop["left"] or top < desktop["top"] or
                right > desktop["left"] + desktop["width"] or bottom > desktop["top"] + desktop["height"]):
            raise CaptureError("游戏窗口有部分移出屏幕，请让完整客户区保持可见。")
        pixels = np.asarray(grabber.grab({"left": left, "top": top, "width": right - left, "height": bottom - top}))
    image = np.ascontiguousarray(pixels[:, :, :3])
    if float(image.mean()) < 1.0 and float(image.std()) < 1.0:
        raise CaptureError("捕获到黑屏；请切换游戏为窗口或无边框模式。")
    return image


def load_image(path: str | Path) -> np.ndarray:
    """Open Unicode Windows paths and normalize to a three-channel BGR image."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise CaptureError(f"读取截图失败：{exc}") from exc
    if image is None or not image.size:
        raise CaptureError("文件不是可读取的截图（支持 PNG、JPG、WEBP 等）。")
    return image
