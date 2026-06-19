from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from typing import Iterator
import webbrowser

import psutil


FORBIDDEN_BROWSER_TOKENS = (
    "msedge",
    "microsoft-edge",
    "edge.exe",
    "cmd /c start",
    "cmd.exe /c start",
)


class NoGuiBrowserViolation(RuntimeError):
    """Raised when a headed browser launch is attempted during rendering."""


def _command_text(command: object) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command)


def assert_subprocess_command_safe(command: object) -> None:
    text = _command_text(command).lower().replace("\\", "/")
    if any(token in text for token in FORBIDDEN_BROWSER_TOKENS):
        raise NoGuiBrowserViolation(f"Forbidden GUI/Edge browser command: {_command_text(command)}")
    if "start-process" in text and any(suffix in text for suffix in (".svg", ".pdf", ".html", ".htm")):
        raise NoGuiBrowserViolation(f"Forbidden GUI file open command: {_command_text(command)}")


def assert_playwright_launch_safe(kwargs: dict[str, object]) -> None:
    if kwargs.get("headless") is not True:
        raise NoGuiBrowserViolation("Playwright may only be launched with headless=True.")
    executable_path = str(kwargs.get("executable_path", "")).lower()
    if "edge" in executable_path or "msedge" in executable_path:
        raise NoGuiBrowserViolation("Playwright must not use Microsoft Edge as its executable.")


def child_process_snapshot() -> set[int]:
    current = psutil.Process(os.getpid())
    return {child.pid for child in current.children(recursive=True) if child.is_running()}


def assert_no_orphan_children(before: set[int]) -> None:
    after = child_process_snapshot()
    leftovers = sorted(after - before)
    if leftovers:
        raise NoGuiBrowserViolation(f"Rendering left child processes running: {leftovers}")


@contextmanager
def no_gui_browser_guard() -> Iterator[dict[str, object]]:
    """Temporarily block GUI browser opens and unsafe subprocess launches."""

    before = child_process_snapshot()
    original_open = webbrowser.open
    original_startfile = getattr(os, "startfile", None)
    original_run = subprocess.run
    original_popen = subprocess.Popen

    def blocked_webbrowser_open(*args: object, **kwargs: object) -> bool:
        raise NoGuiBrowserViolation("webbrowser.open is forbidden during production rendering.")

    def blocked_startfile(path: object, *args: object, **kwargs: object) -> None:
        suffix = Path(str(path)).suffix.lower()
        if suffix in {".svg", ".pdf", ".html", ".htm"}:
            raise NoGuiBrowserViolation(f"os.startfile is forbidden for render artifacts: {path}")
        if original_startfile is None:
            raise AttributeError("os.startfile is unavailable")
        return original_startfile(path, *args, **kwargs)

    def guarded_run(command: object, *args: object, **kwargs: object):
        assert_subprocess_command_safe(command)
        return original_run(command, *args, **kwargs)

    class GuardedPopen(original_popen):  # type: ignore[misc, valid-type]
        def __init__(self, command: object, *args: object, **kwargs: object) -> None:
            assert_subprocess_command_safe(command)
            super().__init__(command, *args, **kwargs)

    webbrowser.open = blocked_webbrowser_open
    if original_startfile is not None:
        os.startfile = blocked_startfile  # type: ignore[attr-defined]
    subprocess.run = guarded_run  # type: ignore[assignment]
    subprocess.Popen = GuardedPopen  # type: ignore[assignment]
    try:
        yield {"child_processes_before": sorted(before)}
    finally:
        webbrowser.open = original_open
        if original_startfile is not None:
            os.startfile = original_startfile  # type: ignore[attr-defined]
        subprocess.run = original_run  # type: ignore[assignment]
        subprocess.Popen = original_popen  # type: ignore[assignment]
        assert_no_orphan_children(before)
