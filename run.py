"""Portable entry point, including diagnostics before Qt is imported."""
import datetime
import os
from pathlib import Path
import sys
import traceback


def launch():
    if getattr(sys, "frozen", False):
        directory = Path(os.environ.get("TALOS_DATA_DIR", str(Path(sys.executable).parent / "user_data")))
        directory.mkdir(parents=True, exist_ok=True)
        # Windowed executables have no standard streams; some dependencies
        # expect a writable stream even when the product has no console.
        log = (directory / "startup.log").open("a", encoding="utf-8", buffering=1)
        if sys.stdout is None: sys.stdout = log
        if sys.stderr is None: sys.stderr = log
        print(f"\n{datetime.datetime.now().isoformat()} starting", file=log)
    try:
        if "--self-test" in sys.argv:
            # Write the starting report before the potentially failing Qt import.
            from talos_assistant.selfcheck import run
            index = sys.argv.index("--self-test")
            report = sys.argv[index + 1] if len(sys.argv) > index + 1 else str(Path(sys.executable).parent / "self-test.json")
            return run(report)
        from talos_assistant.app import main
        main()
        return 0
    except Exception:
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(launch())
