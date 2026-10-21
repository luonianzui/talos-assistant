"""Create a portable archive after the executable has passed its self-check."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
folder = root / "dist" / "TalosAssistant"
report_path = root / "artifacts" / "selfcheck-packaged.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
if not report.get("passed") or report.get("network_attempts") != 0:
    raise SystemExit("Run the packaged offline self-check successfully first.")
if report_path.stat().st_mtime < (folder / "TalosAssistant.exe").stat().st_mtime:
    raise SystemExit("The executable is newer than its verification report.")
for name in ("README.md", "THIRD_PARTY.md"):
    shutil.copy2(root / name, folder / name)
archive = root / "dist" / "TalosAssistant-Windows-x64.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if path.is_file() and "user_data" not in relative.parts and path.name.lower() not in {"icuuc.dll", "icudt78.dll", "icuuc.dll.unused"}:
            bundle.write(path, Path(folder.name) / relative)
with zipfile.ZipFile(archive) as bundle:
    failure = bundle.testzip()
    if failure:
        raise SystemExit(f"Archive integrity check failed: {failure}")
    files = len(bundle.infolist())
    # Keep a clean extracted copy for the final launch check and direct use.
    extracted = root / "dist" / "portable"
    if extracted.exists():
        raise SystemExit("dist/portable already exists; choose a new verification directory before extracting.")
    bundle.extractall(extracted)
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
archive.with_suffix(".sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
print(json.dumps({"archive": str(archive), "files": files, "bytes": archive.stat().st_size,
                  "sha256": digest}, indent=2))
