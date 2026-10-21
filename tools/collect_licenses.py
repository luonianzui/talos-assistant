from importlib.metadata import distributions
from pathlib import Path
import json

root = Path(__file__).resolve().parent.parent
target = root / "third_party_licenses"
target.mkdir(exist_ok=True)
records = []
for distribution in distributions():
    name = distribution.metadata.get("Name", "unknown")
    records.append({"name": name, "version": distribution.version, "homepage": distribution.metadata.get("Home-page", ""), "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License", "")})
    for item in distribution.files or []:
        if any(word in str(item).lower() for word in ("license", "copying", "notice")) and str(item).endswith((".txt", ".md", "LICENSE", "COPYING", "NOTICE")):
            path = distribution.locate_file(item)
            if path.is_file() and path.stat().st_size < 500_000:
                out = target / name / Path(str(item)).name
                out.parent.mkdir(exist_ok=True)
                if not out.exists(): out.write_bytes(path.read_bytes())
(target / "packages.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Recorded {len(records)} installed distributions")
