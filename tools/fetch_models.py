"""Build-time only: fetch pinned OCR models from RapidAI's official registry."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/"
FILES = {
    "det.onnx": (BASE + "det/ch_PP-OCRv4_det_mobile.onnx", "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9"),
    "rec.onnx": (BASE + "rec/ch_PP-OCRv4_rec_mobile.onnx", "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b"),
    "cls.onnx": (BASE + "cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    "keys.txt": ("https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt", None),
}


def fetch(item):
    name, (url, expected) = item
    destination = ROOT / "models" / name
    if destination.exists() and expected and sha256(destination.read_bytes()).hexdigest() == expected:
        return {"file": name, "url": url, "sha256": expected}
    request = Request(url, headers={"User-Agent": "Talos-Assistant-Build/0.1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                data = response.read()
            digest = sha256(data).hexdigest()
            if expected and digest != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
            break
        except Exception:
            if attempt == 2:
                raise
            print(f"Retrying interrupted download: {name}", flush=True)
    destination.write_bytes(data)
    print(f"Downloaded and verified {name}: {len(data)} bytes", flush=True)
    return {"file": name, "url": url, "sha256": digest}


if __name__ == "__main__":
    (ROOT / "models").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(fetch, FILES.items()))
    (ROOT / "models" / "provenance.json").write_text(json.dumps({"registry": "https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml", "models": manifest}, indent=2), encoding="utf-8")
