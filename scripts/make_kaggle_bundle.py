"""Zip what a remote GPU (Kaggle/Colab) needs: code, config, built items + their images. No raw data."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "erb2_bundle.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ["src/*.py", "configs/*.yaml", "PREREGISTRATION.md", "requirements.txt",
                    "data/processed/*/*.parquet", "data/processed/*/images/*.jpg", "results/*.jsonl"]:
        for f in ROOT.glob(pattern):
            if f.parent.name == "results" and (f.name.startswith("mock") or "_smoke" in f.name):
                continue  # test-only results are not needed remotely
            z.write(f, f.relative_to(ROOT).as_posix())
            n += 1
print(f"{out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
