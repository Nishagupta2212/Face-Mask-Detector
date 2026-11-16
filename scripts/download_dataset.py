"""Download a public face-mask dataset into ./dataset/<class>/.

Default source: prajnasb/observations (GitHub) - ~1,376 images,
classes `with_mask` and `without_mask`.

You can also use any dataset laid out as dataset/<class_name>/*.jpg, e.g.
Kaggle "omkargurav/face-mask-dataset" (7.5k images) or a 3-class set with
`mask_weared_incorrect`. The training script picks up classes from folder names.
"""
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR  # noqa: E402

URL = "https://github.com/prajnasb/observations/archive/refs/heads/master.zip"
PREFIX = "observations-master/experiements/data/"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"Dataset already present at {DATA_DIR} - delete it to re-download.")
        return

    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        payload = resp.read()
    print(f"Downloaded {len(payload) / 1e6:.1f} MB, extracting...")

    counts: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if not name.startswith(PREFIX) or name.endswith("/"):
                continue
            rel = Path(name[len(PREFIX):])
            if len(rel.parts) != 2 or rel.suffix.lower() not in IMAGE_EXTS:
                continue
            cls, fname = rel.parts
            out = DATA_DIR / cls / fname
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[cls] = counts.get(cls, 0) + 1

    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n} images")
    print(f"Dataset ready at {DATA_DIR}")


if __name__ == "__main__":
    main()
