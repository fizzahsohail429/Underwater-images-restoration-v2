import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--udd_root", type=str, required=True, help="Path to UDD root")
    parser.add_argument("--out", type=str, required=True, help="Output dataset path with input/")
    args = parser.parse_args()
    root = Path(args.udd_root)
    out = Path(args.out)
    (out / "input").mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".png", ".jpeg"}
    num = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            shutil.copy2(p, out / "input" / p.name)
            num += 1
    if num == 0:
        raise RuntimeError("No images found in UDD root.")
    print(f"Prepared {num} images at {out / 'input'}")


if __name__ == "__main__":
    main()


