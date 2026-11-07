import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


URL = "https://github.com/zhangyhua/underwater-image-datasets/releases/download/v0.1/sample_blur_dark.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="Underwater-Image-Enhanncement-and-Restoration-main/datasets/sample_blur_dark")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    zip_path = out.with_suffix(".zip")
    print(f"Downloading to {zip_path} ...")
    urlretrieve(URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out.parent)
    print(f"Done. Dataset at {out}")


if __name__ == "__main__":
    main()


