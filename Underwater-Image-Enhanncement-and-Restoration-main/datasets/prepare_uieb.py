import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uieb_root", type=str, required=True, help="Path to UIEB root")
    parser.add_argument("--out", type=str, required=True, help="Output dataset path with input/ and target/")
    args = parser.parse_args()
    uieb = Path(args.uieb_root)
    out = Path(args.out)
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "target").mkdir(parents=True, exist_ok=True)

    # Heuristic: UIEB commonly provides 'raw-890' and 'reference-890' folders (names may vary).
    # We search for common folder names.
    candidates_input = [
        uieb / "raw-890",
        uieb / "raw",
        uieb / "input",
    ]
    candidates_target = [
        uieb / "reference-890",
        uieb / "reference",
        uieb / "target",
    ]
    src_inp = next((p for p in candidates_input if p.exists()), None)
    src_tgt = next((p for p in candidates_target if p.exists()), None)
    if src_inp is None or src_tgt is None:
        raise FileNotFoundError("Could not find UIEB raw/reference folders. Please point --uieb_root to the correct path.")

    exts = {".jpg", ".png", ".jpeg"}
    num = 0
    for inp in sorted(src_inp.rglob("*")):
        if inp.suffix.lower() not in exts:
            continue
        rel = inp.relative_to(src_inp)
        tgt = (src_tgt / rel).with_suffix(".png")
        if not tgt.exists():
            # try other extensions
            alt = None
            for e in [".jpg", ".jpeg", ".png"]:
                cand = (src_tgt / rel).with_suffix(e)
                if cand.exists():
                    alt = cand
                    break
            if alt is None:
                continue
            tgt = alt
        shutil.copy2(inp, out / "input" / rel)
        (out / "target" / rel.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(tgt, (out / "target" / rel).with_suffix(".png"))
        num += 1
    if num == 0:
        raise RuntimeError("No pairs were found in UIEB. Check the folder structure.")
    print(f"Prepared {num} pairs at {out}")


if __name__ == "__main__":
    main()


