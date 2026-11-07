import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--euvp_root", type=str, required=True, help="Path to EUVP root")
    parser.add_argument("--subset", type=str, default="paired", choices=["paired", "paired_synthetic", "paired_real"], help="Which paired subset to prepare")
    parser.add_argument("--out", type=str, required=True, help="Output dataset path with input/ and target/")
    args = parser.parse_args()
    euvp = Path(args.euvp_root)
    out = Path(args.out)
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "target").mkdir(parents=True, exist_ok=True)

    # Heuristic layout: EUVP/ paired/ train/ input, target (or similar)
    subset_dir = None
    for name in [args.subset, "Paired", "paired"]:
        p = euvp / name
        if p.exists():
            subset_dir = p
            break
    if subset_dir is None:
        raise FileNotFoundError("Could not find EUVP paired subset. Check --euvp_root and --subset.")

    # Look for input/ and target/ under subset (possibly under train/)
    def find_dir(root: Path, names):
        for n in names:
            p = root / n
            if p.exists():
                return p
        for sub in ["train", "Train", "training"]:
            for n in names:
                p = root / sub / n
                if p.exists():
                    return p
        return None

    src_inp = find_dir(subset_dir, ["input", "raw", "low" ] )
    src_tgt = find_dir(subset_dir, ["target", "gt", "high" ] )
    if src_inp is None or src_tgt is None:
        raise FileNotFoundError("Could not locate input/ and target/ in EUVP subset.")

    exts = {".jpg", ".png", ".jpeg"}
    num = 0
    for inp in sorted(src_inp.rglob("*")):
        if inp.suffix.lower() not in exts:
            continue
        rel = inp.relative_to(src_inp)
        tgt = (src_tgt / rel).with_suffix(".png")
        if not tgt.exists():
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
        raise RuntimeError("No pairs were found in EUVP. Check the folder structure.")
    print(f"Prepared {num} pairs at {out}")


if __name__ == "__main__":
    main()


