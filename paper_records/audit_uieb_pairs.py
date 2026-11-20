"""Check the locked UIEB pairs, exact file duplicates, and external folders.

This does not train and does not read the test scores. Scene identity is not
inferred from filenames.
"""

import hashlib
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "archive", "UIEB")
REFERENCE = os.path.join(ROOT, "archive", "reference-890")
LISTS = {
    "train": os.path.join(ROOT, "paper_records", "uieb_train.txt"),
    "validation": os.path.join(ROOT, "paper_records", "uieb_validation.txt"),
    "test": os.path.join(ROOT, "paper_records", "uieb_test.txt"),
}
CHECKPOINT = os.path.join(ROOT, "runs", "uieb_reference", "checkpoints", "best_val_psnr.pt")
OUTPUT = os.path.join(ROOT, "paper_records", "uieb_pair_audit.json")


def read_list(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_names():
    found = defaultdict(list)
    for name in os.listdir(REFERENCE):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            found[os.path.splitext(name)[0]].append(name)
    return found


def count_images(directory):
    total = 0
    for _dirpath, _dirnames, filenames in os.walk(directory):
        total += sum(name.lower().endswith((".png", ".jpg", ".jpeg")) for name in filenames)
    return total


def duplicate_groups(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["image_id"])
    return [
        {"sha256": digest, "image_ids": ids}
        for digest, ids in grouped.items()
        if len(ids) > 1
    ]


def main():
    lists = {name: read_list(path) for name, path in LISTS.items()}
    sets = {name: set(ids) for name, ids in lists.items()}
    references = reference_names()
    rows = []
    missing = []
    for split, image_ids in lists.items():
        for image_id in image_ids:
            stem = os.path.splitext(image_id)[0]
            raw_path = os.path.join(RAW, image_id)
            matches = references.get(stem, [])
            if not os.path.isfile(raw_path) or len(matches) != 1:
                missing.append({
                    "split": split,
                    "image_id": image_id,
                    "raw_present": os.path.isfile(raw_path),
                    "reference_matches": len(matches),
                })
                continue
            reference_path = os.path.join(REFERENCE, matches[0])
            rows.append({
                "split": split,
                "image_id": image_id,
                "reference_name": matches[0],
                "raw_sha256": sha256(raw_path),
                "reference_sha256": sha256(reference_path),
            })
    raw_duplicates = duplicate_groups(rows, "raw_sha256")
    reference_duplicates = duplicate_groups(rows, "reference_sha256")
    cross_split_raw = [
        item for item in raw_duplicates
        if len({row["split"] for row in rows if row["image_id"] in item["image_ids"]}) > 1
    ]
    euvp = os.path.join(ROOT, "archive", "EUVP", "Paired")
    report = {
        "counts": {name: len(ids) for name, ids in lists.items()},
        "unique_filenames": len(set().union(*sets.values())),
        "filename_overlap": {
            "train_validation": len(sets["train"] & sets["validation"]),
            "train_test": len(sets["train"] & sets["test"]),
            "validation_test": len(sets["validation"] & sets["test"]),
        },
        "pairs_checked": len(rows),
        "missing_or_unmatched_pairs": missing,
        "exact_duplicate_raw_groups": raw_duplicates,
        "exact_duplicate_reference_groups": reference_duplicates,
        "exact_duplicate_raw_groups_across_splits": cross_split_raw,
        "scene_grouping": "Filenames do not contain a scene or sequence identifier, so grouped resampling was not applied.",
        "external_sources": {
            "challenge-60": {"present": os.path.isdir(os.path.join(ROOT, "archive", "challenge-60"))},
            "EUVP_paired": {
                "present": os.path.isdir(euvp),
                "image_count": count_images(euvp) if os.path.isdir(euvp) else 0,
                "reference_folder_present": os.path.isdir(os.path.join(euvp, "underwater_dark", "trainB")),
            },
        },
    }
    if os.path.isfile(CHECKPOINT):
        report["selected_checkpoint_sha256"] = sha256(CHECKPOINT)
        report["selected_checkpoint"] = "runs/uieb_reference/checkpoints/best_val_psnr.pt"
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "pairs_checked": report["pairs_checked"],
        "missing": len(missing),
        "raw_duplicate_groups": len(raw_duplicates),
        "reference_duplicate_groups": len(reference_duplicates),
        "cross_split_raw_duplicate_groups": len(cross_split_raw),
        "checkpoint_recorded": "selected_checkpoint_sha256" in report,
    }, indent=2))


if __name__ == "__main__":
    main()
