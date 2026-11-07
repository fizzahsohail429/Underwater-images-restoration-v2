"""Sources named by the manuscript and by the dataset preparation scripts.

Every source is readable through this module. The current visual comparison uses
only UIEB archive/raw-890. Other folders are counted even when they contain no
images, and they are not copied into that comparison.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PREPARED_ROOT = os.path.join(
    ROOT,
    "Underwater-Image-Enhanncement-and-Restoration-main",
    "datasets",
)
AQUARIUM_CANDIDATES = (
    os.path.join(ROOT, "aquarium_pretrain"),
    r"C:\Users\Lenovo\OneDrive\Documents\phd work\Underwater Image Codes\underwater_dataset1\aquarium_pretrain",
)
UIEB_RAW = os.path.join(ROOT, "archive", "UIEB")
EUVP_RAW = os.path.join(ROOT, "archive", "EUVP", "Paired", "underwater_dark", "trainA")
UIEB_REFERENCE = os.path.join(ROOT, "archive", "reference-890")
UIEB_CHALLENGE = os.path.join(ROOT, "archive", "challenge-60")


def list_images(directory):
    if not directory or not os.path.isdir(directory):
        return []
    found = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def aquarium_root():
    for path in AQUARIUM_CANDIDATES:
        if os.path.isdir(os.path.join(path, "train", "images")):
            return path
    return ""


def prepared_counts(folder):
    root = os.path.join(PREPARED_ROOT, folder)
    input_files = list_images(os.path.join(root, "input"))
    target_files = list_images(os.path.join(root, "target"))
    target_stems = {os.path.splitext(os.path.basename(path))[0] for path in target_files}
    paired = sum(
        os.path.splitext(os.path.basename(path))[0] in target_stems
        for path in input_files
    )
    return {
        "directory": root,
        "directory_present": os.path.isdir(root),
        "image_count": len(input_files),
        "paired_count": paired,
        "paths": input_files,
    }


def source_rows():
    """One description per source. Counts come from the files on disk."""
    rows = []
    raw_count = len(list_images(UIEB_RAW))
    reference_count = len(list_images(UIEB_REFERENCE))
    challenge_count = len(list_images(UIEB_CHALLENGE))
    rows.append({
        "source": "UIEB raw-890",
        "reference_type": "raw underwater image",
        "role": "current training source and the only visual comparison",
        "directory": UIEB_RAW,
        "directory_present": os.path.isdir(UIEB_RAW),
        "image_count": raw_count,
        "paired_count": 0,
        "in_current_comparison": True,
    })
    rows.append({
        "source": "UIEB reference-890",
        "reference_type": "perceptual reference",
        "role": "paired target for raw-890 when the files are present",
        "directory": UIEB_REFERENCE,
        "directory_present": os.path.isdir(UIEB_REFERENCE),
        "image_count": reference_count,
        "paired_count": reference_count,
        "in_current_comparison": False,
    })
    rows.append({
        "source": "UIEB challenge-60",
        "reference_type": "no satisfactory reference",
        "role": "visual and no-reference assessment when the files are present",
        "directory": UIEB_CHALLENGE,
        "directory_present": os.path.isdir(UIEB_CHALLENGE),
        "image_count": challenge_count,
        "paired_count": 0,
        "in_current_comparison": False,
    })
    prepared = (
        ("UIEB_paired", "paired perceptual reference", "prepared UIEB pairs"),
        ("EUVP_paired", "paired reference", "EUVP paired subset, kept separate from UIEB"),
        ("UODD_noref", "no reference", "held-out no-reference source"),
        ("UTDAC2020_noref", "no reference", "supported no-reference source"),
        ("RUOD_noref", "no reference", "supported no-reference source"),
        ("UDD_noref", "no reference", "supported no-reference source"),
    )
    for folder, reference_type, role in prepared:
        counts = prepared_counts(folder)
        rows.append({
            "source": folder,
            "reference_type": reference_type,
            "role": role,
            "directory": counts["directory"],
            "directory_present": counts["directory_present"],
            "image_count": counts["image_count"],
            "paired_count": counts["paired_count"],
            "in_current_comparison": False,
        })
    aquarium = aquarium_root()
    if aquarium:
        for split in ("train", "valid", "test"):
            paths = list_images(os.path.join(aquarium, split, "images"))
            rows.append({
                "source": "aquarium",
                "reference_type": "capture image, no enhancement reference",
                "role": "readable extra source; excluded from the UIEB comparison",
                "directory": os.path.join(aquarium, split, "images"),
                "directory_present": True,
                "image_count": len(paths),
                "paired_count": 0,
                "in_current_comparison": False,
                "split": split,
            })
    else:
        rows.append({
            "source": "aquarium",
            "reference_type": "capture image, no enhancement reference",
            "role": "readable extra source; excluded from the UIEB comparison",
            "directory": AQUARIUM_CANDIDATES[0],
            "directory_present": False,
            "image_count": 0,
            "paired_count": 0,
            "in_current_comparison": False,
            "split": "absent",
        })
    return rows


def uieb_raw_paths():
    return list_images(UIEB_RAW)


def enhancement_training_paths():
    """Image paths that belong in the current enhancement run.

    UIEB raw-890 is included. Prepared UIEB and EUVP pairs are included when
    their input folders contain files. Aquarium and the no-reference sets are
    readable through source_rows, and they stay out of this list.
    """
    paths = list(uieb_raw_paths())
    for folder in ("UIEB_paired", "EUVP_paired"):
        counts = prepared_counts(folder)
        paths.extend(counts["paths"])
    return paths
