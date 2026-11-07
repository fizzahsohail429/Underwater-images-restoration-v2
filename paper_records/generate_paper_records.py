"""Write the manuscript tables that can be filled from this repository.

Quality scores are left blank when no checkpoint or test images exist.
Parameter counts, memory, and latency are measured on this machine.
"""

import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import urllib.request
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset_catalog import (  # noqa: E402
    UIEB_CHALLENGE,
    UIEB_RAW,
    UIEB_REFERENCE,
    aquarium_root,
    source_rows,
)
from improved_underwater_restoration import (  # noqa: E402
    EVALUATED_SETTINGS,
    build_configuration,
    build_paper_variant,
    device,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASETS_ROOT = os.path.join(
    ROOT,
    "Underwater-Image-Enhanncement-and-Restoration-main",
    "datasets",
)
PREPARED_DATASETS = (
    ("UIEB", "UIEB_paired", "paired perceptual reference", "evaluation source"),
    ("EUVP", "EUVP_paired", "paired reference", "evaluation source"),
    ("UODD", "UODD_noref", "no reference", "intended held-out no-reference source"),
)
AQUARIUM_CANDIDATES = (
    os.path.join(ROOT, "aquarium_pretrain"),
    r"C:\Users\Lenovo\OneDrive\Documents\phd work\Underwater Image Codes\underwater_dataset1\aquarium_pretrain",
)
PUBLIC_REPO = "https://github.com/fizzahsohail429/Underwater-Image-Enhanncement-and-Restoration"


def list_files(directory):
    if not os.path.isdir(directory):
        return []
    found = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_id(filename):
    """Capture id before a Roboflow suffix such as _jpeg_jpg.rf.<hash>."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "_jpeg" in stem:
        return stem.split("_jpeg")[0]
    if stem.startswith("IMG_"):
        return "_".join(stem.split("_")[:2])
    return ""


def prepared_dataset_rows():
    """Count the input/target layouts created by the datasets preparation scripts."""
    rows = []
    identifiers = []
    for source, folder, reference_type, role in PREPARED_DATASETS:
        root = os.path.join(DATASETS_ROOT, folder)
        present = os.path.isdir(root)
        input_files = list_files(os.path.join(root, "input"))
        target_files = list_files(os.path.join(root, "target"))
        paired = 0
        target_names = {os.path.splitext(os.path.basename(path))[0] for path in target_files}
        for path in input_files:
            name = os.path.basename(path)
            has_target = os.path.splitext(name)[0] in target_names
            paired += int(has_target)
            identifiers.append({
                "source": source,
                "reference_type": reference_type,
                "split": "unsplit",
                "identifier": name,
                "sha256": file_sha256(path),
                "sequence_id": sequence_id(name),
                "role": role,
            })
        rows.append({
            "source": source,
            "reference_type": reference_type,
            "split": "unsplit",
            "directory": os.path.relpath(root, ROOT) if present else folder,
            "directory_present": present,
            "image_count": len(input_files),
            "paired_count": paired,
            "sequence_count": len({item["sequence_id"] for item in identifiers if item["source"] == source and item["sequence_id"]}),
            "role": role,
        })
    return rows, identifiers


def aquarium_root():
    for path in AQUARIUM_CANDIDATES:
        if os.path.isdir(os.path.join(path, "train", "images")):
            return path
    return AQUARIUM_CANDIDATES[0]


UIEB_SPLIT_SEED = 2026
UIEB_SPLIT_RATIOS = (0.8, 0.1, 0.1)


def assign_uieb_splits(filenames):
    """Put each UIEB still into one split. UIEB raw-890 is not a video sequence."""
    import numpy as np

    ordered = [filenames[i] for i in np.random.default_rng(UIEB_SPLIT_SEED).permutation(len(filenames))]
    n_train = int(round(len(ordered) * UIEB_SPLIT_RATIOS[0]))
    n_valid = int(round(len(ordered) * UIEB_SPLIT_RATIOS[1]))
    if n_train + n_valid > len(ordered):
        n_valid = len(ordered) - n_train
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train:n_train + n_valid],
        "test": ordered[n_train + n_valid:],
    }


def dataset_inventory():
    """Count every named source. The current comparison remains UIEB raw-890."""
    files = list_files(UIEB_RAW)
    names = [os.path.basename(path) for path in files]
    by_name = {os.path.basename(path): path for path in files}
    splits = assign_uieb_splits(sorted(names))
    reference_names = {os.path.splitext(os.path.basename(path))[0] for path in list_files(UIEB_REFERENCE)}
    rows = []
    identifiers = []
    hashes = defaultdict(list)
    for split, split_names in splits.items():
        paired = 0
        for name in split_names:
            path = by_name[name]
            digest = file_sha256(path)
            hashes[digest].append((split, name))
            has_reference = os.path.splitext(name)[0] in reference_names
            paired += int(has_reference)
            identifiers.append({
                "source": "UIEB",
                "reference_type": "raw underwater image",
                "split": split,
                "identifier": name,
                "sha256": digest,
                "sequence_id": os.path.splitext(name)[0],
                "has_reference": has_reference,
                "in_current_comparison": True,
            })
        rows.append({
            "source": "UIEB raw-890",
            "reference_type": "raw underwater image",
            "split": split,
            "directory": os.path.relpath(UIEB_RAW, ROOT),
            "directory_present": os.path.isdir(UIEB_RAW),
            "image_count": len(split_names),
            "paired_count": paired,
            "sequence_count": len(split_names),
            "role": "current comparison; still-image partition, seed 2026, ratios 0.8/0.1/0.1",
            "in_current_comparison": True,
        })
    aquarium_capture_ids = defaultdict(set)
    for item in source_rows():
        if item["source"] == "UIEB raw-890":
            continue
        split_name = item.get("split", "unsplit")
        rows.append({
            "source": item["source"],
            "reference_type": item["reference_type"],
            "split": split_name,
            "directory": os.path.relpath(item["directory"], ROOT) if os.path.isabs(item["directory"]) else item["directory"],
            "directory_present": item["directory_present"],
            "image_count": item["image_count"],
            "paired_count": item["paired_count"],
            "sequence_count": item["image_count"],
            "role": item["role"],
            "in_current_comparison": item["in_current_comparison"],
        })
        if item["source"] != "aquarium" or not item["directory_present"] or item["image_count"] == 0:
            continue
        for path in list_files(item["directory"]):
            name = os.path.basename(path)
            capture = sequence_id(name)
            aquarium_capture_ids[split_name].add(capture)
            identifiers.append({
                "source": "aquarium",
                "reference_type": item["reference_type"],
                "split": split_name,
                "identifier": name,
                "sha256": file_sha256(path),
                "sequence_id": capture,
                "has_reference": False,
                "in_current_comparison": False,
            })
    shared_captures = []
    aquarium_splits = [name for name in ("train", "valid", "test") if aquarium_capture_ids[name]]
    all_captures = set().union(*aquarium_capture_ids.values()) if aquarium_capture_ids else set()
    for capture in sorted(all_captures):
        owners = [name for name in aquarium_splits if capture in aquarium_capture_ids[name]]
        if len(owners) > 1:
            shared_captures.append({"sequence_id": capture, "splits": owners})
    duplicate_groups = [
        {"sha256": digest, "copies": copies}
        for digest, copies in hashes.items() if len({item[0] for item in copies}) > 1 or len(copies) > 1
    ]
    cross_split = [
        group for group in duplicate_groups
        if len({item[0] for item in group["copies"]}) > 1
    ]
    return rows, identifiers, {
        "source": "UIEB",
        "raw_directory": UIEB_RAW,
        "reference_directory": UIEB_REFERENCE,
        "reference_directory_present": os.path.isdir(UIEB_REFERENCE),
        "reference_image_count": len(reference_names),
        "split_seed": UIEB_SPLIT_SEED,
        "split_ratios": {"train": UIEB_SPLIT_RATIOS[0], "validation": UIEB_SPLIT_RATIOS[1], "test": UIEB_SPLIT_RATIOS[2]},
        "partition_unit": "one still image; raw-890 filenames are not video sequence ids",
        "method": "SHA-256 of raw-890 file bytes after a single assignment to train, validation, or test",
        "images_hashed": sum(item["source"] == "UIEB" for item in identifiers),
        "duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicates": cross_split,
        "capture_ids_in_more_than_one_split": shared_captures,
        "current_comparison": "UIEB archive/raw-890 only",
        "held_out_real_source": (
            "UODD is the held-out no-reference source. Its prepared folder has no images yet. "
            "EUVP_paired, UIEB reference-890, and UIEB challenge-60 are also empty. "
            "Aquarium captures are readable and are excluded from the UIEB comparison."
        ),
        "challenge_directory": UIEB_CHALLENGE,
        "aquarium_directory": aquarium_root(),
        "split_files": {split: names_in for split, names_in in splits.items()},
    }


def public_revision():
    request = urllib.request.Request(
        "https://api.github.com/repos/fizzahsohail429/Underwater-Image-Enhanncement-and-Restoration/commits/main",
        headers={"User-Agent": "eurm-paper-records", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "repository": PUBLIC_REPO,
            "remote_branch": "main",
            "remote_commit": payload.get("sha"),
            "remote_date": payload.get("commit", {}).get("committer", {}).get("date"),
            "local_git": False,
            "evaluated_revision": (
                "Local working tree. The decoder-width fix and these records are not identified "
                "by a git commit in this folder, so they are not the remote commit above."
            ),
        }
    except Exception as exc:
        return {
            "repository": PUBLIC_REPO,
            "remote_commit": None,
            "error": str(exc),
            "local_git": False,
            "evaluated_revision": "Local working tree, with no commit hash available in this folder.",
        }


def process_working_set():
    """Current Windows process working set, in bytes."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def measure_variant(name, batch_size, resolution, repeats):
    import threading

    model = build_paper_variant(name).to(device)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    sample = torch.randn(batch_size, 3, resolution, resolution, device=device)
    use_cuda = device.type == "cuda"
    peak_rss = {"bytes": process_working_set()}
    stop = {"value": False}

    def sample_rss():
        while not stop["value"]:
            peak_rss["bytes"] = max(peak_rss["bytes"], process_working_set())
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.inference_mode():
        model(sample)
        if use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    stop["value"] = True
    sampler.join()
    peak_bytes = torch.cuda.max_memory_allocated() if use_cuda else peak_rss["bytes"]
    del model, sample
    if use_cuda:
        torch.cuda.empty_cache()
    mean_s = sum(times) / len(times)
    memory_definition = (
        "torch.cuda.max_memory_allocated during the timed forwards"
        if use_cuda else
        "peak process working set during the forward, including loaded weights"
    )
    return {
        "variant": name,
        "checkpoint": "not trained; architecture only",
        "parameters": parameters,
        "resolution": resolution,
        "batch_size": batch_size,
        "device": str(device),
        "warmup_passes": 1,
        "timed_passes": repeats,
        "latency_ms_per_batch_mean": round(mean_s * 1000.0, 2),
        "latency_ms_per_image_mean": round(mean_s * 1000.0 / batch_size, 2),
        "latency_ms_per_batch_min": round(min(times) * 1000.0, 2),
        "latency_ms_per_batch_max": round(max(times) * 1000.0, 2),
        "peak_memory_mib": round(peak_bytes / (1024 * 1024), 2),
        "peak_memory_definition": memory_definition,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, identifiers, overlap = dataset_inventory()
    write_csv(os.path.join(OUT, "dataset_counts.csv"), rows)
    write_csv(os.path.join(OUT, "image_identifiers.csv"), identifiers or [{
        "source": "",
        "reference_type": "",
        "split": "",
        "identifier": "",
        "sha256": "",
        "sequence_id": "",
    }])
    with open(os.path.join(OUT, "overlap_check.json"), "w", encoding="utf-8") as handle:
        json.dump(overlap, handle, indent=2)

    revision = public_revision()
    config = build_configuration(device)
    config.update({
        "seed": 2026,
        "seed_covers": ["random", "numpy", "torch", "torch.cuda if available"],
        "checkpoint_rule": "Save the epoch whose mean validation loss is the lowest so far.",
        "checkpoint_path": "runs/evaluated/checkpoint.pt",
        "checkpoint_present": os.path.isfile(os.path.join(ROOT, "runs", "evaluated", "checkpoint.pt")),
        "repository": revision,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    })
    with open(os.path.join(OUT, "training_configuration.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    measurements = []
    errors = []
    for name in ("e-urm", "no-attention", "no-color", "capacity-matched"):
        try:
            print(f"measuring {name}", flush=True)
            measurements.append(measure_variant(name, batch_size=6, resolution=256, repeats=2))
            print(measurements[-1], flush=True)
        except Exception:
            errors.append({"variant": name, "error": traceback.format_exc()})
    external = [
        {
            "variant": "unprocessed input",
            "checkpoint": "none",
            "parameters": 0,
            "resolution": 256,
            "batch_size": 6,
            "device": str(device),
            "warmup_passes": 0,
            "timed_passes": 0,
            "latency_ms_per_batch_mean": "",
            "latency_ms_per_image_mean": "",
            "latency_ms_per_batch_min": "",
            "latency_ms_per_batch_max": "",
            "peak_memory_mib": "",
            "peak_memory_definition": "no network",
        },
        {
            "variant": "Water-Net",
            "checkpoint": "not in this repository",
            "parameters": "",
            "resolution": 256,
            "batch_size": 6,
            "device": str(device),
            "warmup_passes": "",
            "timed_passes": "",
            "latency_ms_per_batch_mean": "",
            "latency_ms_per_image_mean": "",
            "latency_ms_per_batch_min": "",
            "latency_ms_per_batch_max": "",
            "peak_memory_mib": "",
            "peak_memory_definition": "not measured; published checkpoint and code are not present",
        },
        {
            "variant": "FUnIE-GAN",
            "checkpoint": "not in this repository",
            "parameters": "",
            "resolution": 256,
            "batch_size": 6,
            "device": str(device),
            "warmup_passes": "",
            "timed_passes": "",
            "latency_ms_per_batch_mean": "",
            "latency_ms_per_image_mean": "",
            "latency_ms_per_batch_min": "",
            "latency_ms_per_batch_max": "",
            "peak_memory_mib": "",
            "peak_memory_definition": "not measured; published checkpoint and code are not present",
        },
        {
            "variant": "PCAFA-Net",
            "checkpoint": "not in this repository",
            "parameters": "",
            "resolution": 256,
            "batch_size": 6,
            "device": str(device),
            "warmup_passes": "",
            "timed_passes": "",
            "latency_ms_per_batch_mean": "",
            "latency_ms_per_image_mean": "",
            "latency_ms_per_batch_min": "",
            "latency_ms_per_batch_max": "",
            "peak_memory_mib": "",
            "peak_memory_definition": "not measured; published checkpoint and code are not present",
        },
    ]
    write_csv(os.path.join(OUT, "computational_cost.csv"), measurements + external)
    with open(os.path.join(OUT, "measurement_errors.json"), "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2)

    count_lines = "\n".join(
        f"| {row['source']} | {row['reference_type']} | {row['split']} | {row['image_count']} | {row['directory_present']} |"
        for row in rows
    )
    cost_lines = "\n".join(
        f"| {row['variant']} | {row['parameters']} | {row['peak_memory_mib']} | {row['latency_ms_per_batch_mean']} | {row['latency_ms_per_image_mean']} | {row['checkpoint']} |"
        for row in measurements + external
    )
    full = next((row for row in measurements if row["variant"] == "e-urm"), None)
    matched = next((row for row in measurements if row["variant"] == "capacity-matched"), None)
    ratio = ""
    if full and matched and full["parameters"]:
        ratio = f"{matched['parameters'] / full['parameters']:.3f}"

    write_markdown(os.path.join(OUT, "PAPER_TABLES.md"), f"""# Records for the Sensors manuscript

These files were generated from the local code and from measurements on this computer. Cells that require trained outputs or missing datasets are left empty rather than filled with assumed scores.

## Sample counts

| Source | Reference type | Split | Images | Directory present |
|---|---|---|---:|---|
{count_lines}

Image and sequence identifiers are in `image_identifiers.csv` ({len(identifiers)} rows). The `datasets` preparation folders are included even when they contain no images.

Overlap check: SHA-256 of aquarium file bytes. Images hashed: {overlap['images_hashed']}. Identical-byte groups: {overlap['duplicate_groups']}. Identical files in more than one split: {overlap['cross_split_duplicate_groups']}. Capture ids shared by more than one aquarium split: {len(overlap['capture_ids_in_more_than_one_split'])}.

Held-out test source: {overlap['held_out_real_source']}

## Training configuration

- Seed: 2026, applied to `random`, NumPy, and PyTorch before training.
- Input: 256 × 256. Batch size: {EVALUATED_SETTINGS['batch_size']}. Epochs configured: {EVALUATED_SETTINGS['epochs']}.
- Optimizer: AdamW, learning rate {EVALUATED_SETTINGS['learning_rate']}, weight decay {EVALUATED_SETTINGS['weight_decay']}.
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 5, mode min.
- Loss weights: L1 {EVALUATED_SETTINGS['loss_weights']['l1']}, MSE {EVALUATED_SETTINGS['loss_weights']['mse']}, Sobel {EVALUATED_SETTINGS['loss_weights']['sobel']}, degradation consistency {EVALUATED_SETTINGS['loss_weights']['degradation_consistency']}.
- Normalization: (pixel / 255 − 0.5) / 0.5, so network tensors lie in [−1, 1].
- Checkpoint rule: keep `runs/evaluated/checkpoint.pt` from the epoch with the lowest mean validation loss.
- Checkpoint file present: {config['checkpoint_present']}.
- Repository named in the manuscript: {PUBLIC_REPO}
- Remote commit lookup: {revision.get('remote_commit') or revision.get('error')}
- Evaluated revision: {revision['evaluated_revision']}

The full machine-readable copy is `training_configuration.json`.

## Metric definition

PSNR and SSIM are defined for later evaluation. This run did not score images, so the sample count is 0 and the uncertainty interval is undefined.

- Scaling: denormalize with `y = clamp(x * 0.5 + 0.5, 0, 1)` before either metric. The peak value is 1, so PSNR is in decibels on the unit interval.
- PSNR: for each image, MSE is the mean squared error over the three channels and all pixels. `PSNR = 10 log10(1 / MSE)`.
- SSIM: Wang et al. (2004), 11 × 11 Gaussian window with sigma 1.5, `C1 = (0.01)^2`, `C2 = (0.03)^2`, computed per channel and then averaged across channels and pixels. One SSIM value is kept per image.
- Averaging unit: the reported mean is the unweighted mean of the per-image scores. It is not a mean over pixels pooled across the dataset.
- Uncertainty: 95% Student-t interval of that per-image mean, `mean ± t_(n−1, 0.975) * s / sqrt(n)`, where n is the number of images in that source and split. The interval is reported only when n ≥ 2.
- UIQM and UCIQE, when implemented, are computed on the denormalized prediction only and are averaged per image. They are reported apart from PSNR and SSIM. This repository does not yet run the Panetta et al. (2016) or Yang and Sowmya (2015) implementations, so no UIQM or UCIQE values are included.

## Computational cost

Resolution 256 × 256, batch size 6, device `{device}`. One untimed warmup pass and two timed passes, in evaluation mode, without gradients. Processor: {platform.processor() or platform.platform()}. PyTorch threads: {torch.get_num_threads()}.

| Variant | Parameters | Peak memory (MiB) | Latency per batch (ms) | Latency per image (ms) | Checkpoint |
|---|---:|---:|---:|---:|---|
{cost_lines}

Peak memory is CUDA allocated memory when a GPU is used. On this run the definition is: {measurements[0]['peak_memory_definition'] if measurements else 'no successful measurement'}.

Capacity-matched control parameter ratio versus E-URM: {ratio or 'not available'}. The control replaces each multiscale attention block with one 3 × 3 convolution, batch normalization, and ReLU, and it removes the color module. The ratio is the measured parameter count, not a claim of an exact match.

Water-Net, FUnIE-GAN, and PCAFA-Net were not timed. Their code and checkpoints are not in this folder, so a shared-hardware comparison is not available.

## Ablations and baselines

| Name | What changes | Checkpoint |
|---|---|---|
| Unprocessed input | The degraded image is the output | none |
| E-URM | Multiscale attention and adaptive color correction | `runs/evaluated/checkpoint.pt`, absent until training |
| Capacity-matched control | `attention_mode=conv3x3`, color module removed | not trained |
| No attention | `attention_mode=none`; color module kept | not trained |
| No color correction | color module removed; attention kept | not trained |
| No degradation consistency | same network; loss weight on the consistency term set to 0 | not trained |
| No synthetic augmentation | same network; training uses the clean image as the input | not trained |

No quality table is written because those checkpoints have not been trained and the image directories contain no files.
""")
    print(f"wrote records in {OUT}")


def write_count_files():
    rows, identifiers, overlap = dataset_inventory()
    write_csv(os.path.join(OUT, "dataset_counts.csv"), rows)
    write_csv(os.path.join(OUT, "image_identifiers.csv"), identifiers)
    for split, names_in in overlap["split_files"].items():
        with open(os.path.join(OUT, f"uieb_{split}.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(names_in) + "\n")
    overlap_public = {key: value for key, value in overlap.items() if key != "split_files"}
    with open(os.path.join(OUT, "overlap_check.json"), "w", encoding="utf-8") as handle:
        json.dump(overlap_public, handle, indent=2)
    count_lines = "\n".join(
        f"| {row['source']} | {row['reference_type']} | {row['split']} | {row['image_count']} | {row['paired_count']} |"
        for row in rows
    )
    section = f"""## Sample counts

The code reads every source below. The saved image comparison is UIEB `archive/raw-890` only.

| Source | Reference type | Split | Images | Paired references |
|---|---|---|---:|---:|
{count_lines}

The UIEB partition is one still image per file, seed {overlap['split_seed']}, ratios 0.8 / 0.1 / 0.1. Identifiers are in `image_identifiers.csv` ({len(identifiers)} rows). Split lists are `uieb_train.txt`, `uieb_validation.txt`, and `uieb_test.txt`.

UIEB overlap check: SHA-256 of `raw-890`. Images hashed in that source: {overlap['images_hashed']}. Identical-byte groups inside UIEB: {overlap['duplicate_groups']}. Identical UIEB files in more than one split: {overlap['cross_split_duplicate_groups']}. Aquarium capture ids shared by more than one aquarium split: {len(overlap['capture_ids_in_more_than_one_split'])}.

Current visual comparison: {overlap['current_comparison']}. The ten-image panels are in `uieb_workflow10/`.

Held-out source: {overlap['held_out_real_source']}

Paired-reference PSNR and SSIM against `reference-890` are not inserted because that directory is not in the archive. UIQM and UCIQE are not computed.

"""
    md_path = os.path.join(OUT, "PAPER_TABLES.md")
    if os.path.isfile(md_path):
        with open(md_path, encoding="utf-8") as handle:
            text = handle.read()
        start = text.index("## Sample counts")
        end = text.index("## Training configuration")
        write_markdown(md_path, text[:start] + section + text[end:])
    print(f"counts written, identifiers={len(identifiers)}")
    return rows, overlap


def fill_memory():
    """Measure each variant in its own process and update the existing cost table."""
    import subprocess

    peaks = {}
    definition = ""
    for name in ("e-urm", "no-attention", "no-color", "capacity-matched"):
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--memory", name],
            check=True,
            capture_output=True,
            text=True,
        )
        line = [item for item in completed.stdout.splitlines() if item.startswith("{")][-1]
        payload = json.loads(line)
        peaks[payload["variant"]] = payload["peak_memory_mib"]
        definition = payload["definition"]
        print(payload, flush=True)

    csv_path = os.path.join(OUT, "computational_cost.csv")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["variant"] in peaks:
            row["peak_memory_mib"] = peaks[row["variant"]]
            row["peak_memory_definition"] = definition
    write_csv(csv_path, rows)

    md_path = os.path.join(OUT, "PAPER_TABLES.md")
    with open(md_path, encoding="utf-8") as handle:
        text = handle.read()
    for name, peak in peaks.items():
        text = text.replace(f"| {name} |", f"| {name} |", 1)
        text = text.replace(
            f"| {name} | {next(row['parameters'] for row in rows if row['variant'] == name)} | None |",
            f"| {name} | {next(row['parameters'] for row in rows if row['variant'] == name)} | {peak} |",
        )
    text = text.replace(
        "Peak memory is CUDA allocated memory when a GPU is used. On this run the definition is: CUDA is not available, so peak GPU memory was not measured.",
        f"Peak memory on this CPU run is the process working set. Definition: {definition}.",
    )
    write_markdown(md_path, text)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--memory":
        row = measure_variant(sys.argv[2], batch_size=6, resolution=256, repeats=1)
        print(json.dumps({
            "variant": row["variant"],
            "peak_memory_mib": row["peak_memory_mib"],
            "definition": row["peak_memory_definition"],
        }))
    elif len(sys.argv) > 1 and sys.argv[1] == "--fill-memory":
        fill_memory()
    elif len(sys.argv) > 1 and sys.argv[1] == "--counts-only":
        write_count_files()
    else:
        main()
