"""Train the six manuscript variants on the locked UIEB raw-890 split.

The target of every score is the raw image. The test input is the
filename-seeded synthetic degradation. reference-890 is not used, and these
scores are not paired-reference UIEB results.
"""

import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from improved_underwater_restoration import (  # noqa: E402
    EVALUATED_SETTINGS,
    ImprovedUnderwaterDataset,
    PhysicsInformedLoss,
    build_configuration,
    build_paper_variant,
    create_improved_transforms,
    set_reproducibility_seed,
    train_improved_model,
)

RAW_DIR = os.path.join(ROOT, "archive", "raw-890")
RECORDS = os.path.join(ROOT, "paper_records")
SPLIT_FILES = {
    "train": os.path.join(RECORDS, "uieb_train.txt"),
    "valid": os.path.join(RECORDS, "uieb_validation.txt"),
    "test": os.path.join(RECORDS, "uieb_test.txt"),
}
RESULTS_PATH = os.path.join(RECORDS, "variant_results.json")
PER_IMAGE_CSV = os.path.join(RECORDS, "quality_per_image.csv")
SUMMARY_CSV = os.path.join(RECORDS, "quality_summary.csv")
TABLES_PATH = os.path.join(RECORDS, "PAPER_TABLES.md")

# Two-sided 95% Student-t critical value at df = 88 (n = 89).
T_CRIT_DF88 = 1.987289865

VARIANTS = [
    {
        "key": "e-urm",
        "method": "E-URM",
        "architecture": "e-urm",
        "consistency_weight": 0.1,
        "apply_synthetic_degradation": True,
        "output_dir": os.path.join(ROOT, "runs", "evaluated"),
        "change": "Multiscale attention and adaptive color correction",
    },
    {
        "key": "capacity-matched",
        "method": "Capacity-matched control",
        "architecture": "capacity-matched",
        "consistency_weight": 0.1,
        "apply_synthetic_degradation": True,
        "output_dir": os.path.join(ROOT, "runs", "ablations", "capacity-matched"),
        "change": "attention_mode=conv3x3, color module removed",
    },
    {
        "key": "no-attention",
        "method": "No attention",
        "architecture": "no-attention",
        "consistency_weight": 0.1,
        "apply_synthetic_degradation": True,
        "output_dir": os.path.join(ROOT, "runs", "ablations", "no-attention"),
        "change": "attention_mode=none; color module kept",
    },
    {
        "key": "no-color",
        "method": "No color correction",
        "architecture": "no-color",
        "consistency_weight": 0.1,
        "apply_synthetic_degradation": True,
        "output_dir": os.path.join(ROOT, "runs", "ablations", "no-color"),
        "change": "color module removed; attention kept",
    },
    {
        "key": "no-degradation-consistency",
        "method": "No degradation consistency",
        "architecture": "e-urm",
        "consistency_weight": 0.0,
        "apply_synthetic_degradation": True,
        "output_dir": os.path.join(ROOT, "runs", "ablations", "no-degradation-consistency"),
        "change": "same network; loss weight on the consistency term set to 0",
    },
    {
        "key": "no-synthetic-augmentation",
        "method": "No synthetic augmentation",
        "architecture": "e-urm",
        "consistency_weight": 0.1,
        "apply_synthetic_degradation": False,
        "output_dir": os.path.join(ROOT, "runs", "ablations", "no-synthetic-augmentation"),
        "change": "same network; training uses the clean image as the input",
    },
]


def split_paths(name):
    with open(SPLIT_FILES[name], encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    paths = []
    for filename in names:
        path = os.path.join(RAW_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def gaussian_window(window_size=11, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] * kernel[None, :]
    return window.view(1, 1, window_size, window_size)


def denormalize(tensor):
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def psnr_per_image(prediction, target):
    scores = []
    for pred, truth in zip(prediction, target):
        mse = torch.mean((pred - truth) ** 2).item()
        scores.append(99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse))
    return scores


def ssim_per_image(prediction, target, window):
    channel_window = window.to(prediction.device)
    scores = []
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    for pred, truth in zip(prediction, target):
        channel_scores = []
        for channel in range(pred.shape[0]):
            x = pred[channel:channel + 1].unsqueeze(0)
            y = truth[channel:channel + 1].unsqueeze(0)
            mu_x = F.conv2d(x, channel_window, padding=0)
            mu_y = F.conv2d(y, channel_window, padding=0)
            sigma_x = F.conv2d(x * x, channel_window, padding=0) - mu_x ** 2
            sigma_y = F.conv2d(y * y, channel_window, padding=0) - mu_y ** 2
            sigma_xy = F.conv2d(x * y, channel_window, padding=0) - mu_x * mu_y
            numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
            channel_scores.append(torch.mean(numerator / denominator).item())
        scores.append(float(np.mean(channel_scores)))
    return scores


def interval(values):
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, None, None
    variance = float(np.sum((np.asarray(values) - mean) ** 2) / (n - 1))
    half = T_CRIT_DF88 * math.sqrt(variance) / math.sqrt(n)
    return mean, mean - half, mean + half


def load_results():
    if not os.path.isfile(RESULTS_PATH):
        return []
    with open(RESULTS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def save_results(rows):
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def score_method(method, model, loader, device, window):
    per_image = []
    psnr_values = []
    ssim_values = []
    for paths, degraded, target in loader:
        degraded = degraded.to(device)
        target = target.to(device)
        if model is None:
            prediction = degraded
        else:
            prediction = model(degraded)
        prediction = denormalize(prediction)
        truth = denormalize(target)
        psnr_values.extend(psnr_per_image(prediction, truth))
        ssim_values.extend(ssim_per_image(prediction, truth, window))
        for path, psnr, ssim in zip(paths, psnr_values[-len(paths):], ssim_values[-len(paths):]):
            per_image.append({
                "image_id": os.path.basename(path),
                "dataset": "UIEB raw-890 test, synthetic degradation, target = raw image",
                "method": method,
                "PSNR": f"{psnr:.6f}",
                "SSIM": f"{ssim:.6f}",
            })
    psnr_mean, psnr_low, psnr_high = interval(psnr_values)
    ssim_mean, ssim_low, ssim_high = interval(ssim_values)
    summary = {
        "method": method,
        "n": len(psnr_values),
        "psnr_mean": psnr_mean,
        "psnr_low": psnr_low,
        "psnr_high": psnr_high,
        "ssim_mean": ssim_mean,
        "ssim_low": ssim_low,
        "ssim_high": ssim_high,
    }
    return per_image, summary


class PathDataset(ImprovedUnderwaterDataset):
    def __getitem__(self, idx):
        path, image, degraded, _params = self.load_pair(idx)
        if self.transform:
            augmented = self.transform(image=image, degraded=degraded)
            image = augmented["image"]
            degraded = augmented["degraded"]
        return path, degraded, image


def evaluate_checkpoint(spec, device, window, test_loader):
    model = build_paper_variant(spec["architecture"]).to(device)
    checkpoint = torch.load(os.path.join(spec["output_dir"], "checkpoint.pt"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        per_image, summary = score_method(spec["method"], model, test_loader, device, window)
    summary.update({
        "key": spec["key"],
        "change": spec["change"],
        "checkpoint": os.path.relpath(os.path.join(spec["output_dir"], "checkpoint.pt"), ROOT),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "consistency_weight": spec["consistency_weight"],
        "training_input": "clean raw image" if not spec["apply_synthetic_degradation"] else "synthetic degradation",
    })
    del model
    return per_image, summary


def write_tables(results):
    per_image_rows = []
    summary_rows = []
    for row in results:
        per_image_rows.extend(row["per_image"])
        summary_rows.append(row["summary"])
    headers = ["image_id", "dataset", "method", "PSNR", "SSIM"]
    with open(PER_IMAGE_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(per_image_rows)
    summary_headers = [
        "method", "n", "psnr_mean", "psnr_ci95_low", "psnr_ci95_high",
        "ssim_mean", "ssim_ci95_low", "ssim_ci95_high",
        "best_epoch", "best_val_loss", "checkpoint",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_headers)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "method": row["method"],
                "n": row["n"],
                "psnr_mean": f"{row['psnr_mean']:.6f}",
                "psnr_ci95_low": f"{row['psnr_low']:.6f}",
                "psnr_ci95_high": f"{row['psnr_high']:.6f}",
                "ssim_mean": f"{row['ssim_mean']:.6f}",
                "ssim_ci95_low": f"{row['ssim_low']:.6f}",
                "ssim_ci95_high": f"{row['ssim_high']:.6f}",
                "best_epoch": "" if row.get("best_epoch") is None else row["best_epoch"],
                "best_val_loss": "" if row.get("best_val_loss") is None else f"{row['best_val_loss']:.6f}",
                "checkpoint": row.get("checkpoint", ""),
            })
    patch_markdown(summary_rows)


def fmt(value, digits):
    return f"{value:.{digits}f}"


def patch_markdown(summary_rows):
    with open(TABLES_PATH, encoding="utf-8") as handle:
        text = handle.read()
    marker = "## Ablations and baselines"
    if marker not in text:
        raise RuntimeError("PAPER_TABLES.md is missing the ablations section")
    head = text[:text.index(marker)].rstrip()
    head = head.replace(
        "- Checkpoint file present: False.",
        "- Checkpoint file present: True for each variant that has finished training. See the quality table.",
    )
    head = head.replace(
        "Full-reference PSNR and SSIM, UIQM and UCIQE on a challenge subset, ablation scores, and transfer scores are not inserted. `reference-890` is not in the archive, and no checkpoint has been trained.",
        "Paired-reference PSNR and SSIM against `reference-890`, UIQM, UCIQE, and transfer scores are not inserted. `reference-890` is not in the archive. The quality table below is the synthetic-degradation test, with the raw image as the target.",
    )
    head = head.replace(
        "PSNR and SSIM are defined for later evaluation. The UIEB test list contains 89 raw images. They were not scored, because `reference-890` is absent and no checkpoint has been trained, so the evaluated sample count is 0 and the uncertainty interval is undefined.",
        "PSNR and SSIM below are computed on the 89-image UIEB test list after filename-seeded synthetic degradation. The target is the raw image. Paired `reference-890` scores remain undefined because that directory is absent.",
    )
    lines = [
        marker,
        "",
        "Quality scores use the locked UIEB test list (89 images). The input is the filename-seeded synthetic degradation, and the target is the raw image. `reference-890` is not in the archive, so these are not paired-reference UIEB scores. UIQM and UCIQE are not computed.",
        "",
        "| Name | What changes | Checkpoint | Test PSNR | Test SSIM |",
        "|---|---|---|---:|---:|",
    ]
    by_method = {row["method"]: row for row in summary_rows}
    order = [
        ("Unprocessed input", "The degraded image is the output", "none"),
        ("E-URM", "Multiscale attention and adaptive color correction", "runs/evaluated/checkpoint.pt"),
        ("Capacity-matched control", "attention_mode=conv3x3, color module removed", "runs/ablations/capacity-matched/checkpoint.pt"),
        ("No attention", "attention_mode=none; color module kept", "runs/ablations/no-attention/checkpoint.pt"),
        ("No color correction", "color module removed; attention kept", "runs/ablations/no-color/checkpoint.pt"),
        ("No degradation consistency", "same network; loss weight on the consistency term set to 0", "runs/ablations/no-degradation-consistency/checkpoint.pt"),
        ("No synthetic augmentation", "same network; training uses the clean image as the input", "runs/ablations/no-synthetic-augmentation/checkpoint.pt"),
    ]
    for method, change, default_checkpoint in order:
        row = by_method.get(method)
        if row is None:
            lines.append(f"| {method} | {change} | {default_checkpoint}, not trained |  |  |")
            continue
        checkpoint = row.get("checkpoint") or default_checkpoint
        psnr = f"{row['psnr_mean']:.2f} [{row['psnr_low']:.2f}, {row['psnr_high']:.2f}]"
        ssim = f"{row['ssim_mean']:.4f} [{row['ssim_low']:.4f}, {row['ssim_high']:.4f}]"
        lines.append(f"| {method} | {change} | `{checkpoint}` | {psnr} | {ssim} |")
    lines.extend([
        "",
        "The bracket is the 95% Student-t interval of the per-image mean, n = 89. One row per image is in `quality_per_image.csv`. The means are in `quality_summary.csv`.",
        "",
    ])
    with open(TABLES_PATH, "w", encoding="utf-8") as handle:
        handle.write(head + "\n\n" + "\n".join(lines))


def train_variant(spec, device, train_paths, valid_paths):
    set_reproducibility_seed(2026)
    train_transform, val_transform = create_improved_transforms(image_size=256)
    train_dataset = ImprovedUnderwaterDataset(
        train_paths,
        transform=train_transform,
        is_training=True,
        apply_synthetic_degradation=spec["apply_synthetic_degradation"],
    )
    val_dataset = ImprovedUnderwaterDataset(
        valid_paths,
        transform=val_transform,
        is_training=False,
        apply_synthetic_degradation=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    model = build_paper_variant(spec["architecture"])
    weights = EVALUATED_SETTINGS["loss_weights"]
    criterion = PhysicsInformedLoss(
        alpha=weights["l1"],
        beta=weights["mse"],
        gamma=weights["sobel"],
        delta=spec["consistency_weight"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=EVALUATED_SETTINGS["learning_rate"],
        weight_decay=EVALUATED_SETTINGS["weight_decay"],
    )
    scheduler_spec = EVALUATED_SETTINGS["scheduler"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_spec["mode"],
        factor=scheduler_spec["factor"],
        patience=scheduler_spec["patience"],
    )
    config = build_configuration(device)
    config["seed"] = 2026
    config["variant"] = spec["key"]
    config["checkpoint_rule"] = "Keep the epoch with the lowest mean validation loss."
    config["checkpoint_path"] = os.path.relpath(os.path.join(spec["output_dir"], "checkpoint.pt"), ROOT)
    config["loss_weights"] = dict(weights)
    config["loss_weights"]["degradation_consistency"] = spec["consistency_weight"]
    config["training_input"] = (
        "clean raw image" if not spec["apply_synthetic_degradation"] else "synthetic degradation of the raw image"
    )
    config["validation_input"] = "filename-seeded synthetic degradation; target is the raw image"
    config["split_files"] = {name: os.path.relpath(path, ROOT) for name, path in SPLIT_FILES.items()}
    print(f"\nTraining {spec['method']} -> {config['checkpoint_path']}", flush=True)
    started = time.time()
    train_improved_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=EVALUATED_SETTINGS["epochs"],
        device=device,
        output_dir=spec["output_dir"],
        config=config,
    )
    print(f"Finished {spec['method']} in {(time.time() - started) / 3600:.2f} h", flush=True)
    del model
    del optimizer


def rebuild_from_checkpoints(device):
    _, val_transform = create_improved_transforms(image_size=256)
    test_dataset = PathDataset(
        split_paths("test"),
        transform=val_transform,
        is_training=False,
        apply_synthetic_degradation=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
    window = gaussian_window()
    results = []
    with torch.inference_mode():
        per_image, summary = score_method("Unprocessed input", None, test_loader, device, window)
    summary.update({
        "key": "unprocessed",
        "change": "The degraded image is the output",
        "checkpoint": "none",
        "best_epoch": None,
        "best_val_loss": None,
    })
    results.append({"per_image": per_image, "summary": summary})
    for spec in VARIANTS:
        checkpoint_path = os.path.join(spec["output_dir"], "checkpoint.pt")
        if not os.path.isfile(checkpoint_path):
            print("missing checkpoint", checkpoint_path, flush=True)
            continue
        per_image, summary = evaluate_checkpoint(spec, device, window, test_loader)
        results.append({"per_image": per_image, "summary": summary})
        print(
            summary["method"],
            f"PSNR {summary['psnr_mean']:.3f}",
            f"SSIM {summary['ssim_mean']:.4f}",
            flush=True,
        )
    save_results(results)
    write_tables(results)
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    train_paths = split_paths("train")
    valid_paths = split_paths("valid")
    print(f"train {len(train_paths)} validation {len(valid_paths)}", flush=True)
    resume = "--evaluate-only" in sys.argv
    if not resume:
        done = {
            row["summary"]["key"]
            for row in load_results()
            if os.path.isfile(os.path.join(ROOT, row["summary"].get("checkpoint", "missing")))
        }
        for spec in VARIANTS:
            checkpoint_path = os.path.join(spec["output_dir"], "checkpoint.pt")
            if os.path.isfile(checkpoint_path):
                print("skip existing", checkpoint_path, flush=True)
                continue
            if spec["key"] in done:
                continue
            train_variant(spec, device, train_paths, valid_paths)
    rebuild_from_checkpoints(device)
    print("quality tables written", flush=True)


if __name__ == "__main__":
    main()
