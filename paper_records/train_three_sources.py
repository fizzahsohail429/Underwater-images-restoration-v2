"""Train and test E-URM on UIEB, EUVP, and the aquarium pretrain split.

Each source is trained separately. PSNR and SSIM use the original image as the
target after synthetic degradation. UIQM and UCIQE are computed on the output
only. A model trained on one source is also scored on the other two test lists.
"""

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset_catalog import EUVP_RAW, UIEB_RAW, aquarium_root, list_images  # noqa: E402
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
from quality_metrics import (  # noqa: E402
    denormalize,
    interval,
    psnr_per_image,
    ssim_per_image,
    tensor_to_rgb_list,
    uciqe_image,
    uiqm_image,
)

RECORDS = os.path.join(ROOT, "paper_records")
SUMMARY_CSV = os.path.join(RECORDS, "quality_summary.csv")
PER_IMAGE_CSV = os.path.join(RECORDS, "quality_per_image.csv")
TABLES = os.path.join(RECORDS, "PAPER_TABLES.md")


class NamedDataset(ImprovedUnderwaterDataset):
    def __getitem__(self, idx):
        path, image, degraded, _params = self.load_pair(idx)
        if self.transform:
            augmented = self.transform(image=image, degraded=degraded)
            image = augmented["image"]
            degraded = augmented["degraded"]
        return os.path.basename(path), degraded, image


def split_paths(paths, seed=2026):
    order = np.random.default_rng(seed).permutation(len(paths))
    ordered = [paths[index] for index in order]
    n_train = int(round(len(ordered) * 0.8))
    n_valid = int(round(len(ordered) * 0.1))
    return {
        "train": ordered[:n_train],
        "valid": ordered[n_train:n_train + n_valid],
        "test": ordered[n_train + n_valid:],
    }


def locked_uieb_paths():
    lists = {
        "train": os.path.join(RECORDS, "uieb_train.txt"),
        "valid": os.path.join(RECORDS, "uieb_validation.txt"),
        "test": os.path.join(RECORDS, "uieb_test.txt"),
    }
    loaded = {}
    for split, list_path in lists.items():
        with open(list_path, encoding="utf-8") as handle:
            loaded[split] = [os.path.join(UIEB_RAW, line.strip()) for line in handle if line.strip()]
        missing = [path for path in loaded[split] if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(missing[0])
    return loaded


def aquarium_paths():
    root = aquarium_root()
    if not root:
        raise FileNotFoundError("aquarium_pretrain was not found")
    return {
        "train": list_images(os.path.join(root, "train", "images")),
        "valid": list_images(os.path.join(root, "valid", "images")),
        "test": list_images(os.path.join(root, "test", "images")),
    }


def write_split_lists(name, groups):
    for split, paths in groups.items():
        filename = "validation" if split == "valid" else split
        with open(os.path.join(RECORDS, f"{name}_{filename}.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(os.path.basename(path) for path in paths) + "\n")


def shape_check(device):
    model = build_paper_variant("e-urm").to(device)
    model.train()
    sample = torch.randn(2, 3, 256, 256, device=device)
    output = model(sample)
    loss = output.mean()
    loss.backward()
    if tuple(output.shape) != tuple(sample.shape):
        raise RuntimeError(f"output shape {tuple(output.shape)}")
    record = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "input": [2, 3, 256, 256],
        "output": list(output.shape),
        "decoder4_in": model.decoder4[0].in_channels,
        "decoder4_out": model.decoder4[0].out_channels,
        "decoder3_in": model.decoder3[0].in_channels,
        "decoder2_in": model.decoder2[0].in_channels,
        "decoder1_in": model.decoder1[0].in_channels,
        "rgb_head_in": model.final_conv[0].in_channels,
        "forward_and_backward": "passed",
    }
    with open(os.path.join(RECORDS, "architecture_check.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("shape check passed", record["parameters"], flush=True)
    del model


def train_source(name, groups, device):
    output_dir = os.path.join(ROOT, "runs", name)
    checkpoint = os.path.join(output_dir, "checkpoint.pt")
    if os.path.isfile(checkpoint):
        print("skip training, checkpoint exists", checkpoint, flush=True)
        return checkpoint
    set_reproducibility_seed(2026)
    train_transform, val_transform = create_improved_transforms(256)
    train_data = ImprovedUnderwaterDataset(groups["train"], train_transform, is_training=True)
    val_data = ImprovedUnderwaterDataset(groups["valid"], val_transform, is_training=False)
    train_loader = DataLoader(train_data, batch_size=6, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=6, shuffle=False, num_workers=0)
    model = build_paper_variant("e-urm")
    weights = EVALUATED_SETTINGS["loss_weights"]
    criterion = PhysicsInformedLoss(weights["l1"], weights["mse"], weights["sobel"], weights["degradation_consistency"])
    optimizer = optim.AdamW(model.parameters(), lr=EVALUATED_SETTINGS["learning_rate"], weight_decay=EVALUATED_SETTINGS["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    config = build_configuration(device)
    config["seed"] = 2026
    config["source"] = name
    config["checkpoint_rule"] = "Keep the epoch with the lowest mean validation loss."
    config["checkpoint_path"] = os.path.relpath(checkpoint, ROOT)
    print(f"training {name}: {len(train_data)} train, {len(val_data)} validation", flush=True)
    started = time.time()
    train_improved_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=EVALUATED_SETTINGS["epochs"], device=device, output_dir=output_dir, config=config,
    )
    print(f"finished {name} in {(time.time() - started) / 3600:.2f} h", flush=True)
    del model
    return checkpoint


def evaluate_checkpoint(checkpoint, method, dataset_name, paths, device, include_baseline):
    _, val_transform = create_improved_transforms(256)
    data = NamedDataset(paths, val_transform, is_training=False)
    loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=0)
    model = build_paper_variant("e-urm").to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    rows = []
    with torch.inference_mode():
        for names, degraded, target in loader:
            degraded = degraded.to(device)
            target = target.to(device)
            restored = model(degraded)
            prediction = denormalize(restored)
            truth = denormalize(target)
            baseline = denormalize(degraded)
            psnr_model = psnr_per_image(prediction, truth)
            ssim_model = ssim_per_image(prediction, truth)
            psnr_base = psnr_per_image(baseline, truth)
            ssim_base = ssim_per_image(baseline, truth)
            restored_rgb = tensor_to_rgb_list(restored)
            degraded_rgb = tensor_to_rgb_list(degraded)
            for index, image_id in enumerate(names):
                rows.append({
                    "image_id": image_id,
                    "dataset": dataset_name,
                    "method": method,
                    "PSNR": f"{psnr_model[index]:.6f}",
                    "SSIM": f"{ssim_model[index]:.6f}",
                    "UIQM": f"{uiqm_image(restored_rgb[index]):.6f}",
                    "UCIQE": f"{uciqe_image(restored_rgb[index]):.6f}",
                })
                if include_baseline:
                    rows.append({
                        "image_id": image_id,
                        "dataset": dataset_name,
                        "method": "unprocessed input",
                        "PSNR": f"{psnr_base[index]:.6f}",
                        "SSIM": f"{ssim_base[index]:.6f}",
                        "UIQM": f"{uiqm_image(degraded_rgb[index]):.6f}",
                        "UCIQE": f"{uciqe_image(degraded_rgb[index]):.6f}",
                    })
    del model
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (dataset, method), items in grouped.items():
        def values(name):
            return [float(item[name]) for item in items]
        entry = {"dataset": dataset, "method": method, "n": len(items)}
        for metric in ("PSNR", "SSIM", "UIQM", "UCIQE"):
            mean, low, high = interval(values(metric))
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_low"] = low
            entry[f"{metric}_high"] = high
        summary.append(entry)
    return summary


def write_outputs(per_image, summary):
    headers = ["image_id", "dataset", "method", "PSNR", "SSIM", "UIQM", "UCIQE"]
    with open(PER_IMAGE_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(per_image)
    summary_headers = ["dataset", "method", "n"]
    for metric in ("PSNR", "SSIM", "UIQM", "UCIQE"):
        summary_headers.extend([f"{metric}_mean", f"{metric}_low", f"{metric}_high"])
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_headers)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: (f"{row[key]:.6f}" if isinstance(row[key], float) else row[key]) for key in summary_headers})
    lines = [
        "## Measured quality",
        "",
        "PSNR and SSIM compare the output with the original image. The network input is the filename-seeded synthetic degradation. These are not scores against a separate UIEB reference file, because `archive/UIEB` contains the raw images only. UIQM and UCIQE are computed on the output image alone.",
        "",
        "| Dataset | Method | n | PSNR | SSIM | UIQM | UCIQE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def cell(metric):
            return f"{row[metric + '_mean']:.3f} [{row[metric + '_low']:.3f}, {row[metric + '_high']:.3f}]"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n']} | {cell('PSNR')} | {cell('SSIM')} | {cell('UIQM')} | {cell('UCIQE')} |"
        )
    lines.append("")
    lines.append("Brackets are 95% Student-t intervals of the per-image means. One row per image and method is in `quality_per_image.csv`.")
    lines.append("")
    section = "\n".join(lines)
    with open(TABLES, encoding="utf-8") as handle:
        text = handle.read()
    marker = "## Measured quality"
    if marker in text:
        text = text[:text.index(marker)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    with open(TABLES, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    shape_check(device)
    sources = {
        "uieb": locked_uieb_paths(),
        "euvp": split_paths(list_images(EUVP_RAW)),
        "aquarium": aquarium_paths(),
    }
    write_split_lists("euvp", sources["euvp"])
    write_split_lists("aquarium", sources["aquarium"])
    for name, groups in sources.items():
        print({split: len(paths) for split, paths in groups.items()}, name, flush=True)
        train_source(name, groups, device)
    uieb_checkpoint = os.path.join(ROOT, "runs", "uieb", "checkpoint.pt")
    if os.path.isfile(uieb_checkpoint):
        evaluated = os.path.join(ROOT, "runs", "evaluated")
        os.makedirs(evaluated, exist_ok=True)
        torch.save(torch.load(uieb_checkpoint, map_location="cpu"), os.path.join(evaluated, "checkpoint.pt"))
    per_image = []
    for trained_on, groups in sources.items():
        checkpoint = os.path.join(ROOT, "runs", trained_on, "checkpoint.pt")
        for tested_on, test_groups in sources.items():
            method = f"E-URM trained on {trained_on}"
            if trained_on != tested_on:
                method += " (transfer)"
            print("evaluate", method, "on", tested_on, flush=True)
            per_image.extend(evaluate_checkpoint(
                checkpoint, method, tested_on, test_groups["test"], device,
                include_baseline=(trained_on == tested_on),
            ))
    summary = summarize(per_image)
    write_outputs(per_image, summary)
    print("records written", SUMMARY_CSV, flush=True)


if __name__ == "__main__":
    main()
