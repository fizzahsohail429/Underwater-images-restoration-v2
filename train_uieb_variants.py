"""Train one controlled UIEB variant without touching the completed E-URM run.

The data, split, preprocessing, epoch budget, and checkpoint rule match
train_uieb_reference.py. Water-Net, FUnIE-GAN, and PCAFA-Net are not trained
here: their official code and checkpoints are not in this repository.
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import torch

import train_uieb_reference as base
from improved_underwater_restoration import (
    EVALUATED_SETTINGS,
    create_advanced_degradation,
    create_improved_transforms,
    set_reproducibility_seed,
)
from torch.utils.data import DataLoader

ROOT = base.ROOT
VARIANTS = {
    "capacity-matched": {
        "role": "capacity-matched encoder-decoder",
        "architecture": "capacity-matched",
        "consistency_weight": 0.0,
        "synthetic_training_input": False,
    },
    "no-attention": {
        "role": "ablation with multiscale attention removed",
        "architecture": "no-attention",
        "consistency_weight": 0.0,
        "synthetic_training_input": False,
    },
    "no-color": {
        "role": "ablation with adaptive color correction removed",
        "architecture": "no-color",
        "consistency_weight": 0.0,
        "synthetic_training_input": False,
    },
    "consistency": {
        "role": "extension: E-URM with degradation-consistency weight 0.1",
        "architecture": "e-urm",
        "consistency_weight": 0.1,
        "synthetic_training_input": False,
    },
    "synthetic": {
        "role": "extension: E-URM trained on a synthetic degradation of the raw input",
        "architecture": "e-urm",
        "consistency_weight": 0.0,
        "synthetic_training_input": True,
    },
}


def run_directory(name):
    path = os.path.join(ROOT, "runs", "uieb_variants", name)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


class SyntheticInputDataset(base.ReferencePairDataset):
    """Training input is a filename-seeded degradation. The target stays reference-890."""

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        pair = self.pairs[image_id]
        raw = base.read_rgb(pair["raw"])
        reference = base.read_rgb(pair["reference"])
        digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
        degraded, _params = create_advanced_degradation(raw, np.random.default_rng(int(digest[:16], 16)))
        augmented = self.transform(image=reference, degraded=degraded)
        return augmented["degraded"], augmented["image"]


def make_variant_loaders(splits, pairs, synthetic_training_input):
    _train_transform, pair_transform = create_improved_transforms(image_size=256)
    del _train_transform
    train_set = (
        SyntheticInputDataset(splits["train"], pairs, pair_transform)
        if synthetic_training_input
        else base.ReferencePairDataset(splits["train"], pairs, pair_transform)
    )
    generator = torch.Generator()
    generator.manual_seed(2026)
    train_loader = DataLoader(
        train_set, batch_size=EVALUATED_SETTINGS["batch_size"], shuffle=True,
        num_workers=0, generator=generator,
    )
    val_loader = DataLoader(
        base.ReferencePairDataset(splits["validation"], pairs, pair_transform),
        batch_size=EVALUATED_SETTINGS["batch_size"], shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


def variant_config(name, spec, device, learning_rate):
    config = base.configuration(device, learning_rate)
    config.update({
        "variant": name,
        "role": spec["role"],
        "architecture": spec["architecture"],
        "consistency_weight": spec["consistency_weight"],
        "synthetic_training_input": spec["synthetic_training_input"],
        "validation_and_test_input": "raw UIEB",
        "target": "archive/reference-890 same filename stem",
        "completed_eurm_run": "runs/uieb_reference",
        "started": datetime.now().isoformat(timespec="seconds"),
    })
    if spec["consistency_weight"] == 0.0:
        config["loss"]["degradation_consistency"] = 0.0
    else:
        config["loss"]["degradation_consistency"] = spec["consistency_weight"]
        config["loss"]["reason_delta_zero"] = "Not used. This extension sets the consistency weight to 0.1."
    return config


def latest_checkpoint(last_path):
    item = base.checkpoint_progress(last_path)
    if item is None:
        return None
    return item


def train_variant(name, learning_rate):
    spec = VARIANTS[name]
    device = torch.device("cpu")
    run = run_directory(name)
    config_path = os.path.join(run, "variant_config.json")
    log_path = os.path.join(run, "training_log.csv")
    last_path = os.path.join(run, "checkpoints", "last.pt")
    best_path = os.path.join(run, "checkpoints", "best_val_psnr.pt")
    splits, pairs = base.verify_split()
    print(f"Variant: {name}", flush=True)
    print(spec["role"], flush=True)
    print("Validation input: raw UIEB", flush=True)
    print("Target: reference-890", flush=True)
    set_reproducibility_seed(2026)
    config = variant_config(name, spec, device, learning_rate)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    train_loader, val_loader = make_variant_loaders(splits, pairs, spec["synthetic_training_input"])
    model, criterion, optimizer, scheduler = base.build_model(
        device, learning_rate, spec["architecture"], spec["consistency_weight"]
    )
    history = []
    best_psnr = -float("inf")
    start_epoch = 1
    resumed = latest_checkpoint(last_path)
    if resumed is not None:
        _rank, path, state = resumed
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        history = list(state.get("history") or [])
        if history:
            best_psnr = max(float(row["val_psnr"]) for row in history)
        saved_epoch = int(state["epoch"])
        start_epoch = saved_epoch + 1 if len(history) >= saved_epoch else saved_epoch
        print(f"Resuming {name} from epoch {start_epoch}", flush=True)
    elif os.path.isfile(log_path):
        os.remove(log_path)
    if start_epoch > 30:
        print(f"{name} already has 30 epochs.", flush=True)
        return
    fields = ["epoch", "train_loss", "val_psnr", "val_ssim", "learning_rate"]
    for epoch in range(start_epoch, 31):
        batch_log = os.path.join(run, "batch_log.csv")
        train_loss = base.train_batches(
            model, criterion, optimizer, train_loader, device, epoch, None, batch_log
        )
        val_psnr, val_ssim = base.validation_scores(model, val_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "learning_rate": current_lr,
        }
        history.append(row)
        base.append_csv(log_path, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_psnr": f"{val_psnr:.6f}",
            "val_ssim": f"{val_ssim:.6f}",
            "learning_rate": f"{current_lr:.8f}",
        }, fields)
        print(
            f"{name} epoch {epoch}/30 train loss {train_loss:.4f} "
            f"val PSNR {val_psnr:.4f} val SSIM {val_ssim:.4f}",
            flush=True,
        )
        state_config = dict(config)
        state_config["history"] = history
        base.save_training_state(last_path, model, optimizer, scheduler, epoch, history, state_config, val_psnr, None)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            base.save_training_state(best_path, model, optimizer, scheduler, epoch, history, state_config, val_psnr, None)
            print(f"New best validation PSNR for {name}: epoch {epoch}", flush=True)
        scheduler.step(val_psnr)


def test_variant(name):
    spec = VARIANTS[name]
    device = torch.device("cpu")
    run = run_directory(name)
    best_path = os.path.join(run, "checkpoints", "best_val_psnr.pt")
    if not os.path.isfile(best_path):
        raise RuntimeError(f"Missing selected checkpoint {best_path}")
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    model, _criterion, _optimizer, _scheduler = base.build_model(
        device, 0.0001, spec["architecture"], spec["consistency_weight"]
    )
    model.load_state_dict(state["model"])
    bad_name = base.weights_finite(model)
    if bad_name is not None:
        raise RuntimeError(f"Selected checkpoint has a non-finite parameter: {bad_name}")
    splits, pairs = base.verify_split()
    _train_transform, pair_transform = create_improved_transforms(image_size=256)
    dataset = base.ReferencePairDataset(splits["test"], pairs, pair_transform)
    rows = []
    model.eval()
    with torch.no_grad():
        for image_id in splits["test"]:
            model_input, target = dataset[dataset.image_ids.index(image_id)]
            output = model(model_input.unsqueeze(0).to(device))
            base.finite_tensor(output, image_id)
            prediction = base.denormalize(output).cpu()
            reference = base.denormalize(target.unsqueeze(0)).cpu()
            baseline = base.denormalize(model_input.unsqueeze(0)).cpu()
            baseline_psnr = base.psnr_per_image(baseline, reference)[0]
            model_psnr = base.psnr_per_image(prediction, reference)[0]
            baseline_ssim = base.ssim_per_image(baseline, reference)[0]
            model_ssim = base.ssim_per_image(prediction, reference)[0]
            rows.append({
                "filename": image_id,
                "baseline_psnr": baseline_psnr,
                "variant_psnr": model_psnr,
                "psnr_delta": model_psnr - baseline_psnr,
                "baseline_ssim": baseline_ssim,
                "variant_ssim": model_ssim,
                "ssim_delta": model_ssim - baseline_ssim,
            })
            print(image_id, f"{baseline_psnr:.3f}", f"{model_psnr:.3f}", flush=True)
    if len(rows) != 89:
        raise RuntimeError("Test evaluation did not produce 89 rows.")
    csv_path = os.path.join(run, "test_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    psnr = base.paired_summary([row["psnr_delta"] for row in rows])
    ssim = base.paired_summary([row["ssim_delta"] for row in rows])
    summary = {
        "variant": name,
        "checkpoint": os.path.relpath(best_path, ROOT),
        "selected_epoch": int(state["epoch"]),
        "selection_rule": "highest mean validation PSNR",
        "validation_psnr_at_selection": float(state["best_val_loss"]),
        "baseline_psnr_mean": float(sum(row["baseline_psnr"] for row in rows) / len(rows)),
        "variant_psnr_mean": float(sum(row["variant_psnr"] for row in rows) / len(rows)),
        "baseline_ssim_mean": float(sum(row["baseline_ssim"] for row in rows) / len(rows)),
        "variant_ssim_mean": float(sum(row["variant_ssim"] for row in rows) / len(rows)),
        "psnr": psnr,
        "ssim": ssim,
    }
    with open(os.path.join(run, "test_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({
        "selected_epoch": summary["selected_epoch"],
        "validation_psnr": summary["validation_psnr_at_selection"],
        "baseline_psnr_mean": summary["baseline_psnr_mean"],
        "variant_psnr_mean": summary["variant_psnr_mean"],
        "baseline_ssim_mean": summary["baseline_ssim_mean"],
        "variant_ssim_mean": summary["variant_ssim_mean"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--phase", choices=["train", "test"], required=True)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    if args.phase == "train":
        train_variant(args.variant, args.learning_rate)
    else:
        test_variant(args.variant)


if __name__ == "__main__":
    main()
