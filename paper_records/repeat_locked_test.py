"""Repeat locked-test inference without overwriting the retained score files.

Writes paper_records/repeat_inference/. Compares PSNR and SSIM with the
retained tables and adds UIQM and UCIQE on the unprocessed input and output.
"""

import csv
import json
import os
import platform
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_uieb_reference as base  # noqa: E402
from improved_underwater_restoration import build_paper_variant  # noqa: E402
from quality_metrics import denormalize, psnr_per_image, ssim_per_image, uiqm_image, uciqe_image  # noqa: E402

OUT = os.path.join(ROOT, "paper_records", "repeat_inference")
MODELS = [
    ("e-urm", os.path.join(ROOT, "runs", "uieb_reference", "checkpoints", "best_val_psnr.pt"),
     os.path.join(ROOT, "runs", "uieb_reference", "uieb_test_results.csv"), "e_urm"),
    ("no-color", os.path.join(ROOT, "runs", "uieb_variants", "no-color", "checkpoints", "best_val_psnr.pt"),
     os.path.join(ROOT, "runs", "uieb_variants", "no-color", "test_results.csv"), "variant"),
    ("capacity-matched", os.path.join(ROOT, "runs", "uieb_variants", "capacity-matched", "checkpoints", "best_val_psnr.pt"),
     os.path.join(ROOT, "runs", "uieb_variants", "capacity-matched", "test_results.csv"), "variant"),
    ("no-attention", os.path.join(ROOT, "runs", "uieb_variants", "no-attention", "checkpoints", "best_val_psnr.pt"),
     os.path.join(ROOT, "runs", "uieb_variants", "no-attention", "test_results.csv"), "variant"),
]


def load_retained(path, prefix):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["filename"]: {
            "psnr": float(row[f"{prefix}_psnr"]),
            "ssim": float(row[f"{prefix}_ssim"]),
            "baseline_psnr": float(row["baseline_psnr"]),
            "baseline_ssim": float(row["baseline_ssim"]),
        }
        for row in rows
    }


def rgb_of(tensor):
    array = tensor.detach().cpu().numpy()
    return np.transpose(array, (1, 2, 0))


def score_model(name, checkpoint, retained_csv, prefix, splits, pairs, transform, device):
    retained = load_retained(retained_csv, prefix)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_paper_variant(name).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = base.ReferencePairDataset(splits["test"], pairs, transform)
    rows = []
    max_psnr = 0.0
    max_ssim = 0.0
    max_base_psnr = 0.0
    max_base_ssim = 0.0
    with torch.no_grad():
        for image_id in splits["test"]:
            model_input, target = dataset[dataset.image_ids.index(image_id)]
            output = model(model_input.unsqueeze(0).to(device))
            prediction = denormalize(output)[0].cpu()
            reference = denormalize(target.unsqueeze(0))[0].cpu()
            baseline = denormalize(model_input.unsqueeze(0))[0].cpu()
            model_psnr = psnr_per_image(prediction.unsqueeze(0), reference.unsqueeze(0))[0]
            model_ssim = ssim_per_image(prediction.unsqueeze(0), reference.unsqueeze(0))[0]
            baseline_psnr = psnr_per_image(baseline.unsqueeze(0), reference.unsqueeze(0))[0]
            baseline_ssim = ssim_per_image(baseline.unsqueeze(0), reference.unsqueeze(0))[0]
            previous = retained[image_id]
            max_psnr = max(max_psnr, abs(model_psnr - previous["psnr"]))
            max_ssim = max(max_ssim, abs(model_ssim - previous["ssim"]))
            max_base_psnr = max(max_base_psnr, abs(baseline_psnr - previous["baseline_psnr"]))
            max_base_ssim = max(max_base_ssim, abs(baseline_ssim - previous["baseline_ssim"]))
            pred_rgb = rgb_of(prediction)
            base_rgb = rgb_of(baseline)
            rows.append({
                "filename": image_id,
                "baseline_psnr": baseline_psnr,
                "model_psnr": model_psnr,
                "baseline_ssim": baseline_ssim,
                "model_ssim": model_ssim,
                "baseline_uiqm": uiqm_image(base_rgb),
                "model_uiqm": uiqm_image(pred_rgb),
                "baseline_uciqe": uciqe_image(base_rgb),
                "model_uciqe": uciqe_image(pred_rgb),
            })
            print(name, image_id, f"{model_psnr:.3f}", flush=True)
    summary = {
        "variant": name,
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "selected_epoch": int(state["epoch"]),
        "n": len(rows),
        "max_abs_psnr_difference_vs_retained": max_psnr,
        "max_abs_ssim_difference_vs_retained": max_ssim,
        "max_abs_baseline_psnr_difference_vs_retained": max_base_psnr,
        "max_abs_baseline_ssim_difference_vs_retained": max_base_ssim,
        "mean_model_psnr": float(np.mean([row["model_psnr"] for row in rows])),
        "mean_model_ssim": float(np.mean([row["model_ssim"] for row in rows])),
        "mean_baseline_psnr": float(np.mean([row["baseline_psnr"] for row in rows])),
        "mean_baseline_ssim": float(np.mean([row["baseline_ssim"] for row in rows])),
        "mean_model_uiqm": float(np.mean([row["model_uiqm"] for row in rows])),
        "mean_baseline_uiqm": float(np.mean([row["baseline_uiqm"] for row in rows])),
        "mean_model_uciqe": float(np.mean([row["model_uciqe"] for row in rows])),
        "mean_baseline_uciqe": float(np.mean([row["baseline_uciqe"] for row in rows])),
    }
    csv_path = os.path.join(OUT, f"{name}_repeat.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    return summary


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu")
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "num_workers": 0,
    }
    splits, pairs = base.verify_split()
    _train_transform, pair_transform = base.create_improved_transforms(image_size=256)
    summaries = []
    for name, checkpoint, retained_csv, prefix in MODELS:
        summaries.append(score_model(name, checkpoint, retained_csv, prefix, splits, pairs, pair_transform, device))
    payload = {"environment": environment, "models": summaries}
    with open(os.path.join(OUT, "repeat_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
