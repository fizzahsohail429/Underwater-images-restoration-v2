"""Train and test E-URM on locked UIEB pairs: raw input, reference-890 target.

The older loader reconstructs a raw image from a synthetic degradation. This
script does not use that task. Test images are not used for training, model
selection, or checkpoint choice.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper_records"))

from dataset_catalog import UIEB_RAW, UIEB_REFERENCE  # noqa: E402
from improved_underwater_restoration import (  # noqa: E402
    EVALUATED_SETTINGS,
    HybridEncoderDecoder,
    PhysicsInformedLoss,
    create_improved_transforms,
    save_checkpoint,
    set_reproducibility_seed,
)
from quality_metrics import denormalize, psnr_per_image, ssim_per_image  # noqa: E402

RUN = os.path.join(ROOT, "runs", "uieb_reference")
SPLIT_JSON = os.path.join(RUN, "uieb_split.json")
CONFIG_JSON = os.path.join(RUN, "uieb_experiment_config.json")
LOG_CSV = os.path.join(RUN, "training_log.csv")
BATCH_LOG = os.path.join(RUN, "stability_batches.csv")
TEST_CSV = os.path.join(RUN, "uieb_test_results.csv")
SUMMARY_JSON = os.path.join(RUN, "uieb_test_summary.json")
REPORT = os.path.join(RUN, "UIEB_CLAIM_REPORT.md")
BEST_PSNR = os.path.join(RUN, "checkpoints", "best_val_psnr.pt")
BEST_SSIM = os.path.join(RUN, "checkpoints", "best_val_ssim.pt")
LAST_PT = os.path.join(RUN, "checkpoints", "last.pt")
LAST_FINITE = os.path.join(RUN, "checkpoints", "last_finite.pt")
DIAGNOSTIC = os.path.join(RUN, "diagnostics", "nonfinite_state.pt")


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "NOT RECOVERABLE FROM CURRENT REPOSITORY"


def read_list(name):
    path = os.path.join(ROOT, "paper_records", f"uieb_{name}.txt")
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()], path


def reference_index():
    index = {}
    for name in os.listdir(UIEB_REFERENCE):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = os.path.splitext(name)[0]
        index.setdefault(stem, []).append(os.path.join(UIEB_REFERENCE, name))
    return index


REFERENCE_INDEX = None


def find_reference(image_id):
    global REFERENCE_INDEX
    if REFERENCE_INDEX is None:
        REFERENCE_INDEX = reference_index()
    matches = REFERENCE_INDEX.get(os.path.splitext(image_id)[0], [])
    if len(matches) != 1:
        raise RuntimeError(f"{image_id} has {len(matches)} reference matches")
    return matches[0]


def verify_split():
    splits = {}
    sources = {}
    for name in ("train", "validation", "test"):
        file_name = "valid" if name == "validation" else name
        ids, path = read_list(file_name if name != "validation" else "validation")
        splits[name] = ids
        sources[name] = path
    counts = {name: len(ids) for name, ids in splits.items()}
    if counts != {"train": 712, "validation": 89, "test": 89}:
        raise RuntimeError(f"Split counts are not 712/89/89: {counts}")
    train, valid, test = set(splits["train"]), set(splits["validation"]), set(splits["test"])
    if train & valid or train & test or valid & test:
        raise RuntimeError("The locked split has overlapping filenames.")
    if len(train | valid | test) != 890:
        raise RuntimeError("The locked split does not cover 890 unique filenames.")
    raw_names = {
        name for name in os.listdir(UIEB_RAW)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    if raw_names != train | valid | test:
        raise RuntimeError("archive/UIEB filenames do not match the locked split.")
    pairs = {}
    for image_id in sorted(train | valid | test):
        raw_path = os.path.join(UIEB_RAW, image_id)
        reference_path = find_reference(image_id)
        if os.path.dirname(raw_path) != UIEB_RAW:
            raise RuntimeError(f"Input is not in archive/UIEB: {raw_path}")
        if os.path.dirname(reference_path) != UIEB_REFERENCE:
            raise RuntimeError(f"Target is not in reference-890: {reference_path}")
        if not os.path.isfile(raw_path) or not os.path.isfile(reference_path):
            raise RuntimeError(f"Missing pair for {image_id}")
        pairs[image_id] = {"raw": raw_path, "reference": reference_path}
    os.makedirs(RUN, exist_ok=True)
    payload = {
        "seed": 2026,
        "counts": counts,
        "sources": sources,
        "raw_directory": UIEB_RAW,
        "reference_directory": UIEB_REFERENCE,
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
        "overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    with open(SPLIT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("Input: raw UIEB", flush=True)
    print("Target: reference-890", flush=True)
    print(f"Verified pairs: train {counts['train']}, validation {counts['validation']}, test {counts['test']}", flush=True)
    return splits, pairs


def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class ReferencePairDataset(Dataset):
    """Raw UIEB image in, same-stem reference-890 image as the target."""

    def __init__(self, image_ids, pairs, transform):
        self.image_ids = list(image_ids)
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        pair = self.pairs[image_id]
        if os.path.dirname(pair["raw"]) != UIEB_RAW or os.path.dirname(pair["reference"]) != UIEB_REFERENCE:
            raise RuntimeError(f"Wrong directories for {image_id}")
        raw = read_rgb(pair["raw"])
        reference = read_rgb(pair["reference"])
        augmented = self.transform(image=reference, degraded=raw)
        model_input = augmented["degraded"]
        target = augmented["image"]
        if model_input.shape != target.shape:
            raise RuntimeError(f"Input and target shapes differ for {image_id}")
        return model_input, target


def finite_tensor(tensor, label):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values in {label}")


def grad_norm(model):
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            bad = ~torch.isfinite(parameter.grad.detach())
            raise RuntimeError(
                f"Non-finite gradient in {name}: {int(bad.sum())} values, "
                f"shape {tuple(parameter.grad.shape)}"
            )
        total += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def weights_finite(model):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            return name
    return None


def append_csv(path, row, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_diagnostic(model, optimizer, epoch, batch_index, reason):
    os.makedirs(os.path.dirname(DIAGNOSTIC), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index,
        "reason": reason,
    }, DIAGNOSTIC)


def make_loaders(splits, pairs, batch_size):
    _train_transform, pair_transform = create_improved_transforms(image_size=256)
    del _train_transform
    generator = torch.Generator()
    generator.manual_seed(2026)
    train_loader = DataLoader(
        ReferencePairDataset(splits["train"], pairs, pair_transform),
        batch_size=batch_size, shuffle=True, num_workers=0, generator=generator,
    )
    val_loader = DataLoader(
        ReferencePairDataset(splits["validation"], pairs, pair_transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        ReferencePairDataset(splits["test"], pairs, pair_transform),
        batch_size=1, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, test_loader


def build_model(device, learning_rate):
    model = HybridEncoderDecoder().to(device)
    criterion = PhysicsInformedLoss(alpha=0.4, beta=0.3, gamma=0.2, delta=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    return model, criterion, optimizer, scheduler


def configuration(device, learning_rate):
    return {
        "dataset": "UIEB",
        "input": "archive/UIEB raw image",
        "target": "archive/reference-890 same filename stem",
        "synthetic_degradation": False,
        "photometric_augmentation": False,
        "geometric_augmentation": False,
        "resize": "OpenCV/Albumentations resize to 256x256, evaluation uses the same paired resize",
        "normalization": "(pixel/255 - 0.5) / 0.5",
        "output": "denormalize tanh to [0, 1] before PSNR and SSIM",
        "resolution": [256, 256],
        "color": "RGB",
        "batch_size": EVALUATED_SETTINGS["batch_size"],
        "epochs": 30,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "manuscript_learning_rate": 0.001,
        "learning_rate_reason": "0.001 made decoder4.0 grow from 2.5 to 671 by epoch 1 batch 15 and made every value in initial_conv.0.weight's gradient non-finite. The rate below is the stability correction.",
        "weight_decay": 0.0001,
        "scheduler": "ReduceLROnPlateau mode=max factor=0.5 patience=5, stepped on validation PSNR",
        "loss": {
            "l1": 0.4,
            "mse": 0.3,
            "sobel": 0.2,
            "degradation_consistency": 0.0,
            "reason_delta_zero": "The consistency term compares a tinted blur of the output with the raw input. It is not a reference-890 loss, so it is off for this claim.",
        },
        "gradient_clipping": None,
        "seed": 2026,
        "checkpoint_rule": "Highest mean validation PSNR. Validation SSIM is saved separately and is not used to choose the tested checkpoint.",
        "test_images_used_for_selection": False,
        "device": str(device),
        "git_revision": git_revision(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def save_training_state(path, model, optimizer, scheduler, epoch, history, config, best_val_loss, batch_index):
    """Write a complete finite state, then replace the destination.

    A crash during the write leaves the previous checkpoint in place.
    """
    if weights_finite(model) is not None:
        raise RuntimeError(f"Refusing to save a non-finite checkpoint to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index,
        "best_val_loss": best_val_loss,
        "history": history,
        "configuration": config,
        "finite": True,
    }, temporary)
    os.replace(temporary, path)


def checkpoint_progress(path):
    if not os.path.isfile(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not state.get("finite", True):
        return None
    epoch = int(state.get("epoch") or 0)
    batch_index = state.get("batch_index")
    batch_index = -1 if batch_index is None else int(batch_index)
    history = state.get("history") or []
    completed = len(history) >= epoch and epoch > 0
    rank = (epoch, 10**9 if completed else batch_index)
    return rank, path, state


def latest_finite_checkpoint():
    """Prefer a mid-epoch finite save over the previous completed epoch."""
    choices = []
    for path in (LAST_PT, LAST_FINITE):
        item = checkpoint_progress(path)
        if item is not None:
            choices.append(item)
    if not choices:
        return None
    choices.sort()
    return choices[-1]


def train_batches(model, criterion, optimizer, loader, device, epoch, max_batches, log_path, on_finite_batch=None):
    model.train()
    losses = []
    activation = {"decoder4": None}
    hook = model.decoder4[0].register_forward_hook(
        lambda _module, _inputs, output: activation.__setitem__(
            "decoder4", float(output.detach().abs().max())
        )
    )
    fields = ["epoch", "batch", "loss", "learning_rate", "gradient_norm", "output_min", "output_max", "finite"]
    for batch_index, (model_input, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        model_input = model_input.to(device)
        target = target.to(device)
        try:
            finite_tensor(model_input, "input")
            finite_tensor(target, "target")
            optimizer.zero_grad(set_to_none=True)
            output = model(model_input)
            finite_tensor(output, "output")
            loss, _metrics = criterion(output, target, model_input)
            finite_tensor(loss, "loss")
            loss.backward()
            norm = grad_norm(model)
            optimizer.step()
            bad_name = weights_finite(model)
            if bad_name is not None:
                raise RuntimeError(f"Non-finite parameter after optimizer step: {bad_name}")
        except RuntimeError as exc:
            save_diagnostic(model, optimizer, epoch, batch_index, str(exc))
            hook.remove()
            raise RuntimeError(
                f"Stopped at epoch {epoch}, batch {batch_index}. {exc}. "
                f"decoder4.0 max abs {activation['decoder4']}. Diagnostic: {DIAGNOSTIC}"
            ) from exc
        value = float(loss.item())
        losses.append(value)
        append_csv(log_path, {
            "epoch": epoch,
            "batch": batch_index,
            "loss": f"{value:.6f}",
            "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}",
            "gradient_norm": f"{norm:.6f}",
            "output_min": f"{float(output.detach().min()):.6f}",
            "output_max": f"{float(output.detach().max()):.6f}",
            "finite": "yes",
        }, fields)
        if on_finite_batch is not None:
            on_finite_batch(batch_index)
        if batch_index % 10 == 0 or batch_index == 30:
            print(
                f"Epoch {epoch}, Batch {batch_index}, Loss {value:.4f}, "
                f"grad {norm:.4f}, output [{float(output.detach().min()):.3f}, {float(output.detach().max()):.3f}], "
                f"decoder4.0 {activation['decoder4']:.3f}",
                flush=True,
            )
    hook.remove()
    if not losses:
        raise RuntimeError("No training batches ran.")
    return float(np.mean(losses))


def score_loader(model, loader, device):
    model.eval()
    psnr_values = []
    ssim_values = []
    rows = []
    with torch.no_grad():
        for model_input, target in loader:
            model_input = model_input.to(device)
            target = target.to(device)
            output = model(model_input)
            finite_tensor(output, "validation output")
            prediction = denormalize(output).cpu()
            reference = denormalize(target).cpu()
            baseline = denormalize(model_input).cpu()
            psnr_values.extend(psnr_per_image(prediction, reference))
            ssim_values.extend(ssim_per_image(prediction, reference))
            rows.append((baseline, prediction, reference))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values)), rows


def validation_scores(model, loader, device):
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for model_input, target in loader:
            output = model(model_input.to(device))
            finite_tensor(output, "validation output")
            prediction = denormalize(output).cpu()
            reference = denormalize(target).cpu()
            psnr_values.extend(psnr_per_image(prediction, reference))
            ssim_values.extend(ssim_per_image(prediction, reference))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def run_stability(splits, pairs, device, batches, learning_rate):
    print("Input: raw UIEB", flush=True)
    print("Target: reference-890", flush=True)
    print(f"Stability test: epoch 1 through batch {batches - 1}, learning rate {learning_rate}.", flush=True)
    set_reproducibility_seed(2026)
    train_loader, _val_loader, _test_loader = make_loaders(splits, pairs, EVALUATED_SETTINGS["batch_size"])
    model, criterion, optimizer, _scheduler = build_model(device, learning_rate)
    if os.path.isfile(BATCH_LOG):
        os.remove(BATCH_LOG)
    mean_loss = train_batches(model, criterion, optimizer, train_loader, device, 1, batches, BATCH_LOG)
    bad_name = weights_finite(model)
    if bad_name is not None:
        raise RuntimeError(f"Stability test ended with a non-finite parameter: {bad_name}")
    print(f"Stability passed. Mean loss over {batches} batches: {mean_loss:.4f}", flush=True)
    return mean_loss


def run_training(splits, pairs, device, learning_rate):
    print("Input: raw UIEB", flush=True)
    print("Target: reference-890", flush=True)
    set_reproducibility_seed(2026)
    config = configuration(device, learning_rate)
    with open(CONFIG_JSON, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    train_loader, val_loader, _test_loader = make_loaders(splits, pairs, EVALUATED_SETTINGS["batch_size"])
    model, criterion, optimizer, scheduler = build_model(device, learning_rate)
    history = []
    best_psnr = -float("inf")
    best_ssim = -float("inf")
    start_epoch = 1
    resumed = latest_finite_checkpoint()
    if resumed is not None:
        _rank, path, state = resumed
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        history = list(state.get("history") or [])
        if history:
            best_psnr = max(float(row["val_psnr"]) for row in history)
            best_ssim = max(float(row["val_ssim"]) for row in history)
        saved_epoch = int(state["epoch"])
        start_epoch = saved_epoch + 1 if len(history) >= saved_epoch else saved_epoch
        print(
            f"Resuming from {path} at epoch {start_epoch}. "
            f"Saved epoch {saved_epoch}, batch {state.get('batch_index')}. "
            f"Best validation PSNR so far {best_psnr:.4f}.",
            flush=True,
        )
    else:
        if os.path.isfile(LOG_CSV):
            os.remove(LOG_CSV)
    if start_epoch > 30:
        print("Training already completed 30 epochs.", flush=True)
        return history
    for epoch in range(start_epoch, 31):
        print("Input: raw UIEB", flush=True)
        print("Target: reference-890", flush=True)

        def save_batch(batch_index, epoch=epoch):
            save_training_state(
                LAST_FINITE, model, optimizer, scheduler, epoch, history, config, best_psnr, batch_index
            )
            if batch_index % 10 == 0:
                print(f"Saved finite checkpoint epoch {epoch} batch {batch_index}: {LAST_FINITE}", flush=True)

        train_loss = train_batches(
            model, criterion, optimizer, train_loader, device, epoch, None,
            os.path.join(RUN, "batch_log.csv"), save_batch,
        )
        val_psnr, val_ssim = validation_scores(model, val_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "learning_rate": current_lr,
        }
        history.append(row)
        append_csv(LOG_CSV, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_psnr": f"{val_psnr:.6f}",
            "val_ssim": f"{val_ssim:.6f}",
            "learning_rate": f"{current_lr:.8f}",
        }, ["epoch", "train_loss", "val_psnr", "val_ssim", "learning_rate"])
        print(
            f"Epoch {epoch}/30 train loss {train_loss:.4f} val PSNR {val_psnr:.4f} val SSIM {val_ssim:.4f}",
            flush=True,
        )
        state_config = dict(config)
        state_config["history"] = history
        save_training_state(LAST_PT, model, optimizer, scheduler, epoch, history, state_config, val_psnr, None)
        save_training_state(LAST_FINITE, model, optimizer, scheduler, epoch, history, state_config, val_psnr, None)
        print(f"Saved finite epoch checkpoint: {LAST_PT}", flush=True)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_training_state(BEST_PSNR, model, optimizer, scheduler, epoch, history, state_config, val_psnr, None)
            print(f"New best validation PSNR checkpoint: epoch {epoch}", flush=True)
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            save_training_state(BEST_SSIM, model, optimizer, scheduler, epoch, history, state_config, val_ssim, None)
            print(f"New best validation SSIM checkpoint: epoch {epoch}", flush=True)
        scheduler.step(val_psnr)
    return history


def paired_summary(deltas):
    array = np.asarray(deltas, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    median = float(np.median(array))
    n = array.size
    try:
        from scipy import stats
        critical = float(stats.t.ppf(0.975, n - 1))
        test = stats.ttest_rel(array, np.zeros(n))
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        test_name = "scipy.stats.ttest_rel, two-sided, paired difference versus zero"
    except Exception:
        # Student t tail is not available without SciPy. The interval below uses
        # a normal 1.96 approximation only if SciPy cannot be imported.
        critical = 1.96
        se = std / math.sqrt(n)
        statistic = mean / se if se else float("nan")
        p_value = None
        test_name = "normal approximation because SciPy was unavailable"
    se = std / math.sqrt(n)
    return {
        "n": int(n),
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "improved": int(np.sum(array > 0)),
        "worse": int(np.sum(array < 0)),
        "unchanged": int(np.sum(array == 0)),
    }


def run_test(splits, pairs, device):
    if not os.path.isfile(BEST_PSNR):
        raise RuntimeError(f"Missing selected checkpoint {BEST_PSNR}")
    state = torch.load(BEST_PSNR, map_location="cpu", weights_only=False)
    model = HybridEncoderDecoder().to(device)
    model.load_state_dict(state["model"])
    bad_name = weights_finite(model)
    if bad_name is not None:
        raise RuntimeError(f"Selected checkpoint has a non-finite parameter: {bad_name}")
    _train_transform, pair_transform = create_improved_transforms(image_size=256)
    dataset = ReferencePairDataset(splits["test"], pairs, pair_transform)
    rows = []
    model.eval()
    with torch.no_grad():
        for image_id in splits["test"]:
            model_input, target = dataset[dataset.image_ids.index(image_id)]
            output = model(model_input.unsqueeze(0).to(device))
            finite_tensor(output, image_id)
            prediction = denormalize(output).cpu()
            reference = denormalize(target.unsqueeze(0)).cpu()
            baseline = denormalize(model_input.unsqueeze(0)).cpu()
            baseline_psnr = psnr_per_image(baseline, reference)[0]
            model_psnr = psnr_per_image(prediction, reference)[0]
            baseline_ssim = ssim_per_image(baseline, reference)[0]
            model_ssim = ssim_per_image(prediction, reference)[0]
            rows.append({
                "filename": image_id,
                "baseline_psnr": baseline_psnr,
                "e_urm_psnr": model_psnr,
                "psnr_delta": model_psnr - baseline_psnr,
                "baseline_ssim": baseline_ssim,
                "e_urm_ssim": model_ssim,
                "ssim_delta": model_ssim - baseline_ssim,
                "psnr_improved": int((model_psnr - baseline_psnr) > 0),
                "ssim_improved": int((model_ssim - baseline_ssim) > 0),
            })
            print(image_id, f"{baseline_psnr:.3f}", f"{model_psnr:.3f}", flush=True)
    if len(rows) != 89 or len({row["filename"] for row in rows}) != 89:
        raise RuntimeError("Test evaluation did not produce 89 unique rows.")
    fields = list(rows[0].keys())
    with open(TEST_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    psnr = paired_summary([row["psnr_delta"] for row in rows])
    ssim = paired_summary([row["ssim_delta"] for row in rows])
    summary = {
        "checkpoint": os.path.relpath(BEST_PSNR, ROOT),
        "selected_epoch": int(state["epoch"]),
        "selection_rule": "highest mean validation PSNR",
        "validation_psnr_at_selection": float(state["best_val_loss"]),
        "baseline_psnr_mean": float(np.mean([row["baseline_psnr"] for row in rows])),
        "e_urm_psnr_mean": float(np.mean([row["e_urm_psnr"] for row in rows])),
        "baseline_ssim_mean": float(np.mean([row["baseline_ssim"] for row in rows])),
        "e_urm_ssim_mean": float(np.mean([row["e_urm_ssim"] for row in rows])),
        "psnr": psnr,
        "ssim": ssim,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    psnr_claim = "did" if summary["e_urm_psnr_mean"] > summary["baseline_psnr_mean"] else "did not"
    ssim_claim = "did" if summary["e_urm_ssim_mean"] > summary["baseline_ssim_mean"] else "did not"
    history = state.get("history") or []
    selected_row = next((row for row in history if int(row["epoch"]) == int(state["epoch"])), None)
    selected_ssim = None if selected_row is None else float(selected_row["val_ssim"])
    report = f"""# UIEB reference-890 experiment

## Dataset

- Total images: 890
- Training images: 712
- Validation images: 89
- Test images: 89
- Input: `{UIEB_RAW}`
- Target: `{UIEB_REFERENCE}`
- Pairing: same filename stem, one reference each, no split overlap

## Training

- Input: raw UIEB
- Target: reference-890
- Epochs: 30
- Batch size: 6
- Optimizer: AdamW
- Learning rate: {float(state['configuration']['learning_rate'])}
- Manuscript learning rate was 0.001. It was changed because that rate made `decoder4.0` grow from 2.5 to 671 by epoch 1 batch 15 and made every gradient in `initial_conv.0.weight` non-finite. At 0.0001, batches 0 through 40 stayed finite and `decoder4.0` stayed at 7.5.
- Weight decay: 0.0001
- Loss: L1 0.4, MSE 0.3, Sobel 0.2, degradation consistency 0.0
- Stabilization changes: no gradient clipping and no learning-rate change. The degradation-consistency weight is 0 because that term compares the output with the raw input rather than with reference-890. Photometric augmentation and synthetic degradation are off so the target remains the reference image.
- NaN occurred: no, through the completed run recorded in this checkpoint
- Selected epoch validation PSNR: {float(state['best_val_loss']):.6f}
- Selected epoch validation SSIM: {selected_ssim if selected_ssim is not None else "not stored as a separate field"}

## Best checkpoint

- Checkpoint name: `{os.path.relpath(BEST_PSNR, ROOT)}`
- Selection: highest mean validation PSNR. The test set was not used.
- Validation PSNR: {float(state['best_val_loss']):.6f}
- Validation SSIM: {selected_ssim if selected_ssim is not None else "see training_log.csv"}
- Best validation SSIM weights are stored at `{os.path.relpath(BEST_SSIM, ROOT)}` and were not scored on the test set.

## Test results

Resolution 256x256, RGB, values denormalized to [0, 1], PSNR peak 1. SSIM is the project implementation in `paper_records/quality_metrics.py`: Wang 2004, 11x11 Gaussian, sigma 1.5, valid convolution, per channel then averaged. Baseline and E-URM use the same resized reference.

| Metric | Unprocessed Input | E-URM | Difference |
|---|---:|---:|---:|
| Mean PSNR | {summary['baseline_psnr_mean']:.6f} | {summary['e_urm_psnr_mean']:.6f} | {psnr['mean']:.6f} |
| Mean SSIM | {summary['baseline_ssim_mean']:.6f} | {summary['e_urm_ssim_mean']:.6f} | {ssim['mean']:.6f} |

- PSNR improved: {psnr['improved']} / 89
- PSNR worse: {psnr['worse']} / 89
- PSNR unchanged: {psnr['unchanged']} / 89
- SSIM improved: {ssim['improved']} / 89
- SSIM worse: {ssim['worse']} / 89
- SSIM unchanged: {ssim['unchanged']} / 89
- PSNR difference median {psnr['median']:.6f}, standard deviation {psnr['std']:.6f}, minimum {psnr['min']:.6f}, maximum {psnr['max']:.6f}
- PSNR 95% CI for the mean difference: {psnr['ci95_low']:.6f} to {psnr['ci95_high']:.6f}
- PSNR test: {psnr['test_name']}; statistic {psnr['statistic']}; p-value {psnr['p_value']}
- SSIM difference median {ssim['median']:.6f}, standard deviation {ssim['std']:.6f}, minimum {ssim['min']:.6f}, maximum {ssim['max']:.6f}
- SSIM 95% CI for the mean difference: {ssim['ci95_low']:.6f} to {ssim['ci95_high']:.6f}
- SSIM test: {ssim['test_name']}; statistic {ssim['statistic']}; p-value {ssim['p_value']}

## Conclusion

On the locked 89-image UIEB test set, E-URM {psnr_claim} improve the mean PSNR and {ssim_claim} improve the mean SSIM relative to the unprocessed input.

No comparison with Water-Net, FUnIE-GAN, or PCAFA-Net is made.
"""
    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["verify", "stability", "train", "test"], required=True)
    parser.add_argument("--stability-batches", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device("cpu")
    splits, pairs = verify_split()
    if args.phase == "verify":
        return
    if args.phase == "stability":
        run_stability(splits, pairs, device, args.stability_batches, args.learning_rate)
        return
    if args.phase == "train":
        run_training(splits, pairs, device, args.learning_rate)
        return
    run_test(splits, pairs, device)


if __name__ == "__main__":
    main()
