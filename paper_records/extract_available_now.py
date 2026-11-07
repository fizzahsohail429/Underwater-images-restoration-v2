"""Extract the measurements that the current code can support.

Reference PSNR/SSIM stay blank when no paired reference file exists.
Synthetic scores are written to a separate CSV.
"""

import csv
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset_catalog import UIEB_RAW  # noqa: E402
from improved_underwater_restoration import (  # noqa: E402
    _degradation_rng,
    create_advanced_degradation,
    create_improved_transforms,
)
from quality_metrics import denormalize, psnr_per_image, ssim_per_image  # noqa: E402
from dataset_catalog import UIEB_REFERENCE  # noqa: E402

OUT = os.path.join(ROOT, "paper_records", "available_now")
TEST_LIST = os.path.join(ROOT, "paper_records", "uieb_test.txt")
N_IMAGES = 15
REFERENCE_CANDIDATES = [
    UIEB_REFERENCE,
    os.path.join(ROOT, "archive", "UIEB", "reference-890"),
    os.path.join(ROOT, "archive", "UIEB", "reference"),
]


def read_ids():
    with open(TEST_LIST, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()][:N_IMAGES]


def find_reference(image_id):
    stem = os.path.splitext(image_id)[0]
    for folder in REFERENCE_CANDIDATES:
        if not os.path.isdir(folder):
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(folder, stem + ext)
            if os.path.isfile(path):
                return path
    return None


def label_bar(text, width):
    bar = np.full((36, width, 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def missing_panel(height, width, line1, line2):
    panel = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(panel, line1, (12, height // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(panel, line2, (12, height // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return panel


def main():
    os.makedirs(OUT, exist_ok=True)
    image_ids = read_ids()
    _, val_transform = create_improved_transforms(256)
    reference_rows = []
    synthetic_rows = []
    thumbs = []
    for image_id in image_ids:
        raw_path = os.path.join(UIEB_RAW, image_id)
        raw_bgr = cv2.imread(raw_path, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise FileNotFoundError(raw_path)
        raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        degraded, _params = create_advanced_degradation(raw, _degradation_rng(raw_path, False))
        augmented = val_transform(image=raw, degraded=degraded)
        target = denormalize(augmented["image"].unsqueeze(0))
        prediction = denormalize(augmented["degraded"].unsqueeze(0))
        psnr = psnr_per_image(prediction, target)[0]
        ssim = ssim_per_image(prediction, target)[0]
        synthetic_rows.append({
            "image_id": image_id,
            "dataset": "UIEB test",
            "method": "unprocessed synthetic input",
            "evaluation": "synthetic degradation; target is the original image",
            "PSNR": f"{psnr:.6f}",
            "SSIM": f"{ssim:.6f}",
        })
        reference_path = find_reference(image_id)
        raw_psnr = raw_ssim = ""
        if reference_path:
            reference_bgr = cv2.imread(reference_path, cv2.IMREAD_COLOR)
            if reference_bgr.shape[:2] != raw_bgr.shape[:2]:
                reference_bgr = cv2.resize(
                    reference_bgr, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_AREA
                )
            reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            raw_rgb = raw.astype(np.float32) / 255.0
            raw_tensor = torch.from_numpy(np.transpose(raw_rgb, (2, 0, 1))).unsqueeze(0)
            reference_tensor = torch.from_numpy(np.transpose(reference_rgb, (2, 0, 1))).unsqueeze(0)
            raw_psnr = f"{psnr_per_image(raw_tensor, reference_tensor)[0]:.6f}"
            raw_ssim = f"{ssim_per_image(raw_tensor, reference_tensor)[0]:.6f}"
        reference_rows.append({
            "image_id": image_id,
            "dataset": "UIEB test",
            "method": "unprocessed input",
            "PSNR": raw_psnr,
            "SSIM": raw_ssim,
        })
        reference_rows.append({
            "image_id": image_id,
            "dataset": "UIEB test",
            "method": "E-URM",
            "PSNR": "",
            "SSIM": "",
        })
        height, width = raw_bgr.shape[:2]
        view_w = 320
        view_h = max(1, int(height * view_w / width))
        raw_view = cv2.resize(raw_bgr, (view_w, view_h), interpolation=cv2.INTER_AREA)
        reference_view = missing_panel(view_h, view_w, "reference", "not on disk")
        if reference_path:
            reference_view = cv2.resize(cv2.imread(reference_path), (view_w, view_h), interpolation=cv2.INTER_AREA)
        enhanced_view = missing_panel(view_h, view_w, "E-URM", "no checkpoint")
        columns = [
            np.vstack([label_bar("original input", view_w), raw_view]),
            np.vstack([label_bar("reference", view_w), reference_view]),
            np.vstack([label_bar("E-URM output", view_w), enhanced_view]),
        ]
        scene = np.vstack([label_bar(image_id, view_w * 3), np.hstack(columns)])
        cv2.imwrite(os.path.join(OUT, f"{os.path.splitext(image_id)[0]}_comparison.png"), scene)
        thumbs.append(cv2.resize(scene, (960, int(scene.shape[0] * 960 / scene.shape[1])), interpolation=cv2.INTER_AREA))
        print(image_id, "synthetic_psnr", f"{psnr:.3f}", "reference", bool(reference_path), flush=True)
    cv2.imwrite(os.path.join(OUT, "contact_sheet.png"), np.vstack(thumbs))
    write_csv(os.path.join(OUT, "uieb_reference_metrics.csv"), ["image_id", "dataset", "method", "PSNR", "SSIM"], reference_rows)
    write_csv(
        os.path.join(OUT, "uieb_synthetic_metrics.csv"),
        ["image_id", "dataset", "method", "evaluation", "PSNR", "SSIM"],
        synthetic_rows,
    )
    print("wrote", OUT, flush=True)


def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
