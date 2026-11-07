"""Paired PSNR and SSIM of the unprocessed UIEB test input against reference-890.

E-URM is not scored here. A blank row is written for it so the table keeps one
row per image and method without inventing a trained score.
"""

import csv
import os
import sys

import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset_catalog import UIEB_RAW, UIEB_REFERENCE  # noqa: E402
from quality_metrics import interval, psnr_per_image, ssim_per_image  # noqa: E402

TEST_LIST = os.path.join(ROOT, "paper_records", "uieb_test.txt")
OUT_CSV = os.path.join(ROOT, "paper_records", "uieb_test_unprocessed_vs_reference.csv")
OUT_TXT = os.path.join(ROOT, "paper_records", "uieb_test_unprocessed_vs_reference.txt")


def unit_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    array = rgb.astype("float32") / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def find_reference(image_id):
    stem = os.path.splitext(image_id)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(UIEB_REFERENCE, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def main():
    with open(TEST_LIST, encoding="utf-8") as handle:
        image_ids = [line.strip() for line in handle if line.strip()]
    rows = []
    psnr_values = []
    ssim_values = []
    missing = []
    resized_count = 0
    with torch.inference_mode():
        for image_id in image_ids:
            raw = cv2.imread(os.path.join(UIEB_RAW, image_id), cv2.IMREAD_COLOR)
            reference_path = find_reference(image_id)
            if raw is None or reference_path is None:
                missing.append(image_id)
                psnr = ssim = ""
            else:
                reference = cv2.imread(reference_path, cv2.IMREAD_COLOR)
                if reference.shape[:2] != raw.shape[:2]:
                    reference = cv2.resize(reference, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
                    resized_count += 1
                psnr_value = psnr_per_image(unit_tensor(raw), unit_tensor(reference))[0]
                ssim_value = ssim_per_image(unit_tensor(raw), unit_tensor(reference))[0]
                psnr = f"{psnr_value:.6f}"
                ssim = f"{ssim_value:.6f}"
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
            rows.append({
                "image_id": image_id,
                "dataset": "UIEB test",
                "method": "unprocessed input",
                "PSNR": psnr,
                "SSIM": ssim,
            })
            rows.append({
                "image_id": image_id,
                "dataset": "UIEB test",
                "method": "E-URM",
                "PSNR": "",
                "SSIM": "",
            })
            print(image_id, psnr, ssim, flush=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "dataset", "method", "PSNR", "SSIM"])
        writer.writeheader()
        writer.writerows(rows)
    psnr_mean, psnr_low, psnr_high = interval(psnr_values)
    ssim_mean, ssim_low, ssim_high = interval(ssim_values)
    with open(OUT_TXT, "w", encoding="utf-8") as handle:
        handle.write(
            "Comparison: unprocessed archive/UIEB input versus archive/reference-890.\n"
            "Range [0, 1], PSNR peak 1. SSIM is Wang 2004, 11x11 Gaussian, sigma 1.5, valid convolution.\n"
            "This is not an E-URM score. E-URM rows in the CSV are blank because no checkpoint is selected yet.\n"
            f"scored: {len(psnr_values)} of {len(image_ids)} test images\n"
            f"missing reference or raw: {len(missing)}\n"
            f"reference resized to the raw size: {resized_count}\n"
            f"PSNR mean {psnr_mean:.6f}, 95% interval [{psnr_low:.6f}, {psnr_high:.6f}]\n"
            f"SSIM mean {ssim_mean:.6f}, 95% interval [{ssim_low:.6f}, {ssim_high:.6f}]\n"
        )
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
