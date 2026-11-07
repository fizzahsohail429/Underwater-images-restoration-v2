"""Ten-image UIEB comparison using the extracted reference-890 files.

PSNR and SSIM are filled only for the unprocessed input against its reference.
E-URM stays blank until runs/uieb/checkpoint.pt exists. Untrained weights are
not rendered.
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
from quality_metrics import psnr_per_image, ssim_per_image  # noqa: E402

RAW_DIR = UIEB_RAW
REF_DIR = os.path.join(ROOT, "archive", "reference-890")
TEST_LIST = os.path.join(ROOT, "paper_records", "uieb_test.txt")
OUT_DIR = os.path.join(ROOT, "paper_records", "uieb_workflow10")
N_IMAGES = 10
DIFFICULT_ID = "716_img_.png"
DIFFICULT_REASON = (
    "Strong blue cast, distant water with almost no visible structure, "
    "and a very dark foreground."
)


def read_ids():
    with open(TEST_LIST, encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return names[:N_IMAGES]


def find_reference(image_id):
    if not os.path.isdir(REF_DIR):
        return None
    stem = os.path.splitext(image_id)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(REF_DIR, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def unit_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def paired_scores(raw_bgr, reference_bgr):
    resized = reference_bgr.shape[:2] != raw_bgr.shape[:2]
    if resized:
        reference_bgr = cv2.resize(
            reference_bgr, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_AREA
        )
    prediction = unit_tensor(raw_bgr)
    target = unit_tensor(reference_bgr)
    return psnr_per_image(prediction, target)[0], ssim_per_image(prediction, target)[0], resized


def reference_panel(reference_path, size_hw):
    height, width = size_hw
    if reference_path is None:
        panel = np.full((height, width, 3), 235, dtype=np.uint8)
        cv2.putText(panel, "reference", (20, height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        cv2.putText(panel, "unavailable", (20, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        return panel
    image = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def label_bar(text, width):
    bar = np.full((36, width, 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def pending_panel(size_hw, line1, line2):
    height, width = size_hw
    panel = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(panel, line1, (12, height // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(panel, line2, (12, height // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return panel


def stack_scene(image_id, raw, reference, difficult):
    height, width = raw.shape[:2]
    panels = [
        ("input", raw),
        ("reference", reference),
        ("E-URM checkpoint not saved", pending_panel((height, width), "E-URM", "no checkpoint")),
    ]
    columns = []
    for title, image in panels:
        view = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        columns.append(np.vstack([label_bar(title, width), view]))
    scene = np.hstack(columns)
    note = "PSNR/SSIM: unprocessed input versus reference; E-URM not scored"
    if difficult:
        note += " | difficult input: strong color cast and low visibility"
    banner = label_bar(f"{image_id} | {note}", scene.shape[1])
    return np.vstack([banner, scene])


def write_xlsx(path, headers, rows):
    try:
        import openpyxl
    except ImportError:
        openpyxl = None
    if openpyxl is not None:
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "uieb_workflow10"
        sheet.append(headers)
        for row in rows:
            sheet.append([row[name] for name in headers])
        book.save(path)
        return
    # SpreadsheetML fallback so the file opens in Excel without extra packages.
    xml_rows = [
        '<ss:Row>' + ''.join(f'<ss:Cell><ss:Data ss:Type="String">{header}</ss:Data></ss:Cell>' for header in headers) + '</ss:Row>'
    ]
    for row in rows:
        cells = []
        for header in headers:
            value = "" if row[header] is None else str(row[header])
            value = value.replace("&", "&amp;").replace("<", "&lt;")
            cells.append(f'<ss:Cell><ss:Data ss:Type="String">{value}</ss:Data></ss:Cell>')
        xml_rows.append("<ss:Row>" + "".join(cells) + "</ss:Row>")
    xml = (
        '<?xml version="1.0"?>'
        '<?mso-application progid="Excel.Sheet"?>'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<Worksheet ss:Name="uieb_workflow10"><Table>'
        + "".join(xml_rows)
        + "</Table></Worksheet></Workbook>"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    image_ids = read_ids()
    rows = []
    thumbs = []
    with torch.inference_mode():
        for image_id in image_ids:
            raw_path = os.path.join(RAW_DIR, image_id)
            raw = cv2.imread(raw_path, cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(raw_path)
            reference_path = find_reference(image_id)
            reference = reference_panel(reference_path, raw.shape[:2])
            psnr = ssim = ""
            if reference_path:
                reference_bgr = cv2.imread(reference_path, cv2.IMREAD_COLOR)
                if reference_bgr is None:
                    raise FileNotFoundError(reference_path)
                psnr_value, ssim_value, resized = paired_scores(raw, reference_bgr)
                psnr = f"{psnr_value:.6f}"
                ssim = f"{ssim_value:.6f}"
                print(image_id, "reference", psnr, ssim, "resized" if resized else "same-size", flush=True)
            else:
                print(image_id, "reference missing", flush=True)
            difficult = image_id == DIFFICULT_ID
            scene = stack_scene(image_id, raw, reference, difficult)
            stem = os.path.splitext(image_id)[0]
            scene_path = os.path.join(OUT_DIR, f"{stem}_comparison.png")
            cv2.imwrite(scene_path, scene)
            thumb = cv2.resize(scene, (1280, int(scene.shape[0] * 1280 / scene.shape[1])), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)
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
    contact = np.vstack(thumbs)
    cv2.imwrite(os.path.join(OUT_DIR, "contact_sheet.png"), contact)
    headers = ["image_id", "dataset", "method", "PSNR", "SSIM"]
    csv_path = os.path.join(OUT_DIR, "uieb_workflow10.csv")
    xlsx_path = os.path.join(OUT_DIR, "uieb_workflow10.xlsx")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        csv_path = os.path.join(OUT_DIR, "uieb_workflow10_unlocked.csv")
        xlsx_path = os.path.join(OUT_DIR, "uieb_workflow10_unlocked.xlsx")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print("uieb_workflow10.csv is open elsewhere; wrote", csv_path)
    write_xlsx(xlsx_path, headers, rows)
    with open(os.path.join(OUT_DIR, "difficult_case.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            f"image_id: {DIFFICULT_ID}\n"
            f"reason: {DIFFICULT_REASON}\n"
            "checkpoint: none. E-URM PSNR and SSIM are blank.\n"
            "reference_directory: archive/reference-890 (official reference-890, 890 files)\n"
            "PSNR_SSIM: filled only for the unprocessed input against the matching reference, full resolution, range [0, 1], peak 1.\n"
        )
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
