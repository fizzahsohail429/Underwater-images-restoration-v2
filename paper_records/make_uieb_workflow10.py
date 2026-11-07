"""Ten-image comparison for the current manuscript figures.

Only UIEB archive/raw-890 is rendered. EUVP, UODD, and aquarium are readable
through dataset_catalog.py and are not included in these panels. PSNR and SSIM
stay blank unless a matching file exists in archive/reference-890.
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
from improved_underwater_restoration import build_paper_variant  # noqa: E402

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


def to_model_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
    return tensor


def from_model_tensor(output, size_hw):
    array = output.detach().float().cpu().numpy()[0]
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array * 0.5 + 0.5, 0.0, 1.0)
    array = (array * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    height, width = size_hw
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)


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


def stack_scene(image_id, raw, baseline, eurm, reference, difficult):
    height, width = raw.shape[:2]
    panels = [
        ("input", raw),
        ("baseline untrained", baseline),
        ("E-URM untrained", eurm),
        ("reference", reference),
    ]
    columns = []
    for title, image in panels:
        view = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        columns.append(np.vstack([label_bar(title, width), view]))
    scene = np.hstack(columns)
    note = "UNTRAINED INITIALIZATION, not a trained result"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = build_paper_variant("capacity-matched").to(device).eval()
    eurm_model = build_paper_variant("e-urm").to(device).eval()
    rows = []
    thumbs = []
    with torch.inference_mode():
        for image_id in image_ids:
            raw_path = os.path.join(RAW_DIR, image_id)
            raw = cv2.imread(raw_path, cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(raw_path)
            tensor = to_model_tensor(raw).to(device)
            baseline = from_model_tensor(baseline_model(tensor), raw.shape[:2])
            eurm = from_model_tensor(eurm_model(tensor), raw.shape[:2])
            reference_path = find_reference(image_id)
            reference = reference_panel(reference_path, raw.shape[:2])
            difficult = image_id == DIFFICULT_ID
            scene = stack_scene(image_id, raw, baseline, eurm, reference, difficult)
            stem = os.path.splitext(image_id)[0]
            scene_path = os.path.join(OUT_DIR, f"{stem}_comparison.png")
            cv2.imwrite(scene_path, scene)
            thumb = cv2.resize(scene, (1280, int(scene.shape[0] * 1280 / scene.shape[1])), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)
            for method in (
                "capacity-matched control (untrained initialization)",
                "E-URM (untrained initialization)",
            ):
                rows.append({
                    "image_id": image_id,
                    "dataset": "UIEB raw-890 test split",
                    "method": method,
                    "PSNR": "",
                    "SSIM": "",
                })
            print(image_id, "reference", "found" if reference_path else "unavailable")
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
            "checkpoint: none\n"
            "weights: untrained initialization\n"
            f"reference_directory_present: {os.path.isdir(REF_DIR)}\n"
            "PSNR_SSIM: left blank because no paired reference file was available\n"
        )
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
