import os
import csv
import glob
import re
from pathlib import Path
from PIL import Image

# path
TEST_IMAGE_DIR = "/home/shangche45/hw2_NTHU_114064545/data/images/test"
PRED_LABEL_DIR = "/home/shangche45/hw2_NTHU_114064545/runs/predict/bbn_y8n/labels"
OUTPUT_CSV     = "/home/shangche45/hw2_NTHU_114064545/submission.csv"

def denorm_xywh_to_ltrb(xc, yc, w, h, img_w, img_h):
    """
    (left, top, width, height)
    """
    x_center_px = xc * img_w
    y_center_px = yc * img_h
    w_px = w * img_w
    h_px = h * img_h
    left = x_center_px - w_px / 2
    top  = y_center_px - h_px / 2
    return left, top, w_px, h_px

def load_predictions(label_path, img_w, img_h):
    """
    txt -> list of tuples: (conf, left, top, width, height, cls)
    """
    preds = []
    if not os.path.exists(label_path):
        return preds

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])

            left, top, ww, hh = denorm_xywh_to_ltrb(xc, yc, w, h, img_w, img_h)
            preds.append((conf, left, top, ww, hh, cls))
    return preds

def natural_numeric_key(p: str) -> int:
    """
   "img0523.png" -> 523
    """
    stem = Path(p).stem
    digits = re.sub(r'\D', '', stem)
    return int(digits) if digits else 0

def main():

    exts = ("*.jpg", "*.jpeg", "*.png")
    test_images = []
    for ext in exts:
        test_images.extend(glob.glob(os.path.join(TEST_IMAGE_DIR, ext)))


    test_images = sorted(test_images, key=natural_numeric_key)

    rows = []

    for idx, img_path in enumerate(test_images, start=1):
        stem = Path(img_path).stem
        label_path = os.path.join(PRED_LABEL_DIR, f"{stem}.txt")


        with Image.open(img_path) as im:
            img_w, img_h = im.size  # (width, height)

        preds = load_predictions(label_path, img_w, img_h)

        if preds:
            parts = []
            for conf, left, top, w, h, cls in preds:
                parts.extend([
                    f"{conf:.6f}",
                    f"{left:.2f}",
                    f"{top:.2f}",
                    f"{w:.2f}",
                    f"{h:.2f}",
                    str(int(cls)),
                ])
            prediction_string = " ".join(parts)
        else:
            prediction_string = ""

        rows.append([idx, prediction_string])  # idx（1..550）

    # generate CSV file
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
