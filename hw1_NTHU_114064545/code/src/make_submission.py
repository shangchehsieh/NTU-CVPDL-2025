import os
import csv
import glob
from pathlib import Path

#PATH:
IMG_W, IMG_H = 640, 360   # image size
TEST_IMAGE_DIR = "/yourpath/images/test"
PRED_LABEL_DIR = "/yourpath/predict/labels"
OUTPUT_CSV = "/yourpath/submission.csv"

def denorm_xywh_to_ltrb(xc, yc, w, h, img_w, img_h):
    """ (left,top,width,height)"""
    x_center_px = xc * img_w
    y_center_px = yc * img_h
    w_px = w * img_w
    h_px = h * img_h
    left = x_center_px - w_px / 2
    top  = y_center_px - h_px / 2
    return left, top, w_px, h_px

def load_predictions(label_path):
    """read txt -> list(conf, left, top, w, h, cls)"""
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
            left, top, ww, hh = denorm_xywh_to_ltrb(xc, yc, w, h, IMG_W, IMG_H)
            preds.append((
                conf,
                max(0, int(round(left))),
                max(0, int(round(top))),
                int(round(ww)),
                int(round(hh)),
                cls
            ))
    return preds

def main():
    # collect testing images
    exts = ("*.jpg", "*.jpeg", "*.png")
    test_images = []
    for ext in exts:
        test_images.extend(glob.glob(os.path.join(TEST_IMAGE_DIR, ext)))
    test_images = sorted(test_images, key=lambda p: int(Path(p).stem))

    rows = []
    for img_path in test_images:
        stem = Path(img_path).stem   # "00000001"
        image_id = stem              
        label_path = os.path.join(PRED_LABEL_DIR, f"{stem}.txt")

        preds = load_predictions(label_path)

        if preds:
            # conf left top width height class
            parts = []
            for conf, left, top, w, h, cls in preds:
                parts.extend([f"{conf:.6f}", str(left), str(top), str(w), str(h), str(cls)])
            prediction_string = " ".join(parts)
        else:
            prediction_string = ""

        rows.append([image_id, prediction_string])

    # write submission.csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])
        writer.writerows(rows)

    print(f" Wrote {OUTPUT_CSV} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
