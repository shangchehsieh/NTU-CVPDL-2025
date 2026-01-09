---
title: README for CVDLhw1

---

# CVDL HW1 — YOLOv8 Training Pipeline

This repository contains the main notebook (`main.ipynb`) used to train a YOLOv8 model  (I've trained YOLOv11 too) and generate a submission file. 

> **environment:** Python 3.12, pytorch launched time: 25.08-py3-latest, and **GPU (CUDA)** is strongly recommended.

## 1. Create environment

- Create a folder `hw1_NTHU_114064545` to store related files
- Open cmd (or Powershell) and enter `hw1_NTHU_114064545`
```bash!
> cd .../hw1_NTHU_114064545
```
- Create a virtual environment and run it with Jupyter
```bash!
> uv run jupyter notebook
```

## 2. Install Python packages
- Include:
  - **Core training stack** (ultralytics, torch, torchvision)
  - Inference & image I/O
  - Other Utilities
```bash
!pip install -r requirements.txt
```

## 3. Dataset layout
:::warning
Please **update your** `data.yaml` **to point to your image/label folders (important!)**.
A typical structure is:
:::

```
dataset/
├── images/
│   ├── train/ 
│   ├── val/
│   └── test/
└── labels/
    ├── train/   
    └── val/
data.yaml
```

- Inside `data.yaml` :
```yaml
train: images/train
val: images/val

nc: 1   # number of classes
names: "swine"
```

## 4. Training - Run the notebook `main.ipynb`

- Open `main.ipynb` and run all cells top-to-bottom. (Excluding cell 2 and 3 which are in order to choose a suitable vision of model)
- The notebook will:
   - install dependencies (`ultralytics`, `opencv-python-headless`, etc.)
   - initialize a YOLOv8(or VOLO11) model
   - start training with `data.yaml` and `hyper_pig.yaml` (for data augmentation)
   ```python
    results = model.train(
    data="/.../data.yaml",
    epochs=200,
    imgsz=960,
    batch=0,
    device=0,
    pretrained=False,
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.0001,
    weight_decay=0.01,
    cfg="/.../hyper_pig.yaml", #data augmentation
    cache="ram",
    workers=2,
    patience=30,
    amp=True,
    cos_lr=True,
    )
    ```
   - export predictions
   ```python
    results = model.predict(
    source="/.../images/test", 
    imgsz=960,
    conf=0.001,
    iou=0.7,
    max_det=150,
    augment=True,
    agnostic_nms=True,
    half=True,
    save=True,      # save images
    save_txt=True,  
    save_conf=True,
    stream=False,
    project="runs/detect", name="predict", exist_ok=True
    )
    for r in results:
        boxes = r.boxes      
        confs = boxes.conf   
        classes = boxes.cls 
    ```
   - generate a submission CSV
   ```python
    !python /.../make_submission.py
    ```

## 5. Make submission

The notebook calls `make_submission.py` to turn predictions into CSV:
```bash
!python /home/shangche45/CVDLhw1/dataset/make_submission.py
```
:::warning
**Adjust paths inside the script (TEST_IMAGE_DIR / PRED_LABEL_DIR / OUTPUT_CSV)**  so it reads your latest predictions and writes a CSV to the expected location.
:::

## 6. License
This homework scaffold uses Ultralytics YOLOv8. Please consult Ultralytics' license for model and CLI usage terms.
