# CVDL HW2 — Training Pipeline

This repository contains the main notebook (`main.ipynb`) used to train a YOLOv8+BBN-lite model and generate a submission file. 

> **environment:** Python 3.12,and **GPU (CUDA)** is strongly recommended.

## 1. Create environment

- Create a folder `hw2_NTHU_114064545` to store related files
- Open cmd (or Powershell) and enter `hw2_NTHU_114064545`
```bash!
> cd .../hw2_NTHU_114064545
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
│   ├── train/ 760 images(img0001 ~ img0760)
│   ├── val/ 190 images(img0761 ~ img0950)
│   └── test/ 550 images
└── labels/
    ├── train/ 760 txts(img0001 ~ img0760)  
    └── val/ 190 txts(img0761 ~ img0950)
data.yaml
```

- Inside `data.yaml` :
```yaml
train: images/train
val: images/val

nc: 4   # number of categories
names: [car, hov, person, motorcycle]
```

## 4. Training - Run the notebook `main.ipynb`

- Open `main.ipynb` and run all cells top-to-bottom. (Excluding cell 2 and 3 which are in order to choose a suitable vision of model)
- The notebook will:
   - install dependencies (`ultralytics`, `opencv-python-headless`, etc.)
   - initialize a YOLOv8 model
   - transform the original label to YOLO form (normalize)
   - define callback and some functions for BBN-lite Module
   - start training with `data.yaml` 
   ```python
    from ultralytics import YOLO

    model = YOLO('yolov8x.yaml')
    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_train_epoch_start', on_train_epoch_start) 

    results = model.train(
        data="data.yaml",
        epochs=150,
        imgsz=1280,
        batch=8,
        device=0,
        pretrained=False,
        optimizer="adamw",
        lr0=0.002,     
        lrf=0.01,       
        weight_decay=0.01,
        cache="ram",
        workers=2,
        patience=50, 
        amp=True,
        cos_lr=True,
        mosaic=0.0, 
        close_mosaic=0,
    )
    ```
   - export predictions
   ```python
    from ultralytics import YOLO

    model = YOLO('/yourpath/runs/detect/train/weights/best.pt')  # 你訓練好的權重
    results = model.predict(
        source="yourpath/data/images/test",
        imgsz=1536,
        conf=0.0005,
        max_det=1200,
        augment=True,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/predict",
        name="bbn_y8n"
)
    ```


## 5. Make submission

The notebook calls `make_submission.py` to turn predictions into CSV:
```bash
!python /yourpath/make_submission.py
```
:::warning
**Adjust paths inside the script (TEST_IMAGE_DIR / PRED_LABEL_DIR / OUTPUT_CSV)**  so it reads your latest predictions and writes a CSV to the expected location.
:::

## 6. License
This homework scaffold uses Ultralytics YOLOv8. Please consult Ultralytics' license for model and CLI usage terms.
