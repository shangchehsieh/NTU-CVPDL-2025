# CVDL HW3 — Training Pipeline

This project implements Denoising Diffusion Probabilistic Models (DDPM) from scratch using PyTorch, trained on MNIST without any pretrained weights or external datasets.
All model components (UNet, diffusion process, sampling) are implemented manually.

> **Environment:** Python 3.12,and **GPU (CUDA)** is strongly recommended.
> **The model achieves**:
> - FID = 25.53 (best checkpoint)
> - Generated 10,000 MNIST-style 28×28 images

## 1. Environment
Install required packages:
```bash
!pip install -r requirements.txt
```
All packages are within the course-approved whitelist.

## 2. Project Structure

- training data
```
mnist/
└── train/
    ├── 00001.png
    ├── ...
    └── 60000.png
```

## 3. Training - Run the notebook `main.ipynb`
Train DDPM on MNIST from scratch:

```python
  args = SimpleNamespace(
    mode="train", data=".hw3_NTHU_114064545/mnist", ckpt_dir="./ckpt", out_dir="./gen",
    epochs=150, batch=128, steps=1000, lr=8e-5, num=10000, s_batch=512
)
train(args)
```
This will:
- Download MNIST (if not present)
- Train UNetSmall-based DDPM for 150 epochs
- Save:
  - `best.pth` (best checkpoint)
  - `last.pth` (final checkpoint)


## 4. Sampling - Run the notebook `main.ipynb`

Run:
```bash
args.mode = "sample"
sample(args)
```
This will:
- Load `best.pth`
- Run reverse diffusion (T = 1000)
- Save 10,000 generated images to `/img_114064545`
## 5. FID Evaluation - Run the notebook `main.ipynb`
- With the training dataset
```bash
!python -m pytorch_fid ./HW3_NTHU_114064545/img_114064545 ./HW3_NTHU_114064545/mnist/train
```
- With precalculated mean and covariance
```bash
!python -m pytorch_fid ./HW3_NTHU_114064545/img_114064545 ./HW3_NTHU_114064545/mnist.npz
```
- Note: If you run them in a terminal, simply remove the leading `!`.

## 6. Visualization

Run:
```bash
device = "cuda" if torch.cuda.is_available() else "cpu"
diff = Diffusion(T=1000, device=device)
model = UNetSmall().to(device)
model.load_state_dict(torch.load("./HW3_NTHU_114064545/ckpt/best.pth", map_location=device))

save_diffusion_process_grid(diff, model,
                            num_samples=8,
                            num_steps=8,
                            filename="diffusion_process.png")
```
This will save: `diffusion_process.png` (best checkpoint)
