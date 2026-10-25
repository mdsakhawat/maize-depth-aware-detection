"""
Inference-time benchmark for the depth-aware maize pipeline.

Measures, on a single GPU at batch size 1:
  A. YOLO11n detection only
  B. Depth Anything V2 only
  C. the two combined

Run this on the same GPU reported in the Methods section (NVIDIA Tesla T4).
On Kaggle choose Accelerator = GPU T4 x2 (it still reports a single T4 as
device 0) -- not P100, or the numbers will not match the paper.
"""

from pathlib import Path
import statistics
import time

import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------
# PATHS - edit these two
# ----------------------------------------------------------------------------
WEIGHTS   = Path("/kaggle/input/yolo11n-maize/best.pt")
TEST_DIR  = Path("/kaggle/working/maize-disease-detection/images/test")

N_IMAGES  = 100    # images to time over
N_WARMUP  = 10     # discarded warm-up iterations
IMGSZ     = 640
CONF      = 0.25

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("torch :", torch.__version__)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
imgs = sorted(p for p in TEST_DIR.rglob("*") if p.suffix.lower() in IMG_EXT)
if not imgs:
    raise SystemExit(f"No images found under {TEST_DIR}")
imgs = imgs[: N_IMAGES + N_WARMUP]
print(f"timing over {len(imgs) - N_WARMUP} images (plus {N_WARMUP} warm-up)")


def summarise(name, ms):
    ms = np.asarray(ms, dtype=float)
    print(f"{name:<34}{ms.mean():>8.1f}{ms.std(ddof=1):>9.1f}"
          f"{np.median(ms):>9.1f}{1000/ms.mean():>9.1f}")
    return ms


print(f"\n{'stage':<34}{'mean ms':>8}{'SD':>9}{'median':>9}{'FPS':>9}")

# ============================================================================
# A. YOLO11n
# ============================================================================
from ultralytics import YOLO

model = YOLO(str(WEIGHTS))

pre, inf, post, wall = [], [], [], []
for i, p in enumerate(imgs):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    r = model.predict(str(p), imgsz=IMGSZ, conf=CONF, device=0, verbose=False)[0]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    if i < N_WARMUP:
        continue
    s = r.speed                      # Ultralytics own breakdown, in ms
    pre.append(s["preprocess"])
    inf.append(s["inference"])
    post.append(s["postprocess"])
    wall.append((t1 - t0) * 1000)

summarise("YOLO11n preprocess", pre)
yolo_inf = summarise("YOLO11n inference", inf)
summarise("YOLO11n postprocess", post)
yolo_total = np.asarray(pre) + np.asarray(inf) + np.asarray(post)
summarise("YOLO11n total (pre+inf+post)", yolo_total)
summarise("YOLO11n wall clock per image", wall)

# ============================================================================
# B. Depth Anything V2
# ============================================================================
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
proc = AutoImageProcessor.from_pretrained(MODEL_ID)
dmodel = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to("cuda").eval()

dpre, dinf, dwall = [], [], []
for i, p in enumerate(imgs):
    im = Image.open(p).convert("RGB")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = proc(images=im, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        _ = dmodel(**batch).predicted_depth
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if i < N_WARMUP:
        continue
    dpre.append((t1 - t0) * 1000)
    dinf.append((t2 - t1) * 1000)
    dwall.append((t2 - t0) * 1000)

summarise("DepthAnythingV2 preprocess", dpre)
summarise("DepthAnythingV2 inference", dinf)
depth_total = summarise("DepthAnythingV2 total", dwall)

# ============================================================================
# C. Combined
# ============================================================================
combined = yolo_total + np.asarray(dwall)
summarise("Detection + depth combined", combined)

print(f"\ndepth branch adds {depth_total.mean():.1f} ms on top of "
      f"{yolo_total.mean():.1f} ms, i.e. "
      f"{depth_total.mean()/yolo_total.mean():.1f}x the detection cost")
print(f"images timed: {len(yolo_total)}   imgsz: {IMGSZ}   batch size: 1")
