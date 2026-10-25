# ============================================================
# STEP 1 — download the subset once, in parallel
# ============================================================
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # needs: pip install hf_transfer

from huggingface_hub import snapshot_download

N_IMAGES = 80
LOCAL    = "/kaggle/working/we3ds"

patterns = []
for i in range(N_IMAGES):
    patterns += [f"data/img_{i:05d}.png",
                 f"fields/depth_refined/img_{i:05d}.png"]

snapshot_download(
    "Voxel51/WE3DS",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=LOCAL,
    max_workers=8,
)
print("downloaded to", LOCAL)

# ============================================================
# STEP 2 — Depth Anything V2 vs measured depth
# ============================================================
import numpy as np, pandas as pd, torch, glob
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.stats import spearmanr

MODEL     = "depth-anything/Depth-Anything-V2-Small-hf"
GRID      = 8      # 8 x 8 = 64 patches per image
MIN_VALID = 0.5    # a patch is used only if >50% of its GT pixels are valid

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()

rows, per_image = [], []
names = sorted(os.path.basename(p) for p in glob.glob(f"{LOCAL}/data/*.png"))
print(f"{len(names)} image pairs found\n")

for name in names:
    rgb = Image.open(f"{LOCAL}/data/{name}").convert("RGB")
    gt  = np.array(Image.open(f"{LOCAL}/fields/depth_refined/{name}")).astype(np.float32)
    H, W = gt.shape

    # Depth Anything V2 — identical to the detection pipeline
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
    )[0, 0].numpy()
    lo, hi = float(np.nanmin(pred)), float(np.nanmax(pred))
    z_pred = (pred - lo) / (hi - lo) if hi > lo else np.zeros_like(pred)

    # ROI medians on both sides, exactly as Z_rel is computed in the paper
    hs, ws = H // GRID, W // GRID
    zp, zg = [], []
    for r in range(GRID):
        for c in range(GRID):
            sl = (slice(r * hs, (r + 1) * hs), slice(c * ws, (c + 1) * ws))
            g = gt[sl]; m = g > 0
            if m.sum() < MIN_VALID * g.size:
                continue
            zp.append(float(np.median(z_pred[sl][m])))
            zg.append(float(np.median(g[m])))

    if len(zp) < 10:
        print(f"{name}  too few valid patches"); continue

    rho, _ = spearmanr(zp, zg)
    per_image.append(rho)
    rows += [{"image": name, "z_pred": a, "gt_mm": b} for a, b in zip(zp, zg)]
    print(f"{name}  patches={len(zp):>3}  rho={rho:+.3f}")

pi = np.array(per_image)
pd.DataFrame(rows).to_csv("/kaggle/working/we3ds_depth_validation.csv", index=False)

print("\n" + "=" * 64)
print(f"images used            : {len(pi)}")
print(f"ROI pairs              : {len(rows)}")
print(f"per-image Spearman rho : mean {pi.mean():+.3f}   median {np.median(pi):+.3f}")
print(f"                         IQR [{np.percentile(pi,25):+.3f}, {np.percentile(pi,75):+.3f}]")
print(f"negative rho in        : {100*(pi<0).mean():.0f}% of images")
print(f"|rho| > 0.5 in         : {100*(np.abs(pi)>0.5).mean():.0f}% of images")
print("=" * 64)
print("GT is measured distance (mm): larger = farther.")
print("A strong NEGATIVE rho means the model's normalized output is larger for")
print("NEARER surfaces, i.e. it behaves as relative inverse depth.")
