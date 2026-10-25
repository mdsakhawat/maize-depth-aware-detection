"""
ERROR ANALYSIS FOR THE YOLO11n DETECTOR

Separates every prediction and every ground-truth box on the test split into
one of six outcomes, so that false positives, localization errors and missed
detections can be reported separately rather than lumped together:

  TP          correct class, IoU >= 0.5 with an unmatched ground-truth box
  Duplicate   correct class and IoU >= 0.5, but that box was already matched
  Loc         correct class, 0.1 <= IoU < 0.5            (localization error)
  Cls         wrong class, IoU >= 0.5 with some box       (classification error)
  Both        wrong class and 0.1 <= IoU < 0.5
  Bkg         IoU < 0.1 with every box                    (background false positive)
  Miss        ground-truth box with no matching detection

It also lists the images contributing the most of each error type, as
candidates for the failure examples the review asks for.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
WEIGHTS   = Path("/kaggle/working/reconstructed_best.pt")
TEST_IMG  = Path("/kaggle/working/maize-disease-detection/images/test")
TEST_LBL  = Path("/kaggle/working/maize-disease-detection/labels/test")
OUT_CSV   = Path("/kaggle/working/error_analysis.csv")

CONF      = 0.25    # same inference setting as Tables 3 and 4
IOU_TP    = 0.50    # match threshold
IOU_LOC   = 0.10    # below this a prediction is treated as background
IMGSZ     = 640
# ----------------------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and every box in b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


model = YOLO(str(WEIGHTS))
names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
nc = len(names)

images = sorted(p for p in TEST_IMG.iterdir() if p.suffix.lower() in IMG_EXT)
print(f"test images: {len(images)}   conf: {CONF}   IoU: {IOU_TP}")

pred_counts = defaultdict(lambda: defaultdict(int))   # [class][outcome]
gt_counts = defaultdict(lambda: defaultdict(int))     # [class][TP|Miss]
confusions = defaultdict(int)                         # (true, predicted) for Cls errors
per_image = []

for ip, img_path in enumerate(images):
    r = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF,
                      device=0, verbose=False)[0]
    H, W = r.orig_shape

    # ground truth, YOLO normalised xywh -> pixel xyxy
    lbl = TEST_LBL / (img_path.stem + ".txt")
    g_cls, g_box = [], []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            f = line.split()
            if len(f) < 5:
                continue
            c, xc, yc, w, h = int(f[0]), *map(float, f[1:5])
            g_cls.append(c)
            g_box.append([(xc - w / 2) * W, (yc - h / 2) * H,
                          (xc + w / 2) * W, (yc + h / 2) * H])
    g_cls = np.array(g_cls, dtype=int)
    g_box = np.array(g_box, dtype=float).reshape(-1, 4)

    if r.boxes is None or len(r.boxes) == 0:
        p_cls = np.zeros(0, dtype=int)
        p_box = np.zeros((0, 4))
        p_conf = np.zeros(0)
    else:
        p_cls = r.boxes.cls.cpu().numpy().astype(int)
        p_box = r.boxes.xyxy.cpu().numpy()
        p_conf = r.boxes.conf.cpu().numpy()

    ious = iou_matrix(p_box, g_box)
    gt_taken = np.zeros(len(g_cls), dtype=bool)
    img_err = defaultdict(int)

    # highest-confidence predictions claim their ground-truth box first
    for pi in np.argsort(-p_conf):
        pc = p_cls[pi]
        row = ious[pi] if ious.size else np.zeros(0)
        same = np.where(g_cls == pc)[0] if len(g_cls) else np.zeros(0, dtype=int)

        best_same, best_same_i = 0.0, -1
        if len(same):
            j = same[np.argmax(row[same])]
            best_same, best_same_i = row[j], j
        best_any = row.max() if row.size else 0.0

        if best_same >= IOU_TP:
            if not gt_taken[best_same_i]:
                gt_taken[best_same_i] = True
                outcome = "TP"
            else:
                outcome = "Duplicate"
        elif best_same >= IOU_LOC:
            outcome = "Loc"
        elif best_any >= IOU_TP:
            outcome = "Cls"
            confusions[(int(g_cls[int(np.argmax(row))]), int(pc))] += 1
        elif best_any >= IOU_LOC:
            outcome = "Both"
        else:
            outcome = "Bkg"

        pred_counts[int(pc)][outcome] += 1
        if outcome != "TP":
            img_err[outcome] += 1

    for gi, gc in enumerate(g_cls):
        gt_counts[int(gc)]["TP" if gt_taken[gi] else "Miss"] += 1
        if not gt_taken[gi]:
            img_err["Miss"] += 1

    per_image.append((img_path.name, dict(img_err)))
    if (ip + 1) % 300 == 0:
        print(f"  {ip + 1}/{len(images)}")


# ============================================================================
# PER-CLASS TABLE
# ============================================================================
print("\n" + "=" * 92)
print("ERROR BREAKDOWN BY CLASS")
print("=" * 92)
cols = ["TP", "Bkg", "Loc", "Cls", "Both", "Duplicate"]
print(f"{'class':<22}{'GT':>6}{'preds':>7}" + "".join(f"{c:>11}" for c in cols)
      + f"{'Miss':>8}")
tot = defaultdict(int)
for c in range(nc):
    gt = gt_counts[c]["TP"] + gt_counts[c]["Miss"]
    pr = sum(pred_counts[c].values())
    row = [pred_counts[c][k] for k in cols]
    miss = gt_counts[c]["Miss"]
    print(f"{names[c]:<22}{gt:>6}{pr:>7}" + "".join(f"{v:>11}" for v in row)
          + f"{miss:>8}")
    for k, v in zip(cols, row):
        tot[k] += v
    tot["Miss"] += miss
    tot["GT"] += gt
    tot["preds"] += pr
print(f"{'all':<22}{tot['GT']:>6}{tot['preds']:>7}"
      + "".join(f"{tot[c]:>11}" for c in cols) + f"{tot['Miss']:>8}")


# ============================================================================
# FALSE POSITIVES AS A SHARE OF EACH CLASS'S PREDICTIONS
# ============================================================================
print("\n" + "=" * 92)
print("FALSE POSITIVES AS A SHARE OF PREDICTIONS")
print("=" * 92)
print(f"{'class':<22}{'preds':>8}{'FP total':>10}{'% FP':>8}"
      f"{'% Bkg':>8}{'% Loc':>8}{'% Cls':>8}")
for c in range(nc):
    pr = sum(pred_counts[c].values())
    if not pr:
        continue
    fp = pr - pred_counts[c]["TP"]
    print(f"{names[c]:<22}{pr:>8}{fp:>10}{100 * fp / pr:>8.1f}"
          f"{100 * pred_counts[c]['Bkg'] / pr:>8.1f}"
          f"{100 * pred_counts[c]['Loc'] / pr:>8.1f}"
          f"{100 * pred_counts[c]['Cls'] / pr:>8.1f}")


# ============================================================================
# CLASS CONFUSIONS
# ============================================================================
if confusions:
    print("\n" + "=" * 92)
    print("CLASSIFICATION ERRORS, true -> predicted")
    print("=" * 92)
    for (t, p), n in sorted(confusions.items(), key=lambda kv: -kv[1])[:15]:
        print(f"   {names[t]:<22} -> {names[p]:<22}{n:>6}")


# ============================================================================
# CANDIDATE FAILURE IMAGES
# ============================================================================
print("\n" + "=" * 92)
print("CANDIDATE FAILURE IMAGES")
print("=" * 92)
for kind in ("Bkg", "Loc", "Cls", "Miss"):
    ranked = sorted(((d.get(kind, 0), n) for n, d in per_image), reverse=True)[:5]
    print(f"\n most {kind}:")
    for n, name in ranked:
        if n:
            print(f"   {name:<40}{n}")

import csv
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image"] + cols + ["Miss"])
    for name, d in per_image:
        w.writerow([name] + [d.get(k, 0) for k in cols] + [d.get("Miss", 0)])
print(f"\nper-image breakdown saved to {OUT_CSV}")
