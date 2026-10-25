"""
FAILURE EXAMPLES FIGURE

Finds one specific instance of each failure mode and crops the image around
it, so that each panel shows the error itself rather than a whole frame full
of mixed outcomes.

  localization   the lesion is found but the box does not fit it (0.2 <= IoU < 0.45)
  background     a predicted box lying away from every annotated lesion
  missed         an annotated lesion with no detection near it

Ground truth is drawn in green, predictions in red. Run after
error_analysis.py, in the same session.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
WEIGHTS  = Path("/kaggle/working/reconstructed_best.pt")
TEST_IMG = Path("/kaggle/working/maize-disease-detection/images/test")
TEST_LBL = Path("/kaggle/working/maize-disease-detection/labels/test")
OUT_PNG  = Path("/kaggle/working/figure_failure_examples.png")

CONF    = 0.25
IMGSZ   = 640
ZOOM    = 2.4      # crop side, as a multiple of the longer box side
MIN_PIX = 416      # skip images smaller than this, they look poor at print size
PICK    = {"Loc": 0, "Bkg": 0, "Miss": 0}   # 0 = best candidate, 1 = next, ...
# ----------------------------------------------------------------------------

model = YOLO(str(WEIGHTS))
names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    bb = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (aa + bb - inter + 1e-9)


def load_gt(stem, W, H):
    cls, box = [], []
    f = TEST_LBL / (stem + ".txt")
    if f.exists():
        for line in f.read_text().split("\n"):
            p = line.split()
            if len(p) < 5:
                continue
            c, xc, yc, w, h = int(p[0]), *map(float, p[1:5])
            cls.append(c)
            box.append([(xc - w / 2) * W, (yc - h / 2) * H,
                        (xc + w / 2) * W, (yc + h / 2) * H])
    return np.array(cls, dtype=int), np.array(box, dtype=float).reshape(-1, 4)


def area_frac(b, W, H):
    return (b[2] - b[0]) * (b[3] - b[1]) / (W * H)


images = sorted(p for p in TEST_IMG.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print(f"scanning {len(images)} test images ...")

cand = {"Loc": [], "Bkg": [], "Miss": []}

for n, path in enumerate(images):
    r = model.predict(str(path), imgsz=IMGSZ, conf=CONF, device=0, verbose=False)[0]
    H, W = r.orig_shape
    if min(W, H) < MIN_PIX:
        continue
    g_cls, g_box = load_gt(path.stem, W, H)
    if len(g_cls) == 0:
        continue
    if r.boxes is None or len(r.boxes) == 0:
        p_cls, p_box = np.zeros(0, int), np.zeros((0, 4))
    else:
        p_cls = r.boxes.cls.cpu().numpy().astype(int)
        p_box = r.boxes.xyxy.cpu().numpy()

    ious = iou_matrix(p_box, g_box)

    for pi in range(len(p_cls)):
        row = ious[pi]
        same = np.where(g_cls == p_cls[pi])[0]
        best_same = row[same].max() if len(same) else 0.0
        best_any = row.max() if row.size else 0.0
        af = area_frac(p_box[pi], W, H)

        if 0.20 <= best_same < 0.45 and 0.01 <= af <= 0.35:
            gi = same[int(np.argmax(row[same]))]
            cand["Loc"].append((abs(best_same - 0.32), path, int(p_cls[pi]),
                                p_box[pi], g_box[gi]))
        if best_any < 0.05 and 0.005 <= af <= 0.25:
            cand["Bkg"].append((af, path, int(p_cls[pi]), p_box[pi], None))

    if len(p_box):
        gt_best = ious.max(axis=0)
    else:
        gt_best = np.zeros(len(g_cls))
    for gi in range(len(g_cls)):
        af = area_frac(g_box[gi], W, H)
        if gt_best[gi] < 0.10 and 0.01 <= af <= 0.35:
            cand["Miss"].append((-af, path, int(g_cls[gi]), None, g_box[gi]))

    if (n + 1) % 400 == 0:
        print(f"  {n + 1}/{len(images)}")

for k in cand:
    cand[k].sort(key=lambda t: t[0])
    print(f"\n{k}: {len(cand[k])} candidates")
    for s, p, c, *_ in cand[k][:5]:
        print(f"   {names[c]:<22} {p.name[:58]}")

PANELS = [("Loc", "Localization error"),
          ("Bkg", "Background false positive"),
          ("Miss", "Missed detection")]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))

for ax, (key, title) in zip(np.atleast_1d(axes), PANELS):
    if not cand[key]:
        ax.set_axis_off()
        continue
    _, path, cls_id, pbox, gbox = cand[key][min(PICK[key], len(cand[key]) - 1)]
    im = Image.open(path).convert("RGB")
    W, H = im.size

    # square crop centred on the selected boxes, large enough to contain them
    boxes = [b for b in (pbox, gbox) if b is not None]
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * ZOOM
    side = min(side, W, H)                       # never larger than the image
    side = max(side, (x2 - x1) * 1.15, (y2 - y1) * 1.15)
    cx = min(max(cx, side / 2), W - side / 2)    # keep the window inside the image
    cy = min(max(cy, side / 2), H - side / 2)
    cx1, cy1 = cx - side / 2, cy - side / 2
    cx2, cy2 = cx + side / 2, cy + side / 2

    ax.imshow(im.crop((cx1, cy1, cx2, cy2)))

    # only the boxes this panel is about, so nothing else distracts
    def draw(b, color, style):
        ax.add_patch(patches.Rectangle((b[0] - cx1, b[1] - cy1),
                                       b[2] - b[0], b[3] - b[1],
                                       linewidth=2.8, edgecolor=color,
                                       facecolor="none", linestyle=style))

    if gbox is not None:
        draw(gbox, "#2ca02c", "-")
    if pbox is not None:
        draw(pbox, "#d62728", "--")

    ax.set_title(f"{title}\n{names[cls_id]}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

handles = [
    patches.Patch(edgecolor="#2ca02c", facecolor="none", linewidth=2.6,
                  label="ground truth"),
    patches.Patch(edgecolor="#d62728", facecolor="none", linewidth=2.6,
                  linestyle="--", label="prediction"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.01), fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
print("\nsaved:", OUT_PNG)

from IPython.display import Image as IPImage, display
display(IPImage(str(OUT_PNG), width=1100))
