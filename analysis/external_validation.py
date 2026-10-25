"""
External validation of the trained YOLO11n detector on an independent maize
leaf dataset (Mendeley Data, DOI 10.17632/sb4wwjs8bz.1, CC BY 4.0).

Stages:
  1. Inventory   - class folders, CSV rows, label names, image counts
  2. Convert     - absolute-pixel CSV boxes -> YOLO labels, with EXIF size check
  3. Overlap     - dHash against the training/test split, and inside the set
  4. Box-level   - model.val() -> mAP@0.5  (secondary result)
  5. Image-level - top-1 class agreement   (primary result)

Set the four paths below, then run.
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import shutil
import sys

from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# PATHS - edit these four
# ----------------------------------------------------------------------------
EXT_ROOT   = Path("/kaggle/input/maize-external-bd")          # folder holding the class subfolders
TRAIN_ROOT = Path("/kaggle/working/maize-disease-detection")  # the dataset used in the paper
WEIGHTS    = Path("/kaggle/input/yolo11n-maize/best.pt")      # trained detector
OUT        = Path("/kaggle/working/ext_val")                  # where the converted set is written

IMGSZ      = 640
CONF_VAL   = 0.001   # Ultralytics val() default, same setting as Table 1
CONF_PRED  = 0.25    # same setting as Tables 3-4

# ----------------------------------------------------------------------------
# CLASS MAPPING - normalised label name -> model class name
# ----------------------------------------------------------------------------
CLASS_MAP = {
    "cornleafblight":     "Blight",
    "grayleafspot":       "grey_leaf_spot",
    "greyleafspot":       "grey_leaf_spot",
    "healthyleaf":        "healthy",
    "healthy":            "healthy",
    "healthyleaves":      "healthy",
    "maizestreakdisease": "maize-streak-disease",
    "maizestreak":        "maize-streak-disease",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def banner(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def dhash(path, size=8):
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(im.getdata())
    bits = 0
    for row in range(size):
        b = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (px[b + col] < px[b + col + 1])
    return bits


# ============================================================================
# STAGE 1 - INVENTORY
# ============================================================================
banner("STAGE 1  INVENTORY")

if not EXT_ROOT.exists():
    sys.exit(f"EXT_ROOT not found: {EXT_ROOT}")

class_dirs = [d for d in sorted(EXT_ROOT.rglob("*")) if d.is_dir()
              and any(f.suffix.lower() == ".csv" for f in d.iterdir() if f.is_file())]
if not class_dirs:
    sys.exit(f"No folder containing a CSV found under {EXT_ROOT}")

records = []
labels_seen = Counter()

for d in class_dirs:
    imgs = {f.name: f for f in d.iterdir() if f.suffix.lower() in IMG_EXT}
    rows = []
    for c in (f for f in d.iterdir() if f.suffix.lower() == ".csv"):
        with open(c, newline="", encoding="utf-8-sig", errors="replace") as fh:
            rows.extend(list(csv.DictReader(fh)))

    by_img, bad = defaultdict(list), 0
    for r in rows:
        name = (r.get("image_name") or "").strip()
        lab = (r.get("label_name") or "").strip()
        labels_seen[lab] += 1
        if name not in imgs:
            bad += 1
            continue
        try:
            by_img[name].append((lab,
                                 float(r["bbox_x"]), float(r["bbox_y"]),
                                 float(r["bbox_width"]), float(r["bbox_height"]),
                                 float(r["image_width"]), float(r["image_height"])))
        except (KeyError, ValueError, TypeError):
            bad += 1

    print(f"{d.name:<26} files={len(imgs):<5} csv_rows={len(rows):<5} "
          f"labelled={len(by_img):<5} unusable={bad}")
    for name, boxes in by_img.items():
        records.append((d.name, imgs[name], boxes))

print("\nlabel_name values in the CSVs:")
for k, v in labels_seen.most_common():
    print(f"   {k!r:<28} rows={v:<5} -> {CLASS_MAP.get(norm(k), '-- NOT MAPPED --')}")

usable = [r for r in records if CLASS_MAP.get(norm(r[2][0][0]))]
print(f"\nlabelled images {len(records)}   usable after mapping {len(usable)}")
if not usable:
    sys.exit("Nothing mapped. Fix CLASS_MAP against the label_name values above.")


# ============================================================================
# STAGE 2 - CONVERT
# ============================================================================
banner("STAGE 2  CONVERT")

from ultralytics import YOLO

model = YOLO(str(WEIGHTS))
names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
name_to_idx = {v: k for k, v in names.items()}
print("model classes:", names)

for t in set(CLASS_MAP.values()):
    if t not in name_to_idx:
        sys.exit(f"Class {t!r} not in model. Model has: {list(name_to_idx)}")

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True)
(OUT / "labels").mkdir(parents=True)

truth = {}            # stem -> model class index
written = size_mismatch = rotated = skipped = 0
per_class = Counter()

for folder, src, boxes in usable:
    # the CSV width/height must agree with the image as Ultralytics will load it
    try:
        with Image.open(src) as im:
            iw_file, ih_file = ImageOps.exif_transpose(im).size
    except Exception:
        skipped += 1
        continue

    lines, cls_here = [], None
    for lab, bx, by, bw, bh, iw, ih in boxes:
        target = CLASS_MAP.get(norm(lab))
        if target is None:
            continue
        if (int(iw), int(ih)) != (iw_file, ih_file):
            if (int(ih), int(iw)) == (iw_file, ih_file):
                rotated += 1
            else:
                size_mismatch += 1
            continue
        if iw <= 0 or ih <= 0 or bw <= 0 or bh <= 0:
            skipped += 1
            continue
        xc, yc, w, h = (bx + bw / 2) / iw, (by + bh / 2) / ih, bw / iw, bh / ih
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= xc <= 1 and 0 <= yc <= 1):
            skipped += 1
            continue
        lines.append(f"{name_to_idx[target]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        per_class[target] += 1
        cls_here = target

    if not lines:
        continue
    stem = f"{norm(folder)}_{src.stem}"
    shutil.copy2(src, OUT / "images" / (stem + src.suffix.lower()))
    (OUT / "labels" / (stem + ".txt")).write_text("\n".join(lines) + "\n")
    truth[stem] = name_to_idx[cls_here]
    written += 1

print(f"images written {written}   boxes skipped {skipped}")
print(f"CSV size disagrees with file : {size_mismatch}   (width/height swapped: {rotated})")
if size_mismatch or rotated:
    print("  WARNING: those rows were dropped. If this number is large the CSV "
          "sizes refer to a different copy of the images.")
for k, v in sorted(per_class.items()):
    print(f"   {k:<24} {v} boxes")


# ============================================================================
# STAGE 3 - OVERLAP
# ============================================================================
banner("STAGE 3  OVERLAP (dHash)")

ext_imgs = sorted((OUT / "images").iterdir())
ext_hash = {p.stem: dhash(p) for p in ext_imgs}

if TRAIN_ROOT.exists():
    tr = [p for p in TRAIN_ROOT.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"hashing {len(tr)} images of the paper's dataset ...")
    train_hashes = {h for h in (dhash(p) for p in tr) if h is not None}
    print(f"distinct hashes: {len(train_hashes)}")
    exact = near = 0
    for h in ext_hash.values():
        if h is None:
            continue
        if h in train_hashes:
            exact += 1
            near += 1
        elif any(bin(h ^ t).count("1") <= 2 for t in train_hashes):
            near += 1
    n = len(ext_imgs)
    print(f"external images {n}")
    print(f"exact match in training dataset : {exact} ({100*exact/max(n,1):.1f}%)")
    print(f"within Hamming distance 2       : {near} ({100*near/max(n,1):.1f}%)")
else:
    print(f"TRAIN_ROOT not found ({TRAIN_ROOT}) - skipped")

# near-duplicate clusters inside the external set, greedy at Hamming <= 2
reps, cluster_of = [], {}
for stem, h in ext_hash.items():
    if h is None:
        continue
    for r_stem, r_h in reps:
        if bin(h ^ r_h).count("1") <= 2:
            cluster_of[stem] = r_stem
            break
    else:
        reps.append((stem, h))
        cluster_of[stem] = stem
unique_stems = {s for s in cluster_of if cluster_of[s] == s}
print(f"near-duplicate clusters inside the external set : "
      f"{len(unique_stems)} distinct of {len(ext_hash)} images")


# ============================================================================
# STAGE 4 - BOX-LEVEL mAP
# ============================================================================
banner("STAGE 4  BOX-LEVEL mAP  (secondary)")

yaml_path = OUT / "data.yaml"
yaml_path.write_text(f"path: {OUT}\ntrain: images\nval: images\nnames:\n"
                     + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items())))

m = model.val(data=str(yaml_path), imgsz=IMGSZ, conf=CONF_VAL, split="val",
              plots=True, save_json=False, project=str(OUT), name="val", verbose=False)

print(f"\n{'class':<26}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
for i, c in enumerate(m.box.ap_class_index):
    print(f"{names[int(c)]:<26}{m.box.p[i]:>8.3f}{m.box.r[i]:>8.3f}"
          f"{m.box.ap50[i]:>9.3f}{m.box.ap[i]:>10.3f}")
print(f"{'all':<26}{m.box.mp:>8.3f}{m.box.mr:>8.3f}{m.box.map50:>9.3f}{m.box.map:>10.3f}")
print("speed (ms/image):", m.speed)

# how different are the two annotation conventions
print("\nmedian box area as % of image")
ext_area = defaultdict(list)
for f in (OUT / "labels").iterdir():
    for line in f.read_text().split("\n"):
        p = line.split()
        if len(p) >= 5:
            ext_area[int(p[0])].append(100 * float(p[3]) * float(p[4]))
import statistics
for c in sorted(ext_area):
    print(f"   {names[c]:<24} external {statistics.median(ext_area[c]):>6.1f}%"
          f"   (n={len(ext_area[c])})")


# ============================================================================
# STAGE 5 - IMAGE-LEVEL AGREEMENT
# ============================================================================
banner("STAGE 5  IMAGE-LEVEL AGREEMENT  (primary)")

idx_of_interest = sorted({name_to_idx[v] for v in set(CLASS_MAP.values())})
conf_mat = Counter()
no_detection = Counter()
pred_of = {}

for p in ext_imgs:
    r = model.predict(str(p), imgsz=IMGSZ, conf=CONF_PRED, verbose=False)[0]
    gt = truth[p.stem]
    if r.boxes is None or len(r.boxes) == 0:
        pred_of[p.stem] = None
        no_detection[gt] += 1
        continue
    best = int(r.boxes.conf.argmax())
    pred = int(r.boxes.cls[best])
    pred_of[p.stem] = pred
    conf_mat[(gt, pred)] += 1


def report(stems, title):
    tot = cor = nod = 0
    cm = Counter()
    for s in stems:
        gt = truth[s]
        pr = pred_of.get(s, None)
        tot += 1
        if pr is None:
            nod += 1
        else:
            cm[(gt, pr)] += 1
            cor += (pr == gt)
    print(f"\n{title}")
    print(f"  images {tot}   top-1 correct {cor} ({100*cor/max(tot,1):.1f}%)   "
          f"no detection {nod} ({100*nod/max(tot,1):.1f}%)")
    print(f"  {'true \\ predicted':<24}" + "".join(f"{names[c][:10]:>12}" for c in idx_of_interest)
          + f"{'other':>12}{'none':>7}")
    for g in idx_of_interest:
        row = [cm[(g, p)] for p in idx_of_interest]
        other = sum(v for (gg, pp), v in cm.items() if gg == g and pp not in idx_of_interest)
        none_ = sum(1 for s in stems if truth[s] == g and pred_of.get(s) is None)
        print(f"  {names[g]:<24}" + "".join(f"{v:>12}" for v in row)
              + f"{other:>12}{none_:>7}")


report(list(truth), "ALL IMAGES")
report([s for s in truth if s in unique_stems], "NEAR-DUPLICATES REMOVED (dHash, Hamming <= 2)")

print("\ndone")
