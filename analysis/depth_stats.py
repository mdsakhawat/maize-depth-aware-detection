"""
STEP 9.7b - CLASS-WISE DEPTH STATISTICS AND SIGNIFICANCE TESTING

Produces every number reported in Table 3, Table 4 and the Class-wise
Relative Depth Analysis subsection:

  1. class-wise Z statistics            -> Table 3
  2. overall X, Y, Z statistics         -> Table 4
  3. Kruskal-Wallis H and epsilon-squared
  4. Dunn post-hoc with Holm correction
  5. interquartile ranges and their overlap

Run it after STEP 9.6 in the same session, or point CSV_PATH at the saved
detection file.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
CSV_PATH = "/kaggle/working/detections_3d_fixed.csv"
# ----------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH).dropna(subset=["Z_relative"])

print("=" * 78)
print("INPUT")
print("=" * 78)
print("detections            :", len(df))
print("images with detections:", df["image_name"].nunique())
print("confidence threshold  : 0.25 (set in STEP 9.6)")


# class names as they are printed in the manuscript tables
DISPLAY = {
    "Blight":               "Blight",
    "brown_spot":           "Brown spot",
    "corn_rust":            "Corn rust",
    "corn_smut":            "Corn smut",
    "downy_mildew":         "Downy mildew",
    "grey_leaf_spot":       "Gray leaf spot",
    "healthy":              "Healthy",
    "maize-streak-disease": "Maize streak disease",
}


# ============================================================================
# 1. TABLE 3 - CLASS-WISE RELATIVE DEPTH  (manuscript layout)
# ============================================================================
print("\n" + "=" * 78)
print("TABLE 3   Class-wise relative depth (Z_rel) statistics")
print("=" * 78)

t3 = (df.groupby(["class_id", "class_name"])["Z_relative"]
        .agg(n="count", mean="mean", median="median",
             sd="std", min="min", max="max")
        .reset_index()
        .sort_values("class_id"))

hdr = (f"{'Class':<24}{'Detections':>12}{'Mean':>9}{'Median':>9}"
       f"{'Std.':>9}{'Min':>9}{'Max':>9}")
print(hdr)
print("-" * len(hdr))
for _, r in t3.iterrows():
    print(f"{DISPLAY.get(r['class_name'], r['class_name']):<24}"
          f"{int(r['n']):>12}{r['mean']:>9.3f}{r['median']:>9.3f}"
          f"{r['sd']:>9.3f}{r['min']:>9.3f}{r['max']:>9.3f}")


# ============================================================================
# 2. TABLE 4 - OVERALL RELATIVE 3D LOCALIZATION  (manuscript layout)
# ============================================================================
print("\n" + "=" * 78)
print("TABLE 4   Overall relative 3D localization summary")
print("=" * 78)

rows = [
    ("Total detections",          f"{len(df)}"),
    ("Images with detections",    f"{df['image_name'].nunique()}"),
    ("Number of disease classes", f"{df['class_name'].nunique()}"),
    ("Mean X_rel",                f"{df['X_relative'].mean():.4f}"),
    ("Median X_rel",              f"{df['X_relative'].median():.4f}"),
    ("Mean Y_rel",                f"{df['Y_relative'].mean():.4f}"),
    ("Median Y_rel",              f"{df['Y_relative'].median():.4f}"),
    ("Mean Z_rel",                f"{df['Z_relative'].mean():.3f}"),
    ("Median Z_rel",              f"{df['Z_relative'].median():.3f}"),
    ("Std. Z_rel",                f"{df['Z_relative'].std():.3f}"),
]
print(f"{'Metric':<28}{'Value':>12}")
print("-" * 40)
for label, value in rows:
    print(f"{label:<28}{value:>12}")


# ============================================================================
# 3. KRUSKAL-WALLIS
# ============================================================================
print("\n" + "=" * 78)
print("KRUSKAL-WALLIS")
print("=" * 78)

names = sorted(df["class_name"].unique())
groups = [df.loc[df.class_name == c, "Z_relative"].values for c in names]

H, p = stats.kruskal(*groups)
k, N = len(groups), len(df)
eps2 = (H - k + 1) / (N - k)

print(f"H = {H:.3f}   df = {k - 1}   p = {p:.3e}")
print(f"epsilon-squared = {eps2:.4f}   "
      f"({'small' if eps2 < 0.06 else 'moderate' if eps2 < 0.14 else 'large'})")
print(f"variance in ranked depth explained by class: {100 * eps2:.1f}%")


# ============================================================================
# 4. DUNN POST-HOC WITH HOLM CORRECTION
# ============================================================================
print("\n" + "=" * 78)
print("DUNN POST-HOC, HOLM-CORRECTED")
print("=" * 78)

ranks = stats.rankdata(df["Z_relative"].values)
mean_rank = pd.Series(ranks, index=df["class_name"].values).groupby(level=0).mean()
n_per = df.groupby("class_name").size()

# tie correction for the Dunn standard error
_, counts = np.unique(df["Z_relative"].values, return_counts=True)
ties = (counts ** 3 - counts).sum()
sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))

pairs, praw = [], []
for a, b in itertools.combinations(names, 2):
    zstat = (mean_rank[a] - mean_rank[b]) / np.sqrt(
        sigma2 * (1 / n_per[a] + 1 / n_per[b]))
    pairs.append((a, b, zstat))
    praw.append(2 * stats.norm.sf(abs(zstat)))

# Holm step-down
order = np.argsort(praw)
m = len(praw)
holm = np.empty(m)
running = 0.0
for i, idx in enumerate(order):
    running = max(running, min(1.0, (m - i) * praw[idx]))
    holm[idx] = running

sig = [(pairs[i][0], pairs[i][1], pairs[i][2], holm[i])
       for i in range(m) if holm[i] < 0.05]

print(f"{len(sig)} significant of {m} pairs at alpha = 0.05\n")
for a, b, zs, ph in sorted(sig, key=lambda t: t[3]):
    hi, lo = (a, b) if mean_rank[a] > mean_rank[b] else (b, a)
    print(f"   {hi:<22} > {lo:<22} z = {abs(zs):6.2f}   p_holm = {ph:.2e}")

appears = {c: sum(1 for a, b, _, _ in sig if c in (a, b)) for c in names}
print("\nnumber of significant pairs each class appears in:")
for c in sorted(appears, key=lambda c: -appears[c]):
    print(f"   {c:<24}{appears[c]}")


# ============================================================================
# 5. INTERQUARTILE RANGES AND OVERLAP
# ============================================================================
print("\n" + "=" * 78)
print("INTERQUARTILE RANGES")
print("=" * 78)

q = df.groupby("class_name")["Z_relative"].quantile([.25, .5, .75]).unstack()
q.columns = ["Q1", "median", "Q3"]
q["IQR"] = q["Q3"] - q["Q1"]
q = q.sort_values("median", ascending=False)

print(f"{'class':<24}{'Q1':>8}{'median':>9}{'Q3':>8}{'IQR':>8}")
for c, r in q.iterrows():
    print(f"{c:<24}{r['Q1']:>8.3f}{r['median']:>9.3f}{r['Q3']:>8.3f}{r['IQR']:>8.3f}")

non_overlap = [(a, b) for a, b in itertools.combinations(names, 2)
               if q.loc[a, "Q1"] > q.loc[b, "Q3"] or q.loc[b, "Q1"] > q.loc[a, "Q3"]]
print(f"\nclass pairs whose interquartile ranges do not overlap: "
      f"{len(non_overlap)} of {m}")
for a, b in non_overlap:
    print(f"   {a} vs {b}")

print("\n" + "=" * 78)
print("DONE")
print("=" * 78)
