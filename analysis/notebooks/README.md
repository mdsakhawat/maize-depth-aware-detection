# Analysis notebooks

The Kaggle notebooks behind the post-hoc analyses. Each has a script equivalent one level up,
which is the version to run outside Kaggle. Every notebook rebuilds `best.pt` from the uploaded
weight directory and clones the dataset itself, so each one is self-contained.

| Notebook | Script | Produces |
|---|---|---|
| `confusion_matrix_recall_check.ipynb` | `../error_analysis.py`, `../failure_examples.py` | **Figure 6**, **Table 4**, **Figure 8** |
| `external_validation.ipynb` | `../external_validation.py` | **Section 5.5** |
| `we3ds_depth_validation.ipynb` | `../we3ds_validation.py` | **Section 4.3**, depth validation |
| `kruskal_wallis_depth_analysis.ipynb` | `../depth_stats.py` | **Tables 5 and 6**, **Figure 9** |
| `inference_time.ipynb` | `../timing_benchmark.py` | **Table 9** timings |

## What each one establishes

**`confusion_matrix_recall_check.ipynb`** does two things. First it compares the diagonal of the
normalized confusion matrix against the per-class recall of Table 3, and shows that the two are
not the same quantity: the matrix is built at a confidence of 0.25 and a matching IoU of 0.45,
while recall is taken at the maximum-F1 confidence at IoU 0.50. It then reproduces the matrix on
both the validation and the test split. Second, it runs the error analysis that splits every
prediction into exactly one outcome, giving Table 4:

| | ground truth | predictions | correct | missed | background | localization | wrong class | duplicate |
|---|---|---|---|---|---|---|---|---|
| all classes | 2947 | 2903 | 2282 | 665 | 131 | 352 | 6 | 132 |

**`external_validation.ipynb`** converts the Mendeley dataset's CSV boxes to YOLO labels,
checks for overlap with the training data by perceptual hash (**0 of 491 images** matched), and
reports image-level top-1 agreement (**34.8%** at a confidence of 0.25, against a four-class
chance level of 25%). Box-level mAP@0.5 is 0.005 because the two datasets annotate at very
different granularity — median box area is 3–31% of the image externally.

**`kruskal_wallis_depth_analysis.ipynb`** reads `3d_localization.csv` and produces the
class-wise depth summary of Table 5, the Kruskal–Wallis test and the Dunn post-hoc with Holm
correction:

| | |
|---|---|
| detections / images | 2,903 over 1,493 |
| Kruskal–Wallis | H = 310.009, df = 7, p = 4.4 × 10⁻⁶³ |
| effect size | ε² = 0.105 (moderate) |
| significant pairs | 18 of 28 after Holm correction |

Blight carries the highest median relative depth (0.743) and the healthy class the lowest
(0.489); each appears in seven of the eighteen significant pairs, and theirs is the only pair
whose interquartile ranges do not overlap.

**`inference_time.ipynb`** times the two branches separately on the Tesla T4 at batch size 1
and 640 × 640 input. Detection is the sum of the Ultralytics preprocess, inference and
postprocess figures; the depth branch is timed over 100 images with the first ten discarded as
warm-up. It also confirms that the `best` and `last` checkpoints of the 120-epoch run give
identical test metrics:

| | per image |
|---|---|
| detection | 11.6 ms (1.18 preprocess + 9.46 inference + 1.01 postprocess) |
| depth branch | 43.6 ms (n = 100, SD 0.9) |
| full pipeline | 55.2 ms, about 18 frames per second |

**`we3ds_depth_validation.ipynb`** compares the normalized output of Depth Anything V2 against
stereo-measured distance on 80 WE3DS images, 64 patches each. Per-image Spearman ρ is negative
in **79 of the 80 images**, median **−0.57**, IQR **−0.67 to −0.43** — the normalized output
increases as surfaces approach the camera, so it carries consistent ordering information about
scene depth.
