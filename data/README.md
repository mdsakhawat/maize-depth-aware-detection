# Data

No image data is stored in this repository. All three datasets are public.

## Training, validation and test

**Maize leaf disease dataset** — https://github.com/ycwsorry/maize-disease-detection-dataset

Released with Yang et al. (2026, *Frontiers in Plant Science* 17:1840425). An aggregation of
publicly available collections (PlantVillage, PlantDoc, maize disease datasets on Kaggle) with
manual annotation in YOLO format. The geographic origin and acquisition conditions of the
underlying images are not documented in those source collections.

The repository ships its own split definitions, which were used unchanged:

| Split | Images | Instances |
|---|---|---|
| Train | 10,609 | 20,386 |
| Validation | 3,004 | 6,033 |
| Test | 1,506 | 2,947 |
| **Total** | **15,119** | **29,366** |

Eight classes: blight, brown spot, corn rust, corn smut, downy mildew, gray leaf spot, healthy,
maize streak disease. Per-class counts are in `../results/class_instance_statistics.csv`.

### Audit run before training

Every split was checked programmatically for label–image consistency. The audit returned:

- label files = image files in every split
- zero empty, missing, orphaned or malformed annotation files across all 15,119 labels

so the counts above carry no duplication or misalignment. The audit is the first part of
`../pipeline/notebooks/01_dataset_audit.ipynb`, and its output is in
`../results/dataset_statistics.csv`.

## External validation

**An open-source multi-format maize leaf image dataset for disease detection and
classification** (Rabbany et al., 2026) — https://data.mendeley.com/datasets/sb4wwjs8bz/1

Collected in Bangladesh in 2026. Four of its classes overlap this label set (491 images). A
perceptual-hash (dHash, Hamming distance) check found **no overlap** with the training data. The
two datasets annotate at different granularity, so agreement was assessed at image level —
see `../analysis/external_validation.py`.

## Depth validation

**WE3DS** (Kitzler et al., 2023, *Sensors* 23:2713) — https://doi.org/10.3390/s23052713

An RGB-D dataset of field crops with stereo-measured distance maps. Used to check that the
relative depth produced by Depth Anything V2 orders surfaces correctly; 80 images, each divided
into an 8 × 8 grid of patches — see `../analysis/we3ds_validation.py`.
