# Results

The CSVs the tables were built from.

| File | Holds | Paper item |
|---|---|---|
| `dataset_statistics.csv` | images, label files and instances per split | the split totals in Section 3.1 |
| `class_instance_statistics.csv` | instances per class per split | **Table 1** |
| `best_training_epoch.csv` | validation metrics at the best epoch of the 120-epoch run | — (the *test*-split figures of Table 3 come from `model.val(split="test")`) |

## Still to add

`3d_localization.csv` — one row per detection with class, confidence, box, centre and
X<sub>rel</sub> / Y<sub>rel</sub> / Z<sub>rel</sub>. Export it from
`../pipeline/notebooks/02_relative_3d_localization_rerun.ipynb`, which is the run behind the
reported numbers; on Kaggle it is saved as `detections_3d_fixed.csv`. It should contain
**2,903 rows across 1,493 images**.

Everything in Tables 5 and 6 and Figures 9 and 10 is derived from that one file, so it is the
single artefact a reader needs in order to check the depth results.
