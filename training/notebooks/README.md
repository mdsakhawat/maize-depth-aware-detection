# Training runs

One notebook per run; each is self-contained and clones the dataset itself. Table 8 of the
paper reports all seven.

| Notebook | Epochs | Seed | Initialization | mAP@0.5 |
|---|---|---|---|---|
| `120ep_seed0.ipynb` | 120 | 0 | pretrained | **0.885** — the run reported throughout the paper |
| `120ep_seed1.ipynb` | 120 | 1 | pretrained | |
| `120ep_seed2.ipynb` | 120 | 2 | pretrained | |
| `200ep_seed0.ipynb` | 200 | 0 | pretrained | |
| `200ep_seed1.ipynb` | 200 | 1 | pretrained | |
| `200ep_seed2.ipynb` | 200 | 2 | pretrained | |
| `200ep_scratch_seed0.ipynb` | 200 | 0 | random | 0.857 |

`120ep_seed0.ipynb` does three things in order, and is the origin of three items in the paper:

1. clones the dataset and writes the class YAML
2. **audits every split** for label–image consistency and counts instances per class
   → **Table 1**, and `../../results/class_instance_statistics.csv`
3. trains YOLO11n and evaluates on the test split → **Table 3**, **Figures 4, 5 and 7**

Its `best.pt` is the checkpoint every later notebook loads.

Summary across seeds, as reported in Table 8:

| Configuration | precision | recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 120 epochs, pretrained | 0.860 ± 0.009 | 0.808 ± 0.010 | 0.884 ± 0.003 | 0.695 ± 0.005 |
| 200 epochs, pretrained | 0.875 ± 0.006 | 0.835 ± 0.003 | 0.906 ± 0.002 | 0.734 ± 0.003 |
| 200 epochs, from scratch | 0.870 | 0.758 | 0.857 | 0.681 |

Mean ± standard deviation over three random seeds, except the from-scratch run, which was
carried out once.

`../train_yolo11n.py` runs any of these configurations from the command line.
