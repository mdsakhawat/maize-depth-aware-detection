# Pipeline notebooks

These take the trained detector (`weights/best.pt`, from
`../../training/notebooks/120ep_seed0.ipynb`) and add the depth branch.

| Notebook | What it does |
|---|---|
| `01_relative_3d_localization.ipynb` | First full run of the pipeline: YOLO11n detection, Depth Anything V2 depth, ROI-median fusion, pinhole back-projection |
| `02_relative_3d_localization_rerun.ipynb` | **The run behind the paper.** The same pipeline re-run after the colour-channel correction described in the top-level README. Every depth, 3D and error-analysis number in the manuscript comes from this notebook, together with Figures 10 and 11 |

Both write `3d_localization.csv` — one row per detection with class, confidence, box,
centre, and X<sub>rel</sub> / Y<sub>rel</sub> / Z<sub>rel</sub>. The copy in `../../results/`
is the output of `02`.

## The method in brief

For a detection with box (x₁, y₁, x₂, y₂) in an image of size W × H:

1. Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`) returns a raw relative depth
   map, normalized per image to [0, 1]:
   `Dnorm(u,v) = (D(u,v) − Dmin) / (Dmax − Dmin)`
2. The depth of the region is the **median** over the box, not a single-pixel sample:
   `Zrel = median{ Dnorm(u,v) : (u,v) ∈ ROI }`
3. The box centre is back-projected through a pinhole model with `cx = W/2`, `cy = H/2`
   and `fx = fy = W`:
   `Xrel = (uc − cx)·Zrel / fx`,  `Yrel = (vc − cy)·Zrel / fy`

Normalization is applied **per image**, so a value locates a detection within its own frame
rather than at a physical distance. Depths are comparable within an image, not across images.
