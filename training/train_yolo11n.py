"""Train YOLO11n on the eight-class maize leaf disease dataset.

A command-line form of the training cells in
`training/notebooks/120ep_seed0.ipynb`, so the configuration can be run
outside Kaggle. The architecture is stock YOLO11n: no redesigned head, no
substituted regression loss, no added attention. Table 2 of the manuscript
lists these settings.

Everything not named below is the Ultralytics 8.4.136 default, including the
optimizer (`auto`, which resolves to SGD at this dataset size), the
augmentation schedule, and `deterministic=True`.

The seven runs reported in Table 8:

    python training/train_yolo11n.py --epochs 120 --seed 0            # reported
    python training/train_yolo11n.py --epochs 120 --seed 1
    python training/train_yolo11n.py --epochs 120 --seed 2
    python training/train_yolo11n.py --epochs 200 --seed 0
    python training/train_yolo11n.py --epochs 200 --seed 1
    python training/train_yolo11n.py --epochs 200 --seed 2
    python training/train_yolo11n.py --epochs 200 --seed 0 --scratch

Results reported throughout the paper come from the first of these.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# the class order the dataset ships with, and the order the trained weights
# use. "grey_leaf_spot" is the dataset's own spelling; the manuscript writes
# the disease name as "gray leaf spot".
CLASSES = ["Blight", "brown_spot", "corn_rust", "corn_smut",
           "downy_mildew", "grey_leaf_spot", "healthy",
           "maize-streak-disease"]


def write_data_yaml(root: Path) -> Path:
    """Recreate kaggle_data.yaml exactly as the notebooks write it."""
    path = root / "kaggle_data.yaml"
    path.write_text(
        f"\npath: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n" + "".join(f"- {c}\n" for c in CLASSES)
    )
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path,
                   default=Path("maize-disease-detection"),
                   help="dataset root holding images/ and labels/; clone it "
                        "from github.com/ycwsorry/maize-disease-detection-dataset")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--scratch", action="store_true",
                   help="start from a random initialization instead of the "
                        "pretrained YOLO11n weights")
    p.add_argument("--project", type=Path, default=Path("runs"))
    args = p.parse_args()

    data = write_data_yaml(args.data_root)
    tag = (f"yolo11n_e{args.epochs}_s{args.seed}"
           f"{'_scratch' if args.scratch else '_pretrained'}")

    # YOLO("yolo11n.pt") loads the pretrained checkpoint;
    # YOLO("yolo11n.yaml") builds the same architecture from scratch.
    model = YOLO("yolo11n.yaml" if args.scratch else "yolo11n.pt")

    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=tag,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # test-split metrics: the numbers in Table 3
    best = YOLO(args.project / tag / "weights" / "best.pt")
    m = best.val(
        data=str(data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="test",
        project=str(args.project),
        name=f"{tag}_test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print(f"\n{tag} — test split")
    print(f"  precision     {m.box.mp:.3f}")
    print(f"  recall        {m.box.mr:.3f}")
    print(f"  mAP@0.5       {m.box.map50:.3f}")
    print(f"  mAP@0.5:0.95  {m.box.map:.3f}")


if __name__ == "__main__":
    main()
