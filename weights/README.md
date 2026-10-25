# Weights

`best.pt` — YOLO11n trained for 120 epochs at 640 × 640, batch 16, seed 0, from the pretrained
YOLO11n initialization. This is the checkpoint behind every detection, depth and timing number
in the paper.

Test-split performance (Table 3):

| | precision | recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| All classes | 0.852 | 0.819 | 0.885 | 0.691 |

Load it with:

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model.predict(source=image_bgr, conf=0.25)   # note: BGR, not RGB
```

The channel order matters — see the note in the top-level README.
