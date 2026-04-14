import json, time, os
from ultralytics import YOLO

# switched to yolov8s here - bigger backbone, more capacity
# also bumping up augmentations and swapping optimizer
# see the readme for full reasoning on each change
mdl = YOLO("yolov8s.pt")

t0 = time.time()

res = mdl.train(
    data="coco128.yaml",
    epochs=100,             # doubled from baseline
    imgsz=800,              # was 640, helps with smaller objects
    batch=8,                # had to drop bc of the larger imgs

    # adamw works better for finetuning pretrained weights imo
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    weight_decay=0.01,

    # longer warmup so it doesnt blow up early
    warmup_epochs=5.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05,
    cos_lr=True,            # cosine decay

    # beefed up augmentations
    hsv_h=0.02,
    hsv_s=0.75,
    hsv_v=0.5,
    degrees=15.0,           # rotation
    translate=0.15,
    scale=0.6,
    shear=5.0,
    perspective=0.001,
    fliplr=0.5,
    flipud=0.1,
    mosaic=1.0,
    mixup=0.15,             # new
    copy_paste=0.1,         # new
    erasing=0.3,            # new

    # extra tricks
    label_smoothing=0.05,
    close_mosaic=10,        # turn off mosaic for last 10 epochs

    project="out/improved",
    name="run1",
    seed=42,
)

elapsed = time.time() - t0

val = mdl.val(data="coco128.yaml", imgsz=800, batch=8, project="out/improved", name="val1")

nums = {
    "map50": round(float(val.box.map50), 4),
    "map50_95": round(float(val.box.map), 4),
    "prec": round(float(val.box.mp), 4),
    "rec": round(float(val.box.mr), 4),
    "time_sec": round(elapsed, 1),
}

os.makedirs("results", exist_ok=True)
with open("results/improved.json", "w") as f:
    json.dump(nums, f, indent=2)

print("\n--- improved done ---")
for k, v in nums.items():
    print(f"  {k}: {v}")
