import json, time, os
from ultralytics import YOLO

# just using yolov8 nano as our starting point
# nothing fancy, all defaults so we have something to compare against
mdl = YOLO("yolov8n.pt")

t0 = time.time()

res = mdl.train(
    data="coco128.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    project="out/baseline",
    name="run1",
    seed=42,
)

elapsed = time.time() - t0

# run val to grab the numbers
val = mdl.val(data="coco128.yaml", imgsz=640, batch=16, project="out/baseline", name="val1")

nums = {
    "map50": round(float(val.box.map50), 4),
    "map50_95": round(float(val.box.map), 4),
    "prec": round(float(val.box.mp), 4),
    "rec": round(float(val.box.mr), 4),
    "time_sec": round(elapsed, 1),
}

os.makedirs("results", exist_ok=True)
with open("results/base.json", "w") as f:
    json.dump(nums, f, indent=2)

print("\n--- baseline done ---")
for k, v in nums.items():
    print(f"  {k}: {v}")
