import os, time, json, sys
import numpy as np
import cv2
from ultralytics import YOLO

def bench_speed(model_path, img_dir, n_warm=10, n_test=50):
    """time inference on a bunch of images, return stats in ms"""
    m = YOLO(model_path)

    imgs = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    ])
    if not imgs:
        print(f"  no images in {img_dir}?")
        return None

    imgs = imgs[:n_test]

    # warmup
    for _ in range(n_warm):
        m.predict(imgs[0], imgsz=640, verbose=False)

    times = []
    det_count = 0
    for p in imgs:
        t0 = time.perf_counter()
        r = m.predict(p, imgsz=640, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
        det_count += len(r[0].boxes)

    times = np.array(times)
    return {
        "n": len(imgs),
        "mean_ms": round(float(times.mean()), 2),
        "med_ms": round(float(np.median(times)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(1000.0 / times.mean(), 1),
        "total_dets": det_count,
    }

def bench_map(model_path, data_yaml="coco128.yaml"):
    m = YOLO(model_path)
    try:
        v = m.val(data=data_yaml, imgsz=640, verbose=False)
        return {
            "map50": round(float(v.box.map50), 4),
            "map50_95": round(float(v.box.map), 4),
        }
    except Exception as e:
        print(f"  map eval failed: {e}")
        return None


if __name__ == "__main__":
    # make sure coco128 is downloaded
    print("checking dataset...")
    tmp = YOLO("yolov8n.pt")
    tmp.val(data="coco128.yaml", imgsz=640, verbose=False)
    del tmp

    # find where ultralytics put the images
    possible = [
        os.path.expanduser("~/datasets/coco128/images/train2017"),
        "datasets/coco128/images/train2017",
    ]
    img_dir = None
    for p in possible:
        if os.path.isdir(p):
            img_dir = p
            break

    if not img_dir:
        print("cant find coco128 images, using dummy")
        os.makedirs("_tmp_imgs", exist_ok=True)
        for i in range(5):
            dummy = np.random.randint(0, 255, (640,640,3), dtype=np.uint8)
            cv2.imwrite(f"_tmp_imgs/test_{i}.jpg", dummy)
        img_dir = "_tmp_imgs"

    # which models to test
    candidates = {"yolov8n_pt": "yolov8n.pt", "yolov8s_pt": "yolov8s.pt"}

    # check for exported ones
    for tag, ext in [("yolov8n_onnx","yolov8n.onnx"), ("yolov8s_onnx","yolov8s.onnx"),
                     ("yolov8n_trt","yolov8n.engine"), ("yolov8s_trt","yolov8s.engine")]:
        if os.path.isfile(ext):
            candidates[tag] = ext

    results = {}
    for tag, mpath in candidates.items():
        print(f"\n=== {tag} ({mpath}) ===")

        print("  speed...")
        spd = bench_speed(mpath, img_dir)

        print("  accuracy...")
        acc = bench_map(mpath)

        results[tag] = {"speed": spd, "accuracy": acc}

    # print table
    print("\n" + "="*85)
    print(f"{'model':<22} {'lat(ms)':>10} {'fps':>8} {'map50':>8} {'map50-95':>10}")
    print("-"*85)
    for tag, r in results.items():
        s = r["speed"]
        a = r["accuracy"]
        if s and a:
            print(f"{tag:<22} {s['mean_ms']:>10.1f} {s['fps']:>8.1f} {a['map50']:>8.4f} {a['map50_95']:>10.4f}")
        elif s:
            print(f"{tag:<22} {s['mean_ms']:>10.1f} {s['fps']:>8.1f} {'n/a':>8} {'n/a':>10}")
    print("="*85)

    os.makedirs("results", exist_ok=True)
    with open("results/benchmarks.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nsaved to results/benchmarks.json")
