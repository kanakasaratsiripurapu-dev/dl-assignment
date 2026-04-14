import os, time, json
from ultralytics import YOLO

# export both models to onnx and tensorrt
# tensorrt needs an nvidia gpu obviously, will skip if not available

out = {}

for tag in ["yolov8n", "yolov8s"]:
    print(f"\n>>> exporting {tag}")
    m = YOLO(f"{tag}.pt")
    out[tag] = {}

    # onnx first
    try:
        t = time.time()
        p = m.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=False, opset=17)
        out[tag]["onnx"] = {"path": str(p), "took": round(time.time()-t, 1)}
        print(f"  onnx ok -> {p}")
    except Exception as e:
        out[tag]["onnx"] = {"err": str(e)}
        print(f"  onnx failed: {e}")

    # tensorrt (fp16)
    try:
        t = time.time()
        p = m.export(format="engine", imgsz=640, half=True, device=0, workspace=4)
        out[tag]["trt"] = {"path": str(p), "took": round(time.time()-t, 1)}
        print(f"  trt ok -> {p}")
    except Exception as e:
        out[tag]["trt"] = {"err": str(e)}
        print(f"  trt failed (need gpu+tensorrt): {e}")

os.makedirs("results", exist_ok=True)
with open("results/exports.json", "w") as f:
    json.dump(out, f, indent=2)
print("\ndone, manifest in results/exports.json")
