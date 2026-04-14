"""
run coco-style mAP evaluation on our own annotated data
compares all available model backends (pt, onnx, trt)
"""

import os, json
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_one(model_path, ann_file, img_dir, sz=640):
    with open(ann_file) as f:
        gt = json.load(f)

    coco_gt = COCO(ann_file)
    catmap = {c["name"]: c["id"] for c in gt["categories"]}

    m = YOLO(model_path)
    preds = []
    pid = 1

    for im in gt["images"]:
        fpath = os.path.join(img_dir, im["file_name"])
        if not os.path.isfile(fpath):
            continue
        r = m.predict(fpath, imgsz=sz, conf=0.001, verbose=False)
        for b in r[0].boxes:
            cn = r[0].names[int(b.cls[0])]
            if cn not in catmap: continue
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            preds.append({
                "id": pid, "image_id": im["id"], "category_id": catmap[cn],
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": float(b.conf[0]),
                "area": (x2-x1)*(y2-y1),
            })
            pid += 1

    if not preds:
        print("  no predictions at all")
        return None

    dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return {
        "map50_95": round(float(ev.stats[0]), 4),
        "map50": round(float(ev.stats[1]), 4),
        "map75": round(float(ev.stats[2]), 4),
        "ar100": round(float(ev.stats[8]), 4),
    }


if __name__ == "__main__":
    ann = "my_annotations/labels.json"
    imgs = "my_annotations/imgs"

    if not os.path.isfile(ann):
        print("no annotations found. run: python tools/annotate.py --vid YOUR_VIDEO.mp4")
        exit(1)

    to_test = {"yolov8n_pt": "yolov8n.pt", "yolov8s_pt": "yolov8s.pt"}
    for tag, path in [("yolov8n_onnx","yolov8n.onnx"),("yolov8s_onnx","yolov8s.onnx"),
                      ("yolov8n_trt","yolov8n.engine"),("yolov8s_trt","yolov8s.engine")]:
        if os.path.isfile(path):
            to_test[tag] = path

    all_res = {}
    for tag, mp in to_test.items():
        print(f"\n--- {tag} ---")
        try:
            r = eval_one(mp, ann, imgs)
            all_res[tag] = r
        except Exception as e:
            print(f"  failed: {e}")
            all_res[tag] = {"err": str(e)}

    print("\n" + "="*70)
    print(f"{'model':<22} {'map50':>8} {'map50-95':>10} {'ar@100':>8}")
    print("-"*70)
    for tag, r in all_res.items():
        if r and "err" not in r:
            print(f"{tag:<22} {r['map50']:>8.4f} {r['map50_95']:>10.4f} {r['ar100']:>8.4f}")
        else:
            print(f"{tag:<22} failed")
    print("="*70)

    os.makedirs("results", exist_ok=True)
    with open("results/custom_eval.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print("\nsaved to results/custom_eval.json")
