"""
pull frames from a video and auto-label them with yolov8 as a starting point
then you go in and fix the mistakes manually in the json

usage: python annotate.py --vid myvideo.mp4 --out my_annotations/
"""

import cv2, json, os, argparse
from datetime import datetime
from ultralytics import YOLO

def go(vid_path, out_dir, model_wt="yolov8s.pt", every=10, maxn=50, min_conf=0.3):
    os.makedirs(os.path.join(out_dir, "imgs"), exist_ok=True)

    m = YOLO(model_wt)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"cant open {vid_path}")
        return

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {w}x{h}, {fps:.1f}fps, {n_total} frames")

    coco = {
        "info": {"desc": os.path.basename(vid_path), "date": datetime.now().isoformat()},
        "images": [],
        "annotations": [],
        "categories": [],
    }

    cats = {}  # name -> id
    ann_id = 1
    img_id = 1
    fi = 0
    got = 0

    while got < maxn:
        ok, frame = cap.read()
        if not ok: break

        if fi % every == 0:
            fname = f"f{fi:05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, "imgs", fname), frame)

            preds = m.predict(frame, conf=min_conf, verbose=False)
            r = preds[0]

            coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})

            for b in r.boxes:
                cname = r.names[int(b.cls[0])]
                if cname not in cats:
                    cats[cname] = len(cats) + 1

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                bw, bh = x2-x1, y2-y1

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cats[cname],
                    "bbox": [round(x1,1), round(y1,1), round(bw,1), round(bh,1)],
                    "area": round(bw*bh, 1),
                    "iscrowd": 0,
                    "score": round(float(b.conf[0]), 3),
                    "review": True,  # flag to remind you to check this
                })
                ann_id += 1

            got += 1
            img_id += 1
            if got % 10 == 0:
                print(f"  {got}/{maxn}...")
        fi += 1

    cap.release()

    coco["categories"] = [{"id": v, "name": k} for k,v in sorted(cats.items(), key=lambda x:x[1])]

    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\ngot {got} frames, {ann_id-1} annotations")
    print(f"classes: {list(cats.keys())}")
    print(f"saved to {out_dir}/labels.json")
    print(f"\n** go through labels.json and fix any wrong boxes **")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--out", default="my_annotations")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--maxn", type=int, default=50)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    go(args.vid, args.out, every=args.every, maxn=args.maxn, min_conf=args.conf)
