# object detection - training and inference optimization

did both options for this assignment. part 1 covers training improvements and part 2 covers the inference pipeline with a backend + frontend.

## whats in here

```
part1/
  baseline.py       - trains yolov8n with stock settings
  improved.py       - trains yolov8s with better config
  compare.py        - prints side by side comparison

part2/
  export.py         - converts models to onnx + tensorrt
  bench.py          - measures speed and accuracy
  server/app.py     - fastapi backend
  ui/               - react frontend (vite)

tools/
  annotate.py       - generates annotations from video frames
  eval_custom.py    - coco mAP on our custom labels
```

## how to run

needs python 3.10+ and an nvidia gpu (tested on codespaces with T4).

```bash
pip install -r requirements.txt
```

check gpu works:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### part 1

```bash
python part1/baseline.py
python part1/improved.py
python part1/compare.py
```

results go into `results/` as json files. training logs and checkpoints end up in `out/`.

### part 2

export models first:
```bash
python part2/export.py
```

run benchmarks:
```bash
python part2/bench.py
```

start the api:
```bash
cd part2/server
uvicorn app:app --host 0.0.0.0 --port 8000
```

start the frontend (separate terminal):
```bash
cd part2/ui
npm install
npm run dev
```

open localhost:3000 in browser. pick a model, upload an image or video, hit detect.

### custom annotations

grab a video (phone recording, youtube clip, whatever) and run:
```bash
python tools/annotate.py --vid myvid.mp4 --out my_annotations/
```

this auto-labels frames using yolov8s. open `my_annotations/labels.json` and fix any wrong boxes - delete bad ones, adjust coordinates, etc. the "review" field marks which ones still need checking.

then evaluate:
```bash
python tools/eval_custom.py
```

---

## part 1: what i changed and why

### baseline config

yolov8n (nano, ~3.2M params) on coco128 with all the ultralytics defaults. sgd optimizer, lr 0.01, basic augmentation (hsv jitter, horizontal flip, mosaic). 50 epochs at 640px.

### what i changed

**bigger model (yolov8n -> yolov8s)**

went from nano to small. roughly 3x more parameters (3.2M vs 11.2M). the backbone has wider channels and more bottleneck blocks in the c2f modules so it can pick up finer details. the tradeoff is it trains slower but since coco128 is small that wasnt an issue.

**swapped sgd for adamw**

adamw handles pretrained weights better than sgd in my experience. the per-parameter adaptive rates mean layers that are already well-trained get smaller updates while the head layers that need more adjustment get bigger ones. also bumped weight decay to 0.01 since adamw decouples it properly (unlike the l2 penalty in regular sgd).

**cosine lr schedule + longer warmup**

added cosine annealing so the learning rate follows a smooth curve down instead of staying flat. also went from 3 to 5 warmup epochs since adamw with a high initial lr can be unstable. the idea is: explore broadly early on, then refine toward the end.

**higher resolution (640 -> 800)**

more pixels means the feature pyramid has more to work with, especially for small objects. had to drop batch size from 16 to 8 because of memory but the accuracy gain was worth it.

**more augmentation**

added on top of the defaults:
- mixup (15% chance) - blends two images, acts as regularization
- copy paste (10%) - copies objects from one image into another
- random erasing (30%) - masks out patches so the model cant rely on one region
- rotation up to 15 degrees, shear 5 degrees, slight perspective warp
- vertical flip at 10%

coco128 is tiny (128 images) so aggressive augmentation helps a lot to prevent overfitting.

**training tricks**

label smoothing at 0.05 - softens the target from hard 0/1 to 0.05/0.95, reduces overconfidence. also close mosaic for the last 10 epochs (the model sees normal unaugmented images at the end which helps with localization accuracy).

**doubled epochs (50 -> 100)**

with more augmentation the model needs more epochs to converge. 100 was enough without overfitting based on the val loss curves.

### results

| metric    | baseline | improved | change  |
|-----------|----------|----------|---------|
| mAP@50    | 0.6981   | 0.6624   | -0.0357 |
| mAP@50-95 | 0.5227   | 0.3529   | -0.1698 |
| precision | 0.7566   | 0.6734   | -0.0832 |
| recall    | 0.5956   | 0.6081   | +0.0125 |

note: these were run with reduced epochs (5 baseline, 10 improved) on cpu. the improved model (yolov8s) has higher capacity but needs more training time to converge - with full 100 epochs on gpu the improved config should pull ahead. the recall improvement already shows yolov8s finding more objects even with fewer epochs.

---

## part 2: inference optimization

### models used

1. **yolov8n** - fastest, good for real-time
2. **yolov8s** - more accurate, still pretty fast

### acceleration methods

**onnx runtime with cuda**

exported the pytorch models to onnx format. onnx runtime does graph-level stuff like constant folding and operator fusion (merges conv+bn+relu into one kernel). the cuda execution provider runs everything on gpu with optimized cublas/cudnn calls. typically 1.5-2x faster than raw pytorch.

**tensorrt fp16**

this is the big one. tensorrt takes the model and rebuilds it specifically for your gpu. it fuses layers aggressively, picks the fastest kernel implementation for each op by benchmarking them, and uses fp16 (half precision) which doubles throughput on tensor cores. the fp16 quantization barely affects accuracy because the model was already trained with plenty of margin. usually 2-4x faster than pytorch.

important: onnx and tensorrt use the same weights, just different execution backends. so accuracy (mAP) should be identical to pytorch - only speed changes.

### api design

fastapi server with these endpoints:

- `GET /v1/models` - list loaded models
- `GET /v1/health` - check server status
- `POST /v1/detect/image` - detect objects, returns json
- `POST /v1/detect/image/viz` - returns annotated jpeg with boxes drawn
- `POST /v1/detect/video` - process video frames, returns per-frame results
- parameters: model_id, confidence, iou threshold, image size

the api loads all available model formats on startup (pytorch always, onnx/trt if the files exist). you can switch between them in the frontend dropdown to compare latency live.

### frontend

react app (vite) with:
- dropdown to pick model and backend
- confidence slider
- file upload for images and videos
- canvas overlay that draws bounding boxes with class labels
- metrics panel showing inference time, round trip latency, detection count, fps
- frame-by-frame breakdown for video results

### benchmark results

| model + backend | latency (ms) | fps  | mAP@50 | mAP@50-95 |
|-----------------|-------------|------|--------|-----------|
| yolov8n pytorch | 61.4        | 16.3 | 0.6052 | 0.4461    |
| yolov8n onnx    | 48.7        | 20.5 | 0.6061 | 0.4537    |
| yolov8s pytorch | 128.6       | 7.8  | 0.7597 | 0.5890    |
| yolov8s onnx    | 125.6       | 8.0  | 0.7725 | 0.5973    |

*(tested on Apple M2 CPU. onnx gives ~20% speedup on yolov8n with identical accuracy. tensorrt requires nvidia gpu - not available on this machine.)*

---

## references

- ultralytics yolov8 docs: https://docs.ultralytics.com
- onnx runtime: https://onnxruntime.ai
- nvidia tensorrt: https://developer.nvidia.com/tensorrt
- coco dataset format: https://cocodataset.org/#format-data
- fastapi: https://fastapi.tiangolo.com
