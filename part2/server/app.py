import os, time, io, uuid, cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="yolo detect api", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# gonna store loaded models here
MODELS = {}

class Det(BaseModel):
    cls_id: int
    cls_name: str
    conf: float
    box: List[float]   # x1 y1 x2 y2

class ImgResult(BaseModel):
    model: str
    backend: str
    w: int
    h: int
    ms: float
    count: int
    dets: List[Det]

class FrameOut(BaseModel):
    idx: int
    ts_ms: float
    ms: float
    count: int
    dets: List[Det]

class VidResult(BaseModel):
    model: str
    backend: str
    w: int
    h: int
    n_frames: int
    done_frames: int
    total_ms: float
    avg_ms: float
    avg_fps: float
    frames: List[FrameOut]

class ModelEntry(BaseModel):
    id: str
    label: str
    backend: str

# ---- load models on startup ----

@app.on_event("startup")
def startup():
    # always load the pytorch ones
    for tag, wt, bk in [("yolov8n-pt","yolov8n.pt","pytorch"), ("yolov8s-pt","yolov8s.pt","pytorch")]:
        try:
            MODELS[tag] = {"m": YOLO(wt), "label": f"YOLOv8{tag[6]}", "bk": bk}
            print(f"loaded {tag}")
        except Exception as e:
            print(f"skip {tag}: {e}")

    # try loading exported formats if they exist
    extras = [
        ("yolov8n-onnx", "yolov8n.onnx", "onnx"),
        ("yolov8s-onnx", "yolov8s.onnx", "onnx"),
        ("yolov8n-trt", "yolov8n.engine", "tensorrt"),
        ("yolov8s-trt", "yolov8s.engine", "tensorrt"),
    ]
    for tag, path, bk in extras:
        if os.path.isfile(path):
            try:
                MODELS[tag] = {"m": YOLO(path), "label": tag, "bk": bk}
                print(f"loaded {tag}")
            except Exception as e:
                print(f"skip {tag}: {e}")

    os.makedirs("_uploads", exist_ok=True)


def _detect(mentry, img, conf, iou, sz):
    """run detection, return list of Det + time in ms"""
    t0 = time.perf_counter()
    r = mentry["m"].predict(img, conf=conf, iou=iou, imgsz=sz, verbose=False)
    ms = (time.perf_counter() - t0) * 1000.0
    out = []
    for b in r[0].boxes:
        out.append(Det(
            cls_id=int(b.cls[0]),
            cls_name=r[0].names[int(b.cls[0])],
            conf=round(float(b.conf[0]), 4),
            box=[round(float(x), 1) for x in b.xyxy[0].tolist()],
        ))
    return out, ms

def _draw(img, dets):
    """draw boxes on the image, return new image"""
    cp = img.copy()
    for d in dets:
        x1,y1,x2,y2 = [int(v) for v in d.box]
        # quick color from class id
        np.random.seed(d.cls_id + 7)
        c = tuple(int(x) for x in np.random.randint(60, 240, 3))
        cv2.rectangle(cp, (x1,y1), (x2,y2), c, 2)
        txt = f"{d.cls_name} {d.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(cp, (x1, y1-th-6), (x1+tw+4, y1), c, -1)
        cv2.putText(cp, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return cp

def _get_model(mid):
    if mid not in MODELS:
        raise HTTPException(404, f"no such model '{mid}', available: {list(MODELS.keys())}")
    return MODELS[mid]

def _read_img(contents):
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "couldnt decode image")
    return img

# ---- routes ----

@app.get("/v1/health")
def health():
    return {"ok": True, "models": list(MODELS.keys())}

@app.get("/v1/models")
def list_models():
    return [ModelEntry(id=k, label=v["label"], backend=v["bk"]) for k,v in MODELS.items()]

@app.post("/v1/detect/image", response_model=ImgResult)
async def detect_img(
    file: UploadFile = File(...),
    model_id: str = Query("yolov8n-pt"),
    confidence: float = Query(0.25, ge=0.01, le=0.99),
    iou: float = Query(0.45, ge=0.1, le=0.95),
    imgsz: int = Query(640),
):
    me = _get_model(model_id)
    raw = await file.read()
    img = _read_img(raw)
    h, w = img.shape[:2]
    dets, ms = _detect(me, img, confidence, iou, imgsz)
    return ImgResult(model=model_id, backend=me["bk"], w=w, h=h, ms=round(ms,2), count=len(dets), dets=dets)

@app.post("/v1/detect/image/viz")
async def detect_img_viz(
    file: UploadFile = File(...),
    model_id: str = Query("yolov8n-pt"),
    confidence: float = Query(0.25),
    iou: float = Query(0.45),
    imgsz: int = Query(640),
):
    me = _get_model(model_id)
    raw = await file.read()
    img = _read_img(raw)
    dets, ms = _detect(me, img, confidence, iou, imgsz)
    drawn = _draw(img, dets)
    _, buf = cv2.imencode('.jpg', drawn, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return StreamingResponse(
        io.BytesIO(buf.tobytes()), media_type="image/jpeg",
        headers={"X-Ms": str(round(ms,1)), "X-Count": str(len(dets))},
    )

@app.post("/v1/detect/video", response_model=VidResult)
async def detect_vid(
    file: UploadFile = File(...),
    model_id: str = Query("yolov8n-pt"),
    confidence: float = Query(0.25),
    iou: float = Query(0.45),
    imgsz: int = Query(640),
    skip: int = Query(1, ge=1),
    maxf: int = Query(100, ge=1, le=500),
):
    me = _get_model(model_id)
    tmp = f"_uploads/{uuid.uuid4().hex}.mp4"
    try:
        data = await file.read()
        with open(tmp, "wb") as fh:
            fh.write(data)
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(400, "cant open video")

        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_out = []
        total_ms = 0
        fi = 0
        done = 0

        while done < maxf:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % skip == 0:
                dets, ms = _detect(me, frame, confidence, iou, imgsz)
                ts = fi / fps * 1000 if fps > 0 else 0
                frames_out.append(FrameOut(
                    idx=fi, ts_ms=round(ts,1), ms=round(ms,2),
                    count=len(dets), dets=dets,
                ))
                total_ms += ms
                done += 1
            fi += 1

        cap.release()
        avg = total_ms / max(done, 1)

        return VidResult(
            model=model_id, backend=me["bk"], w=vw, h=vh,
            n_frames=tot, done_frames=done,
            total_ms=round(total_ms,1), avg_ms=round(avg,1),
            avg_fps=round(1000/avg if avg > 0 else 0, 1),
            frames=frames_out,
        )
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
