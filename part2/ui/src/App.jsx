import { useState, useRef, useEffect, useCallback } from "react"

// some colors for different classes
const COLORS = ["#e74c3c","#2ecc71","#3498db","#f1c40f","#9b59b6","#e67e22","#1abc9c","#e84393"]
const colorFor = id => COLORS[id % COLORS.length]

export default function App() {
  const [models, setModels] = useState([])
  const [pickedModel, setPickedModel] = useState("")
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isVideo, setIsVideo] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [err, setErr] = useState("")
  const [conf, setConf] = useState(0.3)
  const [connected, setConnected] = useState(false)
  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  // grab model list on load
  useEffect(() => {
    fetch("/v1/models")
      .then(r => r.json())
      .then(d => {
        setModels(d)
        if (d.length) setPickedModel(d[0].id)
        setConnected(true)
      })
      .catch(() => setConnected(false))
  }, [])

  function onFile(e) {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setResult(null)
    setErr("")
    if (f.type.startsWith("video")) {
      setIsVideo(true)
      setPreview(URL.createObjectURL(f))
    } else {
      setIsVideo(false)
      const rdr = new FileReader()
      rdr.onload = ev => setPreview(ev.target.result)
      rdr.readAsDataURL(f)
    }
  }

  // draw boxes on canvas overlay
  const paintBoxes = useCallback((dets, srcW, srcH) => {
    const cv = canvasRef.current
    const im = imgRef.current
    if (!cv || !im) return
    cv.width = im.naturalWidth
    cv.height = im.naturalHeight
    const ctx = cv.getContext("2d")
    ctx.drawImage(im, 0, 0)

    const rx = im.naturalWidth / srcW
    const ry = im.naturalHeight / srcH

    for (const d of dets) {
      const [x1,y1,x2,y2] = d.box
      const sx = x1*rx, sy = y1*ry, sw = (x2-x1)*rx, sh = (y2-y1)*ry
      const col = colorFor(d.cls_id)
      ctx.strokeStyle = col
      ctx.lineWidth = 2
      ctx.strokeRect(sx, sy, sw, sh)

      // label
      const lbl = `${d.cls_name} ${Math.round(d.conf*100)}%`
      ctx.font = "bold 13px monospace"
      const tw = ctx.measureText(lbl).width
      ctx.fillStyle = col
      ctx.globalAlpha = 0.8
      ctx.fillRect(sx, sy-18, tw+6, 18)
      ctx.globalAlpha = 1
      ctx.fillStyle = "#fff"
      ctx.fillText(lbl, sx+3, sy-4)
    }
  }, [])

  async function runDetect() {
    if (!file || busy) return
    setBusy(true)
    setErr("")
    setResult(null)
    try {
      const fd = new FormData()
      fd.append("file", file)
      const ep = isVideo ? "video" : "image"
      const t0 = performance.now()
      const resp = await fetch(`/v1/detect/${ep}?model_id=${pickedModel}&confidence=${conf}`, {
        method: "POST", body: fd
      })
      const rtt = Math.round(performance.now() - t0)
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}))
        throw new Error(e.detail || `http ${resp.status}`)
      }
      const data = await resp.json()
      data._rtt = rtt
      setResult(data)

      // draw if image
      if (!isVideo && data.dets) {
        setTimeout(() => paintBoxes(data.dets, data.w, data.h), 80)
      }
    } catch(ex) {
      setErr(ex.message)
    }
    setBusy(false)
  }

  // styles - just inline, keeping it simple
  const panel = { background:"#1a1a1a", borderRadius:8, padding:16, marginBottom:14 }
  const lbl = { fontSize:11, color:"#777", marginBottom:4, display:"block" }
  const sel = { width:"100%",padding:"8px",background:"#222",color:"#ddd",border:"1px solid #333",borderRadius:4,fontSize:12 }

  return (
    <div style={{maxWidth:1200, margin:"0 auto", padding:20}}>
      <h1 style={{fontSize:20, color:"#2ecc71", marginBottom:4}}>detection ui</h1>
      <p style={{fontSize:11, color: connected ? "#2ecc71" : "#e74c3c", marginBottom:20}}>
        {connected ? `server ok, ${models.length} models` : "backend offline - run: uvicorn app:app --port 8000"}
      </p>

      <div style={{display:"grid", gridTemplateColumns:"280px 1fr", gap:16}}>

        {/* left sidebar */}
        <div>
          <div style={panel}>
            <span style={lbl}>model</span>
            <select value={pickedModel} onChange={e=>setPickedModel(e.target.value)} style={sel}>
              {models.map(m => <option key={m.id} value={m.id}>{m.label} ({m.backend})</option>)}
            </select>

            <span style={{...lbl, marginTop:14}}>confidence: {Math.round(conf*100)}%</span>
            <input type="range" min={0.05} max={0.95} step={0.05} value={conf}
              onChange={e => setConf(+e.target.value)}
              style={{width:"100%", accentColor:"#2ecc71"}} />

            <span style={{...lbl, marginTop:14}}>file</span>
            <input type="file" accept="image/*,video/*" onChange={onFile}
              style={{fontSize:12, color:"#aaa", marginBottom:14, display:"block"}} />

            {file && <p style={{fontSize:10, color:"#555"}}>{file.name} ({(file.size/1024).toFixed(0)}kb)</p>}

            <button onClick={runDetect} disabled={!file || busy || !connected}
              style={{
                width:"100%", padding:10, marginTop:12,
                background: busy ? "#333" : "#2ecc71",
                color: busy ? "#666" : "#111",
                border:"none", borderRadius:6, fontWeight:"bold",
                cursor: busy ? "wait" : "pointer", fontSize:13
              }}>
              {busy ? "running..." : "detect"}
            </button>

            {err && <p style={{color:"#e74c3c", fontSize:11, marginTop:8}}>{err}</p>}
          </div>
        </div>

        {/* main area */}
        <div>
          <div style={{...panel, minHeight:380, display:"flex", alignItems:"center", justifyContent:"center", position:"relative"}}>
            {!preview && <span style={{color:"#444"}}>upload something to start</span>}

            {preview && !isVideo && (
              <div style={{position:"relative", maxWidth:"100%"}}>
                <img ref={imgRef} src={preview} alt=""
                  style={{maxWidth:"100%", maxHeight:500, display: result ? "none" : "block", borderRadius:4}} />
                <canvas ref={canvasRef}
                  style={{maxWidth:"100%", maxHeight:500, display: result ? "block" : "none", borderRadius:4}} />
              </div>
            )}

            {preview && isVideo && (
              <video src={preview} controls style={{maxWidth:"100%", maxHeight:500, borderRadius:4}} />
            )}

            {busy && (
              <div style={{position:"absolute",inset:0,background:"rgba(0,0,0,0.7)",display:"flex",alignItems:"center",justifyContent:"center",borderRadius:8}}>
                <span style={{color:"#2ecc71"}}>processing...</span>
              </div>
            )}
          </div>

          {/* results */}
          {result && (
            <div style={panel}>
              <h3 style={{fontSize:13, color:"#2ecc71", marginBottom:10}}>results</h3>
              <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(120px,1fr))", gap:8, marginBottom:16}}>
                {[
                  ["model", result.model],
                  ["backend", result.backend],
                  ["inference", isVideo ? `${result.avg_ms}ms` : `${result.ms}ms`],
                  ["round trip", `${result._rtt}ms`],
                  ["detections", isVideo ? `${result.done_frames} frames` : result.count],
                  ...(isVideo ? [["avg fps", result.avg_fps]] : []),
                ].map(([k,v], i) => (
                  <div key={i} style={{background:"#222", padding:"10px 12px", borderRadius:6}}>
                    <div style={{fontSize:9, color:"#555", textTransform:"uppercase"}}>{k}</div>
                    <div style={{fontSize:16, fontWeight:"bold"}}>{v}</div>
                  </div>
                ))}
              </div>

              {/* detection list for images */}
              {!isVideo && result.dets && result.dets.length > 0 && (
                <div style={{maxHeight:240, overflowY:"auto"}}>
                  {result.dets.map((d, i) => (
                    <div key={i} style={{
                      display:"flex", alignItems:"center", padding:"6px 10px",
                      background: i%2===0 ? "#222" : "transparent", fontSize:12, borderRadius:3,
                    }}>
                      <span style={{width:8,height:8,borderRadius:2,background:colorFor(d.cls_id),marginRight:8,flexShrink:0}} />
                      <span style={{flex:1}}>{d.cls_name}</span>
                      <span style={{color:"#2ecc71", marginRight:12}}>{Math.round(d.conf*100)}%</span>
                      <span style={{color:"#444", fontSize:10}}>[{d.box.map(v=>Math.round(v)).join(",")}]</span>
                    </div>
                  ))}
                </div>
              )}

              {/* frame list for video */}
              {isVideo && result.frames && (
                <div style={{maxHeight:240, overflowY:"auto"}}>
                  {result.frames.slice(0,80).map((fr, i) => (
                    <div key={i} style={{
                      display:"flex", padding:"5px 10px", gap:16,
                      background: i%2===0 ? "#222" : "transparent", fontSize:11,
                    }}>
                      <span style={{color:"#666", width:60}}>#{fr.idx}</span>
                      <span style={{color:"#2ecc71", width:70}}>{fr.ms}ms</span>
                      <span>{fr.count} obj</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
