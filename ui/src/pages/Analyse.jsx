import { useState, useRef, useEffect } from 'react'
import useMicRecorder from '../hooks/useMicRecorder'
import WaveformCanvas from '../components/WaveformCanvas'
import AnalysisResults from '../components/AnalysisResults'

const STEPS = [
  [10,'Loading audio…'], [25,'Estimating speaker count…'],
  [40,'Running diarization (pyannote)…'], [70,'Computing ECAPA-TDNN embeddings…'],
  [85,'Matching staff database…'], [95,'Building results…'],
]

function useProgressTicker(running) {
  const [prog, setProg] = useState({ pct: 0, text: '' })
  const timerRef = useRef(null)
  useEffect(() => {
    if (running) {
      let si = 0; setProg({ pct: 0, text: '' })
      timerRef.current = setInterval(() => {
        if (si < STEPS.length) { setProg({ pct: STEPS[si][0], text: STEPS[si][1] }); si++ }
      }, 1800)
    } else {
      clearInterval(timerRef.current)
      setProg(p => p.pct > 0 ? { pct: 100, text: 'Done!' } : p)
    }
    return () => clearInterval(timerRef.current)
  }, [running])
  return prog
}

export default function Analyse({ setLastResult }) {
  const [threshold, setThreshold] = useState(0.90)
  const [maxDur, setMaxDur] = useState(120)
  const [micResult, setMicResult]   = useState(null)
  const [fileResult, setFileResult] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileError, setFileError]   = useState('')
  const [micError, setMicError]     = useState('')
  const [fileRunning, setFileRunning] = useState(false)
  const [micRunning, setMicRunning]   = useState(false)
  const [amiFiles, setAmiFiles]     = useState([])
  const [activeAmi, setActiveAmi]   = useState(null)
  const fileInputRef = useRef(null)
  const dropRef      = useRef(null)

  const mic      = useMicRecorder()
  const fileProg = useProgressTicker(fileRunning)
  const micProg  = useProgressTicker(micRunning)

  useEffect(() => {
    fetch('/ami_files').then(r => r.json()).then(d => setAmiFiles(d.files || [])).catch(() => {})
  }, [])

  const fmtSecs = s => { const m = Math.floor(s/60), sec = String(s%60).padStart(2,'0'); return `${m}:${sec}` }

  async function runFileUpload() {
    if (!selectedFile) return
    const fd = new FormData()
    fd.append('audio', selectedFile)
    fd.append('threshold', threshold)
    fd.append('max_duration', maxDur)
    setFileRunning(true); setFileError(''); setFileResult(null)
    const { ok, data, error } = await apiFetch('/analyse', { method: 'POST', body: fd })
    setFileRunning(false)
    if (!ok) setFileError(error)
    else { setFileResult(data); setLastResult(data) }
  }

  async function runSample(path) {
    setActiveAmi(path); setFileRunning(true); setFileError(''); setFileResult(null)
    const { ok, data, error } = await apiFetch('/analyse_sample', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, threshold, max_duration: maxDur }),
    })
    setFileRunning(false); setActiveAmi(null)
    if (!ok) setFileError(error)
    else { setFileResult(data); setLastResult(data) }
  }

  async function submitMic() {
    if (!mic.blob) return
    setMicRunning(true); setMicError(''); setMicResult(null)
    const fd = new FormData()
    fd.append('audio', mic.blob, 'recording.webm')
    fd.append('threshold', threshold)
    fd.append('max_duration', maxDur)
    const { ok, data, error } = await apiFetch('/analyse_realtime', { method: 'POST', body: fd })
    setMicRunning(false)
    if (!ok) setMicError(error)
    else { setMicResult(data); setLastResult(data) }
  }

  const meetings = {}
  amiFiles.forEach(f => { if (!meetings[f.meeting]) meetings[f.meeting] = []; meetings[f.meeting].push(f) })

  return (
    <div className="main">

      {/* ── Mic ── */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-row">
            <div className="card-icon red">🎤</div>
            <h2>Real-Time Voice Input</h2>
          </div>
          {mic.state === 'done' && mic.blob && (
            <span className="tag">
              {(mic.blob.size / 1024).toFixed(0)} KB · {fmtSecs(mic.seconds)}
            </span>
          )}
        </div>

        <div className={`mic-zone${mic.state === 'recording' ? ' recording' : mic.state === 'done' ? ' done' : ''}`}>
          <button
            className={`mic-btn ${mic.state === 'idle' || mic.state === 'done' ? 'idle' : mic.state === 'recording' ? 'active' : 'processing'}`}
            onClick={() => mic.state === 'recording' ? mic.stop() : mic.start()}
            title={mic.state === 'recording' ? 'Stop recording' : 'Start recording'}
          >
            {mic.state === 'recording' ? '⏹' : mic.state === 'done' ? '✓' : '🎤'}
          </button>

          <div>
            <div className={`mic-status${mic.state === 'recording' ? ' live' : mic.state === 'done' && mic.blob ? ' ok' : mic.error ? ' error' : ''}`}>
              {mic.error
                ? '✗ ' + mic.error
                : mic.state === 'idle'
                  ? 'Click the microphone to begin recording'
                  : mic.state === 'recording'
                    ? <><span className="live-dot" />Recording — click ⏹ to stop</>
                    : mic.blob
                      ? `✔ Recording ready (${fmtSecs(mic.seconds)}) — click Analyse or re-record`
                      : 'Finalising…'}
            </div>
            {mic.state === 'recording' && <div className="rec-timer">{fmtSecs(mic.seconds)}</div>}
          </div>

          <WaveformCanvas analyserRef={mic.analyserRef} active={mic.state === 'recording'} />
        </div>

        {mic.state === 'done' && (
          <div className="controls" style={{ marginTop: 14 }}>
            <button className="btn btn-success" onClick={submitMic} disabled={micRunning}>
              {micRunning ? <><span className="spinner" />Analysing…</> : '▶ Analyse Recording'}
            </button>
            <button className="btn btn-ghost" onClick={mic.clear}>↺ Discard</button>
          </div>
        )}

        {micRunning && (
          <div className="progress">
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${micProg.pct}%` }} /></div>
            <div className="progress-text"><span className="spinner" />{micProg.text}</div>
          </div>
        )}
        {micError && <div className="error-box">⚠ {micError}</div>}
        {micResult && <div style={{ marginTop: 20 }}><AnalysisResults data={micResult} badge="🎙 REAL-TIME RESULT" /></div>}
      </div>

      {/* ── Settings ── */}
      <div className="card">
        <h2 style={{ marginBottom: 16 }}>Analysis Settings</h2>
        <div className="controls">
          <div className="control-group">
            <label>
              Staff similarity threshold&nbsp;
              <span className="threshold-val">{threshold.toFixed(2)}</span>
            </label>
            <input type="range" min="0.60" max="0.95" step="0.01" value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))} />
            <span className="control-hint">≥ threshold → Staff &nbsp;·&nbsp; below → Customer</span>
          </div>
          <div className="control-group">
            <label>
              Max audio duration&nbsp;
              <span className="threshold-val">{maxDur}s</span>
            </label>
            <input type="range" min="30" max="825" step="30" value={maxDur}
              onChange={e => setMaxDur(parseInt(e.target.value))} />
            <span className="control-hint">Longer audio takes more processing time</span>
          </div>
        </div>
      </div>

      {/* ── File upload ── */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-row">
            <div className="card-icon blue">📂</div>
            <h2>Upload Audio File</h2>
          </div>
          <span style={{ fontSize: '0.73rem', color: 'var(--muted)' }}>WAV · MP3 · MP4 · MPEG · M4A · FLAC · OGG · AAC · OPUS · AIFF</span>
        </div>

        <div
          ref={dropRef}
          className="upload-zone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); dropRef.current?.classList.add('drag') }}
          onDragLeave={() => dropRef.current?.classList.remove('drag')}
          onDrop={e => {
            e.preventDefault(); dropRef.current?.classList.remove('drag')
            if (e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0])
          }}
        >
          <div className="icon">📂</div>
          <p>Drag &amp; drop an audio file, or click to browse</p>
          {selectedFile
            ? <div className="filename">📎 {selectedFile.name} ({(selectedFile.size / 1e6).toFixed(1)} MB)</div>
            : <p style={{ marginTop: 6, fontSize: '0.76rem', color: 'var(--text-faint)' }}>Supports WAV, MP3, MP4, MPEG, M4A, FLAC, OGG, AAC, OPUS, AIFF, WMA — up to 200 MB</p>
          }
        </div>
        <input ref={fileInputRef} type="file" accept=".wav,.mp3,.mp4,.mpeg,.mpg,.mp2,.m4a,.m4v,.flac,.ogg,.webm,.aac,.wma,.opus,.aiff,.aif,audio/*,video/mp4,video/mpeg" style={{ display: 'none' }}
          onChange={e => e.target.files[0] && setSelectedFile(e.target.files[0])} />

        <div className="controls" style={{ marginTop: 14 }}>
          <button className="btn" disabled={!selectedFile || fileRunning} onClick={runFileUpload}>
            {fileRunning ? <><span className="spinner" />Analysing…</> : '▶ Analyse File'}
          </button>
          {selectedFile && !fileRunning && (
            <button className="btn btn-ghost btn-sm" onClick={() => setSelectedFile(null)}>✕ Clear</button>
          )}
        </div>

        {fileRunning && (
          <div className="progress">
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${fileProg.pct}%` }} /></div>
            <div className="progress-text"><span className="spinner" />{fileProg.text}</div>
          </div>
        )}
        {fileError && <div className="error-box">⚠ {fileError}</div>}
      </div>

      {/* ── AMI samples ── */}
      {Object.keys(meetings).length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <div className="card-icon green">🎵</div>
              <h2>Sample Files — AMI Corpus</h2>
            </div>
            <span className="tag">Pre-loaded · Click to analyse</span>
          </div>
          {Object.entries(meetings).map(([meeting, files]) => (
            <div key={meeting} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--blue)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
                {meeting}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {files.map(f => (
                  <button
                    key={f.path}
                    className="ami-btn"
                    onClick={() => runSample(f.path)}
                    disabled={fileRunning}
                    style={{ opacity: activeAmi === f.path ? 0.6 : 1 }}
                  >
                    <div style={{ fontWeight: 700, color: f.kind.includes('Mix') ? 'var(--green)' : 'var(--cyan)' }}>
                      {f.kind.includes('Mix') ? '🎵' : '🎧'} {f.kind}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: 3 }}>{f.size_mb} MB</div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {fileResult && <AnalysisResults data={fileResult} />}
    </div>
  )
}
