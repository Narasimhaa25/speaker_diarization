import { useState, useRef } from 'react'

const PROMPTS = [
  "Hello, welcome to the store. How can I assist you today?",
  "Sure, let me check that for you. What size are you looking for?",
  "We have that item in stock. I'll get one from the back for you.",
]

const INIT_SLOTS = () => [
  { status: 'Ready to record', blob: null, seconds: 0, state: 'idle' },
  { status: 'Ready to record', blob: null, seconds: 0, state: 'idle' },
  { status: 'Ready to record', blob: null, seconds: 0, state: 'idle' },
]

function StepProgress({ step, saveResult }) {
  const steps = ['Staff Details', 'Record Voice', 'Save Enrollment']
  const fillPct = step === 1 ? 0 : step === 2 ? 50 : 100

  return (
    <>
      <div className="enroll-progress-track">
        <div className="enroll-progress-line" />
        <div className="enroll-progress-fill" style={{ width: `${fillPct}%` }} />
        {steps.map((_, i) => {
          const n = i + 1
          const isDone = n < step || (n === 3 && saveResult)
          const isActive = n === step
          return (
            <div
              key={n}
              className={`enroll-step-node${isActive ? ' active' : isDone ? ' done' : ''}`}
              style={{ marginLeft: i === 0 ? 0 : 'auto', marginRight: i === steps.length - 1 ? 0 : 'auto' }}
            >
              {isDone ? '✓' : n}
            </div>
          )
        })}
      </div>
      <div className="enroll-step-labels">
        {steps.map((label, i) => {
          const n = i + 1
          const isDone = n < step || (n === 3 && saveResult)
          const isActive = n === step
          return (
            <div key={n} className={`enroll-step-label${isActive ? ' active' : isDone ? ' done' : ''}`}>
              {label}
            </div>
          )
        })}
      </div>
    </>
  )
}

export default function EnrollStaff({ setPage }) {
  const [step, setStep] = useState(1)
  const [name, setName] = useState('')
  const [staffId, setStaffId] = useState('')
  const [role, setRole] = useState('associate')
  const [slots, setSlots] = useState(INIT_SLOTS())
  const [activeSlot, setActiveSlot] = useState(-1)
  const [saving, setSaving] = useState(false)
  const [saveResult, setSaveResult] = useState(null)
  const [saveError, setSaveError] = useState('')

  const mrRef     = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef  = useRef(null)

  const updateSlot = (i, patch) =>
    setSlots(prev => prev.map((s, idx) => idx === i ? { ...s, ...patch } : s))

  async function startRecord(i) {
    if (mrRef.current?.state === 'recording') { stopRecord(activeSlot); return }
    updateSlot(i, { status: 'Requesting microphone…', state: 'recording' })
    setActiveSlot(i)
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    } catch (err) {
      updateSlot(i, { status: 'Mic denied: ' + err.message, state: 'idle' }); return
    }
    streamRef.current = stream
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : ''
    const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : {})
    chunksRef.current = []
    mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mime || 'audio/webm' })
      setSlots(prev => prev.map((s, idx) => {
        if (idx !== i) return s
        const m = Math.floor(s.seconds / 60), sec = String(s.seconds % 60).padStart(2, '0')
        return { ...s, blob, state: 'done', status: `✓ ${m}:${sec} · ${(blob.size / 1024).toFixed(0)} KB` }
      }))
      setActiveSlot(-1)
    }
    mr.start(250); mrRef.current = mr
    let secs = 0
    timerRef.current = setInterval(() => {
      secs++
      const m = Math.floor(secs / 60), sec = String(secs % 60).padStart(2, '0')
      updateSlot(i, { seconds: secs, status: `⏺ Recording ${m}:${sec}` })
    }, 1000)
  }

  function stopRecord(i) {
    if (mrRef.current?.state === 'recording') mrRef.current.stop()
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null
    clearInterval(timerRef.current)
  }

  function goStep2() {
    if (!name.trim()) { alert('Please enter the staff member\'s name.'); return }
    setStep(2)
  }

  function goStep3() {
    const ready = slots.filter(s => s.blob !== null).length
    if (ready === 0) { alert('Please record at least one voice sample.'); return }
    setStep(3); setSaveResult(null); setSaveError('')
  }

  async function save() {
    const recordings = slots.filter(s => s.blob !== null)
    if (!name.trim() || recordings.length === 0) return
    setSaving(true); setSaveResult(null); setSaveError('')
    const fd = new FormData()
    fd.append('staff_name', name.trim())
    fd.append('staff_role', role)
    recordings.forEach((s, i) => fd.append(`audio_${i}`, s.blob, `rec_${i}.webm`))
    try {
      const resp = await fetch('/enroll', { method: 'POST', body: fd })
      const data = await resp.json()
      if (data.error) setSaveError(data.error)
      else setSaveResult(data)
    } catch (err) { setSaveError('Request failed: ' + err.message) }
    setSaving(false)
  }

  function reset() {
    setStep(1); setName(''); setStaffId(''); setRole('associate')
    setSlots(INIT_SLOTS()); setActiveSlot(-1); setSaveResult(null); setSaveError('')
  }

  const readyCount = slots.filter(s => s.blob !== null).length

  return (
    <div className="main">
      <div className="card">
        <div className="card-header" style={{ marginBottom: 6 }}>
          <div className="card-title-row">
            <div className="card-icon blue">🎙</div>
            <h2>Staff Voice Enrollment</h2>
          </div>
        </div>
        <p style={{ color: 'var(--muted)', fontSize: '0.82rem', marginBottom: 24 }}>
          Record 3 clear voice samples (10–30s each). Embeddings are averaged and stored encrypted — raw audio is never saved.
        </p>

        <StepProgress step={step} saveResult={saveResult} />

        {/* Step 1 */}
        {step === 1 && (
          <div style={{ animation: 'fadeIn 0.25s ease' }}>
            <div className="enroll-form">
              <div className="form-group">
                <label>Staff ID <span style={{ color: 'var(--muted)', fontSize: '0.72rem' }}>(auto-generated if blank)</span></label>
                <input value={staffId} onChange={e => setStaffId(e.target.value)} placeholder="e.g. staff_001" />
              </div>
              <div className="form-group">
                <label>Full Name<span className="form-required">*</span></label>
                <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Rushika Karampuri" />
              </div>
              <div className="form-group">
                <label>Role</label>
                <select value={role} onChange={e => setRole(e.target.value)}>
                  {['associate','supervisor','manager','cashier','security'].map(r => (
                    <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                  ))}
                </select>
              </div>
            </div>
            <div style={{ marginTop: 22 }}>
              <button className="btn" onClick={goStep2}>Next: Record Voice →</button>
            </div>
          </div>
        )}

        {/* Step 2 */}
        {step === 2 && (
          <div style={{ animation: 'fadeIn 0.25s ease' }}>
            <div className="guidance-grid">
              <div className="guidance-box good">
                <div className="guidance-title">✓ GOOD ENROLLMENT</div>
                <div className="guidance-list">Clear speech · Quiet room · 10–30 seconds · Natural pace · 10–30cm from mic</div>
              </div>
              <div className="guidance-box bad">
                <div className="guidance-title">✗ AVOID</div>
                <div className="guidance-list">Background music · Fan noise · Whispering · Short clips · Moving mic</div>
              </div>
            </div>

            <div className="prompt-box">
              💬 {PROMPTS[Math.max(0, activeSlot) % PROMPTS.length]}
            </div>

            <div className="rec-slots">
              {slots.map((s, i) => (
                <div key={i} className={`rec-slot${s.state === 'done' ? ' ready' : s.state === 'recording' ? ' recording-active' : ''}`}>
                  <div className="slot-num">{s.state === 'done' ? '✓' : i + 1}</div>
                  <div className="slot-status" style={{ color: s.state === 'done' ? 'var(--green)' : s.state === 'recording' ? 'var(--red)' : 'var(--text-dim)' }}>
                    {s.status}
                  </div>
                  <button
                    className={`btn btn-sm${s.state === 'recording' ? ' btn-danger' : s.state === 'done' ? ' btn-ghost' : ''}`}
                    disabled={activeSlot !== -1 && activeSlot !== i}
                    onClick={() => s.state === 'recording' ? stopRecord(i) : startRecord(i)}
                  >
                    {s.state === 'recording' ? '⏹ Stop' : s.state === 'done' ? '↺ Re-record' : '⏺ Record'}
                  </button>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 22, display: 'flex', gap: 10, alignItems: 'center' }}>
              <button className="btn btn-ghost" onClick={() => { stopRecord(activeSlot); setStep(1) }}>← Back</button>
              <button className="btn" disabled={readyCount === 0} onClick={goStep3}>
                Next: Save ({readyCount}/3) →
              </button>
              {readyCount > 0 && (
                <span style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>
                  {readyCount === 3 ? '✓ All 3 recorded' : `${3 - readyCount} more recommended`}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Step 3 */}
        {step === 3 && (
          <div style={{ animation: 'fadeIn 0.25s ease' }}>
            {!saveResult && (
              <div className="info-box">
                Ready to enroll <strong>{name}</strong> ({role}) with <strong>{readyCount}</strong> voice recording{readyCount !== 1 ? 's' : ''}.
                {readyCount < 3 && ' Adding more recordings improves accuracy.'}
              </div>
            )}

            {saving && (
              <div className="progress" style={{ marginTop: 16 }}>
                <div className="progress-bar"><div className="progress-fill" style={{ width: '60%' }} /></div>
                <div className="progress-text"><span className="spinner" />Generating voice embeddings…</div>
              </div>
            )}

            {saveError && <div className="error-box">⚠ {saveError}</div>}

            {saveResult && (
              <div className="success-box" style={{ marginTop: 16 }}>
                <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: 6 }}>
                  ✓ Successfully enrolled {saveResult.name}
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-dim)' }}>
                  Role: {saveResult.role} &nbsp;·&nbsp;
                  {saveResult.n_samples} sample{saveResult.n_samples !== 1 ? 's' : ''} &nbsp;·&nbsp;
                  ID: <code style={{ fontFamily: 'monospace', color: 'var(--cyan)' }}>{saveResult.staff_id?.substring(0, 12)}…</code>
                </div>
              </div>
            )}

            <div style={{ marginTop: 22, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {!saveResult && (
                <>
                  <button className="btn btn-ghost" disabled={saving} onClick={() => setStep(2)}>← Back</button>
                  <button className="btn btn-success" disabled={saving} onClick={save}>
                    {saving ? <><span className="spinner" />Enrolling…</> : '💾 Save Enrollment'}
                  </button>
                </>
              )}
              {saveResult && (
                <>
                  <button className="btn btn-ghost" onClick={reset}>➕ Enroll Another</button>
                  <button className="btn" onClick={() => setPage('staffdb')}>👥 View Staff DB</button>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h2 style={{ marginBottom: 12 }}>Re-enrollment Policy</h2>
        <p style={{ color: 'var(--text-dim)', fontSize: '0.84rem', lineHeight: 1.7 }}>
          Voices change over time due to illness, age, and seasonal variation. Periodic re-enrollment keeps
          identification accuracy high. <strong style={{ color: 'var(--text)' }}>Recommended:</strong> Re-enroll every{' '}
          <strong style={{ color: 'var(--blue)' }}>6–12 months</strong>. If accuracy drops noticeably, trigger
          immediate re-enrollment from the Staff DB tab.
        </p>
      </div>
    </div>
  )
}
