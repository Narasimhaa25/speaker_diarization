import { fmt } from '../utils'

export default function IdPanel({ data }) {
  if (!data) return <div className="empty-state"><div className="empty-icon">🪪</div><p>No speakers identified yet</p></div>
  const { segments, speaker_colour, threshold } = data
  const bySpk = {}
  segments.forEach(s => {
    if (!bySpk[s.speaker_id] || s.score > bySpk[s.speaker_id].score) bySpk[s.speaker_id] = s
  })

  return (
    <div className="id-panel">
      {Object.entries(bySpk).sort((a,b) => a[0].localeCompare(b[0])).map(([spk, s]) => {
        const col        = speaker_colour[spk] || '#4f8ef7'
        const isStaff    = s.role === 'staff'
        const pct        = Math.round(s.score * 100)
        const barCol     = isStaff ? 'var(--green)' : 'var(--cyan)'
        const totalTime  = segments.filter(seg => seg.speaker_id === spk).reduce((a, seg) => a + seg.duration, 0)
        // Borderline: within 0.06 above threshold — possible false match
        const borderline = isStaff && s.score < (threshold + 0.06)

        return (
          <div key={spk} className="id-card" style={borderline ? { borderColor: 'rgba(251,191,36,0.4)' } : {}}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
              <div>
                <div className="spk-id" style={{ color: col }}>{spk}</div>
                <div className="spk-name">{isStaff ? s.name : 'Unknown Customer'}</div>
              </div>
              <div style={{ width: 36, height: 36, borderRadius: '50%', background: `${col}22`, border: `2px solid ${col}44`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem', flexShrink: 0 }}>
                {isStaff ? '👤' : '🛒'}
              </div>
            </div>

            <div className="sim-row">
              <div className="sim-label">Similarity</div>
              <div className="sim-bar-bg">
                <div className="sim-bar-fill" style={{ width: `${pct}%`, background: barCol }} />
              </div>
              <div className="sim-val" style={{ color: barCol }}>{s.score.toFixed(3)}</div>
            </div>
            <div className="sim-row">
              <div className="sim-label">Confidence</div>
              <div className="sim-bar-bg">
                <div className="sim-bar-fill" style={{ width: `${pct}%`, background: 'var(--muted)' }} />
              </div>
              <div className="sim-val" style={{ color: 'var(--muted)' }}>{pct}%</div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 10 }}>
              <span className={`role-badge ${isStaff ? 'role-staff' : 'role-customer'}`}>
                {isStaff ? '✓ Staff' : '◇ Customer'}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>{totalTime.toFixed(1)}s total</span>
            </div>

            {borderline && (
              <div style={{ marginTop: 8, fontSize: '0.7rem', color: 'var(--amber)', display: 'flex', alignItems: 'center', gap: 4 }}>
                ⚠ Borderline match — raise threshold or re-enroll
              </div>
            )}

            <div className="id-time">Best: {fmt(s.start)} – {fmt(s.end)}</div>
          </div>
        )
      })}
    </div>
  )
}
