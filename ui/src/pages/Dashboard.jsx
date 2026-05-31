import { useEffect, useState } from 'react'
import { fmt } from '../utils'

function DashCard({ title, icon, value, valueColor, sub, accentColor }) {
  return (
    <div className="dash-card">
      {accentColor && <div className="dash-card-accent" style={{ background: accentColor }} />}
      <div className="dc-title">
        {icon && <span>{icon}</span>}
        {title}
      </div>
      <div className="dc-val" style={{ color: valueColor || 'var(--text)' }}>
        {value}
      </div>
      {sub && <div className="dc-sub">{sub}</div>}
    </div>
  )
}

export default function Dashboard({ setPage, lastResult }) {
  const [staff, setStaff] = useState([])

  useEffect(() => {
    fetch('/staff')
      .then(r => r.json())
      .then(d => setStaff(d.staff || []))
      .catch(() => {})
  }, [])

  const activeStaff = staff.filter(s => s.active)
  const d = lastResult

  return (
    <div className="main">
      {/* Metrics */}
      <div className="dash-grid">
        <DashCard
          title="Microphone" icon="🎙"
          value={<><span className="status-dot grey" />Idle</>}
          valueColor="var(--muted)"
          sub="Click Analyse tab to start"
          accentColor="var(--blue)"
        />
        <DashCard
          title="Recording" icon="⏺"
          value={<><span className="status-dot grey" />Standby</>}
          valueColor="var(--muted)"
          sub="No active session"
        />
        <DashCard
          title="Last Session" icon="📁"
          value={d ? d.filename.substring(0, 20) + (d.filename.length > 20 ? '…' : '') : '—'}
          valueColor={d ? 'var(--text)' : 'var(--muted)'}
          sub={d ? `${d.staff_count} staff · ${d.customer_count} customer turns · ${fmt(d.duration)}` : 'No analysis yet'}
          accentColor={d ? 'var(--green)' : undefined}
        />
        <DashCard
          title="Detected Speakers" icon="👤"
          value={d ? d.est_speakers : '—'}
          valueColor="var(--cyan)"
          sub={d ? `${d.segments.length} segments · RTF ${d.rtf}x` : 'From last session'}
          accentColor="var(--cyan)"
        />
        <DashCard
          title="Enrolled Staff" icon="🔐"
          value={activeStaff.length}
          valueColor="var(--green)"
          sub={`${staff.length} total · ${staff.length - activeStaff.length} inactive`}
          accentColor="var(--green)"
        />
      </div>

      {/* Last result panels */}
      {d ? (
        <>
          <div className="card">
            <div className="card-header">
              <div className="card-title-row">
                <div className="card-icon green">🗂</div>
                <h2>Live Speaker Timeline</h2>
              </div>
              <span className="tag">
                {d.diar_mode || 'pyannote'}
              </span>
            </div>
            <div className="live-seg-list">
              {d.segments.map((s, i) => {
                const col = d.speaker_colour[s.speaker_id] || '#4f8ef7'
                return (
                  <div key={i} className={`live-seg-row ${s.role === 'staff' ? 'staff-row' : 'customer-row'}`}>
                    <div className="live-seg-time">[{fmt(s.start)} → {fmt(s.end)}]</div>
                    <div className="live-seg-spk" style={{ color: col }}>{s.speaker_id}</div>
                    <div className="live-seg-id">
                      {s.name
                        ? <strong style={{ color: 'var(--text)' }}>{s.name}</strong>
                        : <span style={{ color: 'var(--text-faint)' }}>Unknown</span>}
                    </div>
                    <span className={`role-${s.role}`}>{s.role.toUpperCase()}</span>
                    <div className="live-seg-score">sim {s.score.toFixed(3)}</div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title-row">
                <div className="card-icon blue">🪪</div>
                <h2>Identification Panel</h2>
              </div>
            </div>
            <IdCards data={d} />
          </div>
        </>
      ) : (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="empty-state">
            <div className="empty-icon">🎙</div>
            <p>No analysis yet. Go to <strong>Analyse</strong> to process audio and see results here.</p>
            <button className="btn" style={{ marginTop: 16 }} onClick={() => setPage('analyse')}>
              Start Analysis
            </button>
          </div>
        </div>
      )}

      {/* Quick actions + staff */}
      <div className="two-col">
        <div className="card">
          <h2 style={{ marginBottom: 14 }}>Quick Actions</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <button className="btn" onClick={() => setPage('analyse')}>🎙 Start Real-Time Analysis</button>
            <button className="btn btn-ghost" onClick={() => setPage('enroll')}>➕ Enroll New Staff Member</button>
            <button className="btn btn-ghost" onClick={() => setPage('staffdb')}>👥 View Staff Database</button>
          </div>
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 14 }}>Enrolled Staff ({activeStaff.length})</h2>
          {activeStaff.length === 0
            ? <span className="chip-empty">No staff enrolled — go to Enroll Staff tab</span>
            : <div className="staff-chips">
                {activeStaff.map(s => (
                  <span key={s.staff_id} className="chip">
                    <span>👤</span> {s.name}
                    <span style={{ color: 'var(--muted)', fontSize: '0.7rem' }}>({s.role})</span>
                  </span>
                ))}
              </div>
          }
        </div>
      </div>
    </div>
  )
}

function IdCards({ data }) {
  const { segments, speaker_colour } = data
  const bySpk = {}
  segments.forEach(s => {
    if (!bySpk[s.speaker_id] || s.score > bySpk[s.speaker_id].score) bySpk[s.speaker_id] = s
  })
  return (
    <div className="id-panel">
      {Object.entries(bySpk).sort((a, b) => a[0].localeCompare(b[0])).map(([spk, s]) => {
        const col = speaker_colour[spk] || '#4f8ef7'
        const isStaff = s.role === 'staff'
        const pct = Math.round(s.score * 100)
        const barCol = isStaff ? 'var(--green)' : 'var(--cyan)'
        return (
          <div key={spk} className="id-card">
            <div className="spk-id" style={{ color: col }}>{spk}</div>
            <div className="spk-name">{isStaff ? s.name : 'Unknown Customer'}</div>
            <div className="sim-row">
              <div className="sim-label">Similarity</div>
              <div className="sim-bar-bg"><div className="sim-bar-fill" style={{ width: `${pct}%`, background: barCol }} /></div>
              <div className="sim-val" style={{ color: barCol }}>{s.score.toFixed(2)}</div>
            </div>
            <div><span className={`role-badge ${isStaff ? 'role-staff' : 'role-customer'}`}>{isStaff ? '✓ Staff' : '◇ Customer'}</span></div>
          </div>
        )
      })}
    </div>
  )
}
