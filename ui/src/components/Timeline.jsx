import { useState } from 'react'
import { fmt } from '../utils'

export default function Timeline({ data }) {
  const [tooltip, setTooltip] = useState(null)
  if (!data) return null
  const { segments, duration, speaker_colour } = data

  return (
    <>
      <div className="timeline-wrap">
        <div className="timeline" style={{ position: 'relative' }}>
          {segments.map((s, i) => {
            const left  = (s.start / duration * 100).toFixed(2)
            const width = Math.max(s.duration / duration * 100, 0.3).toFixed(2)
            const col   = speaker_colour[s.speaker_id] || '#4f8ef7'
            const alpha = s.role === 'staff' ? 'ff' : '88'
            return (
              <div
                key={i}
                className="timeline-seg"
                style={{ left: `${left}%`, width: `${width}%`, background: `${col}${alpha}` }}
                onMouseEnter={e => setTooltip({ seg: s, x: parseFloat(left), i })}
                onMouseLeave={() => setTooltip(null)}
              >
                {parseFloat(width) > 4 && (
                  <span className="seg-label">{s.name || s.speaker_id}</span>
                )}
              </div>
            )
          })}

          {tooltip && (
            <div style={{
              position: 'absolute', bottom: '100%', marginBottom: 6,
              left: `clamp(0px, ${tooltip.x}%, calc(100% - 200px))`,
              background: 'var(--surface)', border: '1px solid var(--border2)',
              borderRadius: 8, padding: '8px 12px', fontSize: '0.75rem',
              pointerEvents: 'none', zIndex: 10, whiteSpace: 'nowrap',
              boxShadow: 'var(--shadow)', animation: 'fadeIn 0.15s ease',
            }}>
              <div style={{ fontWeight: 700, color: speaker_colour[tooltip.seg.speaker_id] || '#4f8ef7' }}>
                {tooltip.seg.speaker_id}
              </div>
              <div style={{ color: 'var(--text-dim)', marginTop: 2 }}>
                {fmt(tooltip.seg.start)} → {fmt(tooltip.seg.end)} · {tooltip.seg.duration.toFixed(1)}s
              </div>
              <div style={{ marginTop: 3 }}>
                <span className={`role-${tooltip.seg.role}`}>{tooltip.seg.role.toUpperCase()}</span>
                {tooltip.seg.name && <span style={{ color: 'var(--text)', marginLeft: 6 }}>{tooltip.seg.name}</span>}
              </div>
              <div style={{ color: 'var(--muted)', marginTop: 2 }}>
                sim {tooltip.seg.score.toFixed(3)}
              </div>
            </div>
          )}
        </div>

        <div className="timeline-ticks">
          {[0,1,2,3,4,5,6].map(i => (
            <span key={i}>{fmt(duration * i / 6)}</span>
          ))}
        </div>
      </div>

      <div className="legend">
        {Object.entries(speaker_colour).map(([spk, col]) => (
          <div key={spk} className="legend-item">
            <div className="legend-dot" style={{ background: col }} />{spk}
          </div>
        ))}
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#34d399' }} />Staff (solid)
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#4f8ef7', opacity: 0.5 }} />Customer (faded)
        </div>
      </div>
    </>
  )
}
