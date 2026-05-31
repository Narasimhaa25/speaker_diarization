import { fmt } from '../utils'

export default function SegmentTable({ data }) {
  if (!data) return null
  const { segments, speaker_colour, threshold } = data

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Time Range</th>
            <th>Duration</th>
            <th>Speaker</th>
            <th>Identified As</th>
            <th>Role</th>
            <th>Similarity</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {segments.map((s, i) => {
            const pct    = Math.round(s.score * 100)
            const barCol = s.score >= threshold ? 'var(--green)' : 'var(--cyan)'
            const col    = speaker_colour[s.speaker_id] || '#fff'
            return (
              <tr key={i}>
                <td style={{ color: 'var(--text-faint)', fontSize: '0.75rem' }}>{i + 1}</td>
                <td style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                  {fmt(s.start)} → {fmt(s.end)}
                </td>
                <td style={{ color: 'var(--text-dim)' }}>{s.duration.toFixed(1)}s</td>
                <td>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: col, display: 'inline-block', flexShrink: 0 }} />
                    <span style={{ color: col, fontWeight: 600, fontSize: '0.82rem' }}>{s.speaker_id}</span>
                  </span>
                </td>
                <td style={{ fontWeight: s.name ? 600 : 400, color: s.name ? 'var(--text)' : 'var(--text-faint)' }}>
                  {s.name || 'Unknown'}
                </td>
                <td><span className={`role-${s.role}`}>{s.role.toUpperCase()}</span></td>
                <td>
                  <div className="score-bar">
                    <div className="score-track">
                      <div className="score-fill" style={{ width: `${Math.round(s.score * 100)}%`, background: barCol }} />
                    </div>
                    <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)', width: 36, textAlign: 'right', fontFamily: 'monospace' }}>
                      {s.score.toFixed(3)}
                    </span>
                  </div>
                </td>
                <td style={{ color: pct >= 80 ? 'var(--green)' : pct >= 60 ? 'var(--amber)' : 'var(--red)', fontWeight: 600, fontSize: '0.82rem' }}>
                  {pct}%
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
