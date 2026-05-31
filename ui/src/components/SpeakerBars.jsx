export default function SpeakerBars({ data }) {
  if (!data) return null
  const { speaker_totals, speaker_colour } = data
  const maxT = Math.max(...Object.values(speaker_totals), 1)
  const totalT = Object.values(speaker_totals).reduce((a, b) => a + b, 0)

  return (
    <div className="speaker-bars">
      {Object.entries(speaker_totals)
        .sort((a, b) => b[1] - a[1])
        .map(([spk, t]) => {
          const pct = Math.round(t / totalT * 100)
          return (
            <div key={spk} className="speaker-row">
              <div className="speaker-name" title={spk}>{spk}</div>
              <div className="speaker-bar-bg">
                <div
                  className="speaker-bar-fill"
                  style={{ width: `${(t / maxT) * 100}%`, background: speaker_colour[spk] || '#4f8ef7' }}
                >
                  {t > 3 ? `${t.toFixed(1)}s` : ''}
                </div>
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--muted)', width: 30, textAlign: 'right', flexShrink: 0 }}>
                {pct}%
              </span>
            </div>
          )
        })}
    </div>
  )
}
