import { fmt } from '../utils'
import Timeline from './Timeline'
import IdPanel from './IdPanel'
import SpeakerBars from './SpeakerBars'
import PieChart from './PieChart'
import SegmentTable from './SegmentTable'

export default function AnalysisResults({ data, badge }) {
  if (!data) return null
  const trimNote = data.total_duration > data.duration
    ? ` of ${fmt(data.total_duration)}`
    : ''

  return (
    <div style={{ animation: 'fadeIn 0.3s ease' }}>
      {badge && <div className="rt-badge">{badge}</div>}

      <div className="stats-grid">
        <div className="stat">
          <div className="val val-sm">{fmt(data.duration)}{trimNote && <span style={{ fontSize: '0.65rem', color: 'var(--muted)' }}> {trimNote}</span>}</div>
          <div className="lbl">Processed</div>
        </div>
        <div className="stat blue"><div className="val">{data.est_speakers}</div><div className="lbl">Speakers</div></div>
        <div className="stat blue"><div className="val">{data.segments.length}</div><div className="lbl">Segments</div></div>
        <div className="stat green"><div className="val">{data.staff_count}</div><div className="lbl">Staff Turns</div></div>
        <div className="stat amber"><div className="val">{data.customer_count}</div><div className="lbl">Customer Turns</div></div>
        <div className="stat"><div className="val">{data.elapsed}s</div><div className="lbl">Processing</div></div>
        <div className="stat"><div className="val">{data.rtf}x</div><div className="lbl">RTF</div></div>
        <div className="stat purple"><div className="val">{data.threshold}</div><div className="lbl">Threshold</div></div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title-row">
            <div className="card-icon blue">📊</div>
            <h2>Speaker Timeline</h2>
          </div>
          <span className="tag" style={{ fontSize: '0.7rem' }}>Hover for details</span>
        </div>
        <Timeline data={data} />
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title-row">
            <div className="card-icon green">🪪</div>
            <h2>Identification Panel</h2>
          </div>
        </div>
        <IdPanel data={data} />
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <div className="card-icon blue">⏱</div>
              <h2>Speaking Time</h2>
            </div>
          </div>
          <SpeakerBars data={data} />
        </div>
        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <div className="card-icon green">🍩</div>
              <h2>Staff vs Customer</h2>
            </div>
          </div>
          <PieChart data={data} />
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title-row">
            <div className="card-icon blue">📋</div>
            <h2>Segment Details</h2>
          </div>
          <span className="tag">{data.segments.length} segments</span>
        </div>
        <SegmentTable data={data} />
      </div>
    </div>
  )
}
