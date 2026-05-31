import { useEffect, useRef } from 'react'

export default function PieChart({ data }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!data || !canvasRef.current) return
    const { staff_count, customer_count } = data
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')
    const total  = (staff_count + customer_count) || 1
    const cx = 60, cy = 60, r = 50, innerR = 28
    const staffAngle = (staff_count / total) * 2 * Math.PI

    ctx.clearRect(0, 0, 120, 120)

    // Customer arc
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.arc(cx, cy, r, -Math.PI/2 + staffAngle, -Math.PI/2 + 2 * Math.PI)
    ctx.closePath()
    ctx.fillStyle = '#0891b2'
    ctx.fill()

    // Staff arc
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.arc(cx, cy, r, -Math.PI/2, -Math.PI/2 + staffAngle)
    ctx.closePath()
    ctx.fillStyle = '#059669'
    ctx.fill()

    // Donut hole
    ctx.beginPath()
    ctx.arc(cx, cy, innerR, 0, 2 * Math.PI)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--donut-bg').trim() || '#ffffff'
    ctx.fill()

    // Center text
    const staffPct = Math.round(staff_count / total * 100)
    ctx.fillStyle = '#f1f5f9'
    ctx.font = 'bold 14px -apple-system, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`${staffPct}%`, cx, cy - 4)
    ctx.font = '8px -apple-system, sans-serif'
    ctx.fillStyle = '#64748b'
    ctx.fillText('STAFF', cx, cy + 9)
  }, [data])

  if (!data) return null
  const { staff_count, customer_count, diar_mode } = data
  const total = (staff_count + customer_count) || 1

  return (
    <div className="pie-wrap">
      <canvas ref={canvasRef} width={120} height={120} style={{ flexShrink: 0 }} />
      <div className="pie-legend">
        <div className="pie-legend-row">
          <div className="pie-dot" style={{ background: '#059669' }} />
          <span><strong style={{ color: 'var(--green)' }}>{staff_count}</strong> Staff turns ({Math.round(staff_count / total * 100)}%)</span>
        </div>
        <div className="pie-legend-row">
          <div className="pie-dot" style={{ background: '#0891b2' }} />
          <span><strong style={{ color: 'var(--cyan)' }}>{customer_count}</strong> Customer turns ({Math.round(customer_count / total * 100)}%)</span>
        </div>
        <div className="divider" style={{ margin: '8px 0' }} />
        <div style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>
          🤖 {diar_mode || 'pyannote'}
        </div>
      </div>
    </div>
  )
}
