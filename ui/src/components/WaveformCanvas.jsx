import { useEffect, useRef } from 'react'

export default function WaveformCanvas({ analyserRef, active }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)

  useEffect(() => {
    if (!active || !analyserRef.current || !canvasRef.current) {
      cancelAnimationFrame(rafRef.current)
      return
    }
    const canvas = canvasRef.current
    const analyser = analyserRef.current
    const buf = new Uint8Array(analyser.frequencyBinCount)

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(buf)
      const ctx = canvas.getContext('2d')
      const W = canvas.offsetWidth || 400, H = canvas.offsetHeight || 48
      canvas.width = W; canvas.height = H
      ctx.clearRect(0, 0, W, H)
      ctx.strokeStyle = '#4f8ef7'; ctx.lineWidth = 1.5; ctx.beginPath()
      const sliceW = W / buf.length; let x = 0
      for (let i = 0; i < buf.length; i++) {
        const y = (buf[i] / 128) * (H / 2)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        x += sliceW
      }
      ctx.stroke()
    }
    draw()
    return () => cancelAnimationFrame(rafRef.current)
  }, [active, analyserRef])

  return (
    <canvas
      ref={canvasRef}
      className="waveform-canvas"
      style={{ display: active ? 'block' : 'none' }}
    />
  )
}
