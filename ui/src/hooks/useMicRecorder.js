import { useState, useRef, useCallback } from 'react'

// Pick the best MIME type — prefer formats librosa can decode without ffmpeg.
// WAV is universally decodable; webm/ogg need ffmpeg on the server.
function pickMime() {
  const candidates = [
    'audio/wav',               // ideal: no server-side conversion needed
    'audio/webm;codecs=opus',  // Chrome default
    'audio/ogg;codecs=opus',   // Firefox default
    'audio/webm',              // generic webm fallback
  ]
  for (const m of candidates) {
    if (MediaRecorder.isTypeSupported(m)) return m
  }
  return ''
}

export default function useMicRecorder() {
  const [state, setState]     = useState('idle') // idle | recording | done
  const [blob, setBlob]       = useState(null)
  const [seconds, setSeconds] = useState(0)
  const [error, setError]     = useState('')

  const mrRef       = useRef(null)
  const streamRef   = useRef(null)
  const chunksRef   = useRef([])
  const timerRef    = useRef(null)
  const mimeRef     = useRef('')
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)

  const start = useCallback(async () => {
    setError(''); setBlob(null); setSeconds(0)

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
        video: false,
      })
    } catch (err) {
      setError('Microphone access denied: ' + err.message)
      return
    }
    streamRef.current = stream

    // Waveform analyser
    const actx = new (window.AudioContext || window.webkitAudioContext)()
    audioCtxRef.current = actx
    const analyser = actx.createAnalyser()
    analyser.fftSize = 256
    actx.createMediaStreamSource(stream).connect(analyser)
    analyserRef.current = analyser

    const mime = pickMime()
    mimeRef.current = mime
    const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : {})
    chunksRef.current = []
    mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    mr.onstop = () => {
      const b = new Blob(chunksRef.current, { type: mime || 'audio/webm' })
      if (b.size < 1000) {
        setError('Recording too short — please speak for at least 1 second.')
        setState('idle')
        return
      }
      setBlob(b)
      setState('done')
    }
    mr.start(100)  // 100ms chunks for lower latency
    mrRef.current = mr
    setState('recording')
    timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)
  }, [])

  const stop = useCallback(() => {
    if (mrRef.current?.state === 'recording') mrRef.current.stop()
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    clearInterval(timerRef.current)
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null }
    analyserRef.current = null
  }, [])

  const clear = useCallback(() => {
    setBlob(null); setSeconds(0); setError(''); setState('idle')
    stop()
  }, [stop])

  return { state, blob, seconds, error, mimeRef, analyserRef, start, stop, clear }
}
