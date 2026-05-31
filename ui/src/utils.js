export function fmt(s) {
  const m = Math.floor(s / 60)
  const sec = (s % 60).toFixed(1).padStart(4, '0')
  return m > 0 ? `${m}m${sec}s` : `${parseFloat(sec).toFixed(1)}s`
}

export function fmtDate(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })
  } catch {
    return iso.substring(0, 10)
  }
}

export const SPEAKER_COLOURS = ['#4f8ef7', '#f7954f', '#4fc97a', '#f74f4f', '#b44ff7', '#f7e04f']
