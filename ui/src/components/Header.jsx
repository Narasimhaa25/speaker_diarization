export default function Header({ page, setPage, theme, toggleTheme }) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'analyse',   label: 'Analyse',   icon: '🎙' },
    { id: 'enroll',    label: 'Enroll',    icon: '➕' },
    { id: 'staffdb',   label: 'Staff DB',  icon: '👥' },
  ]

  return (
    <div className="header">
      <div className="header-brand">
        <div className="header-logo">🎤</div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h1>Speaker Diarization</h1>
            <span className="badge">Module 2</span>
          </div>
          <div className="subtitle">Who spoke when · Staff or customer · pyannote + ECAPA-TDNN</div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <nav className="nav-tabs">
          {tabs.map(t => (
            <button
              key={t.id}
              className={`nav-tab${page === t.id ? ' active' : ''}`}
              onClick={() => setPage(t.id)}
            >
              <span>{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </nav>

        <button
          className="theme-toggle"
          onClick={toggleTheme}
          title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </div>
    </div>
  )
}
