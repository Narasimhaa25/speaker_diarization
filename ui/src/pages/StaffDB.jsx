import { useState, useEffect, useMemo } from 'react'
import { fmtDate } from '../utils'

export default function StaffDB({ setPage }) {
  const [staff, setStaff] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState('name')
  const [sortDir, setSortDir] = useState(1)
  const [filter, setFilter] = useState('all') // all | active | inactive

  async function load() {
    setLoading(true); setError('')
    try {
      const resp = await fetch('/staff')
      const data = await resp.json()
      setStaff(data.staff || [])
    } catch (err) { setError('Error loading staff: ' + err.message) }
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  async function confirmDelete() {
    if (!deleteTarget) return
    try {
      const resp = await fetch(`/staff/${deleteTarget.id}`, { method: 'DELETE' })
      const data = await resp.json()
      if (data.success) { setDeleteTarget(null); load() }
      else alert('Error: ' + (data.error || 'Unknown error'))
    } catch (err) { alert('Request failed: ' + err.message) }
  }

  async function deactivate(id, name) {
    if (!confirm(`Deactivate ${name}? They will be excluded from identification but the record is kept.`)) return
    try {
      const resp = await fetch(`/staff/${id}/deactivate`, { method: 'POST' })
      const data = await resp.json()
      if (data.success) load()
      else alert('Error: ' + (data.error || 'Unknown error'))
    } catch (err) { alert('Request failed: ' + err.message) }
  }

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => -d)
    else { setSortKey(key); setSortDir(1) }
  }

  const SortIcon = ({ col }) => {
    if (sortKey !== col) return <span style={{ color: 'var(--subtle)', marginLeft: 4 }}>⇅</span>
    return <span style={{ color: 'var(--blue)', marginLeft: 4 }}>{sortDir > 0 ? '↑' : '↓'}</span>
  }

  const displayed = useMemo(() => {
    let list = [...staff]
    if (filter === 'active')   list = list.filter(s => s.active)
    if (filter === 'inactive') list = list.filter(s => !s.active)
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(s => s.name.toLowerCase().includes(q) || s.role.toLowerCase().includes(q) || s.staff_id.toLowerCase().includes(q))
    }
    list.sort((a, b) => {
      const av = a[sortKey] ?? '', bv = b[sortKey] ?? ''
      return typeof av === 'string'
        ? av.localeCompare(bv) * sortDir
        : (av - bv) * sortDir
    })
    return list
  }, [staff, search, sortKey, sortDir, filter])

  const active   = staff.filter(s => s.active).length
  const inactive = staff.length - active
  const totalSamples = staff.reduce((a, s) => a + s.n_samples, 0)

  return (
    <div className="main">
      {/* Stats */}
      <div className="stats-grid" style={{ marginBottom: 20 }}>
        <div className="stat green"><div className="val">{active}</div><div className="lbl">Active Staff</div></div>
        <div className="stat"><div className="val">{staff.length}</div><div className="lbl">Total Records</div></div>
        <div className="stat amber"><div className="val">{inactive}</div><div className="lbl">Inactive</div></div>
        <div className="stat blue"><div className="val">{totalSamples}</div><div className="lbl">Voice Samples</div></div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title-row">
            <div className="card-icon green">🔐</div>
            <h2>Staff Voice Database</h2>
          </div>
          <button className="btn btn-sm" onClick={() => setPage('enroll')}>➕ Add Staff</button>
        </div>

        <p style={{ color: 'var(--muted)', fontSize: '0.78rem', marginBottom: 16 }}>
          Encrypted local storage · AES-256-GCM · No cloud · No raw audio stored
        </p>

        {/* Toolbar */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 16, flexWrap: 'wrap' }}>
          <div className="search-wrap">
            <span className="search-icon">🔍</span>
            <input
              className="search-bar"
              placeholder="Search name, role, ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {['all','active','inactive'].map(f => (
              <button
                key={f}
                className={`btn btn-sm${filter === f ? '' : ' btn-ghost'}`}
                onClick={() => setFilter(f)}
                style={{ fontSize: '0.75rem', padding: '5px 12px' }}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
                {f === 'active' && active > 0 && <span style={{ marginLeft: 4, background: 'rgba(255,255,255,0.2)', borderRadius: 99, padding: '0 5px', fontSize: '0.68rem' }}>{active}</span>}
              </button>
            ))}
          </div>
          {search && (
            <span style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>
              {displayed.length} result{displayed.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {loading && (
          <div className="empty-state">
            <span className="spinner" style={{ width: 20, height: 20 }} />
            <p style={{ marginTop: 12 }}>Loading staff database…</p>
          </div>
        )}
        {error && <div className="error-box">⚠ {error}</div>}

        {!loading && !error && (
          displayed.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">👥</div>
              <p>
                {staff.length === 0
                  ? 'No staff enrolled yet.'
                  : 'No results match your search.'}
              </p>
              {staff.length === 0 && (
                <button className="btn btn-sm" style={{ marginTop: 14 }} onClick={() => setPage('enroll')}>
                  ➕ Enroll First Staff Member
                </button>
              )}
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th className="sortable" onClick={() => toggleSort('staff_id')}>ID <SortIcon col="staff_id" /></th>
                    <th className="sortable" onClick={() => toggleSort('name')}>Name <SortIcon col="name" /></th>
                    <th className="sortable" onClick={() => toggleSort('role')}>Role <SortIcon col="role" /></th>
                    <th className="sortable" onClick={() => toggleSort('n_samples')}>Samples <SortIcon col="n_samples" /></th>
                    <th className="sortable" onClick={() => toggleSort('enrolled_at')}>Enrolled <SortIcon col="enrolled_at" /></th>
                    <th className="sortable" onClick={() => toggleSort('updated_at')}>Updated <SortIcon col="updated_at" /></th>
                    <th className="sortable" onClick={() => toggleSort('active')}>Status <SortIcon col="active" /></th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {displayed.map(s => (
                    <tr key={s.staff_id} style={{ opacity: s.active ? 1 : 0.5 }}>
                      <td>
                        <code style={{ fontFamily: 'monospace', fontSize: '0.72rem', color: 'var(--muted)', background: 'var(--surface2)', padding: '2px 6px', borderRadius: 4 }}>
                          {s.staff_id.substring(0, 8)}…
                        </code>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'linear-gradient(135deg,var(--blue-dim),var(--blue))', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 700, flexShrink: 0 }}>
                            {s.name.charAt(0).toUpperCase()}
                          </div>
                          <strong>{s.name}</strong>
                        </div>
                      </td>
                      <td>
                        <span className="tag">{s.role}</span>
                      </td>
                      <td style={{ color: 'var(--cyan)', fontWeight: 600 }}>{s.n_samples}</td>
                      <td style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>{fmtDate(s.enrolled_at)}</td>
                      <td style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>{fmtDate(s.updated_at)}</td>
                      <td>
                        {s.active
                          ? <span className="role-staff">ACTIVE</span>
                          : <span className="role-unknown">INACTIVE</span>}
                      </td>
                      <td>
                        <div className="staff-actions">
                          <button className="btn btn-xs btn-ghost" onClick={() => setPage('enroll')} title="Re-enroll">↺</button>
                          {s.active
                            ? <button className="btn btn-xs btn-warning" onClick={() => deactivate(s.staff_id, s.name)} title="Deactivate">Deactivate</button>
                            : <button className="btn btn-xs btn-ghost" disabled>Inactive</button>}
                          <button className="btn btn-xs btn-danger" onClick={() => setDeleteTarget({ id: s.staff_id, name: s.name })} title="Delete permanently">🗑</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </div>

      {/* Delete modal */}
      <div className={`modal-overlay${deleteTarget ? ' open' : ''}`} onClick={e => e.target === e.currentTarget && setDeleteTarget(null)}>
        <div className="modal">
          <h3>🗑 Delete Staff Member</h3>
          <p>
            Permanently delete the voice profile for <strong style={{ color: 'var(--text)' }}>{deleteTarget?.name}</strong>?
            This action cannot be undone and the embedding will be lost.
          </p>
          <div className="modal-footer">
            <button className="btn btn-ghost" onClick={() => setDeleteTarget(null)}>Cancel</button>
            <button className="btn btn-danger" onClick={confirmDelete}>Delete Permanently</button>
          </div>
        </div>
      </div>
    </div>
  )
}
