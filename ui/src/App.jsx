import { useState, useEffect } from 'react'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'
import Analyse from './pages/Analyse'
import EnrollStaff from './pages/EnrollStaff'
import StaffDB from './pages/StaffDB'

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [lastResult, setLastResult] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light')

  const pages = { dashboard: Dashboard, analyse: Analyse, enroll: EnrollStaff, staffdb: StaffDB }
  const Page = pages[page] || Dashboard

  return (
    <>
      <Header page={page} setPage={setPage} theme={theme} toggleTheme={toggleTheme} />
      <Page setPage={setPage} lastResult={lastResult} setLastResult={setLastResult} />
    </>
  )
}
