import { useEffect, useState } from 'react'
import { getHealth } from './api.js'
import AnalysisPage from './pages/AnalysisPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'

const GITHUB_URL = import.meta.env.VITE_GITHUB_REPOSITORY_URL || 'https://github.com/msaquib21/resume-analyzer-app'

function StatusPill({ health, healthError }) {
  if (healthError || !health) {
    return <div className="status-pill status-down">Backend not reachable</div>
  }
  const isGroq = health.llm_provider === 'groq'
  const ok = isGroq ? health.groq_configured : health.ollama_reachable
  return (
    <div className={`status-pill ${ok ? 'status-ok' : 'status-warn'}`}>
      {isGroq ? 'Groq Cloud' : 'Local Ollama'} · {ok ? 'Connected' : 'Not configured'}
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('analysis')
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(false)

  useEffect(() => {
    let cancelled = false
    getHealth()
      .then((h) => { if (!cancelled) setHealth(h) })
      .catch(() => { if (!cancelled) setHealthError(true) })
    return () => { cancelled = true }
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Resume AI<span className="brand-sub">Agentic Resume &amp; JD Matcher</span></div>
        <nav className="nav">
          <button className={page === 'analysis' ? 'nav-btn active' : 'nav-btn'} onClick={() => setPage('analysis')}>New analysis</button>
          <button className={page === 'history' ? 'nav-btn active' : 'nav-btn'} onClick={() => setPage('history')}>History</button>
        </nav>
        <StatusPill health={health} healthError={healthError} />
      </header>

      {health?.llm_provider === 'groq' && (
        <div className="cloud-banner">
          <strong>Public cloud demo</strong>
          <span>This instance runs on hosted Groq inference so anyone can try it online. The private local-Ollama architecture lives in the repository.</span>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">View local architecture on GitHub</a>
        </div>
      )}

      <main className="main">
        {page === 'analysis' ? <AnalysisPage /> : <HistoryPage />}
      </main>
    </div>
  )
}
