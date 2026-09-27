import { useEffect, useState } from 'react'
import { getHealth } from './api.js'
import AnalysisPage from './pages/AnalysisPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'

const GITHUB_URL = import.meta.env.VITE_GITHUB_REPOSITORY_URL || 'https://github.com/msaquib21/resume-analyzer-app'

function StatusPill({ health, healthError }) {
  if (healthError || !health) {
    return <div className="status-pill status-down">Backend unreachable</div>
  }
  const isGroq = health.llm_provider === 'groq'
  const ok = isGroq ? health.groq_configured : health.ollama_reachable
  return (
    <div className={`status-pill ${ok ? 'status-ok' : 'status-warn'}`}>
      {isGroq ? 'Groq cloud inference' : 'Local Ollama inference'}
    </div>
  )
}

function CloudNotice({ onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>You're viewing the hosted demo</h3>
        <p>
          This instance runs on cloud inference (Groq) so anyone can try it without installing anything.
          The version built for personal use runs entirely on-device — a local LLM via Ollama and a local
          vector store, with no data leaving the machine.
        </p>
        <p>The source, and instructions for running it locally, are here:</p>
        <a className="modal-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
          {GITHUB_URL.replace('https://', '')}
        </a>
        <button className="btn-primary" onClick={onClose}>Continue to the demo</button>
      </div>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('analysis')
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(false)
  const [showNotice, setShowNotice] = useState(false)

  useEffect(() => {
    let cancelled = false
    getHealth()
      .then((h) => { if (!cancelled) setHealth(h) })
      .catch(() => { if (!cancelled) setHealthError(true) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (health?.llm_provider === 'groq' && !sessionStorage.getItem('seenCloudNotice')) {
      setShowNotice(true)
      sessionStorage.setItem('seenCloudNotice', '1')
    }
  }, [health])

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <span className="brand-mark">RA</span>
          <div>
            <div className="brand-name">Resume Analyzer</div>
            <div className="brand-sub">Agentic JD matcher</div>
          </div>
        </div>

        <nav className="rail-nav">
          <button className={page === 'analysis' ? 'rail-btn active' : 'rail-btn'} onClick={() => setPage('analysis')}>New analysis</button>
          <button className={page === 'history' ? 'rail-btn active' : 'rail-btn'} onClick={() => setPage('history')}>History</button>
        </nav>

        <div className="rail-footer">
          <StatusPill health={health} healthError={healthError} />
          {health?.llm_provider === 'groq' && (
            <button className="rail-link" onClick={() => setShowNotice(true)}>About this demo</button>
          )}
        </div>
      </aside>

      <main className="content">
        {page === 'analysis' ? <AnalysisPage /> : <HistoryPage />}
      </main>

      {showNotice && <CloudNotice onClose={() => setShowNotice(false)} />}
    </div>
  )
}
