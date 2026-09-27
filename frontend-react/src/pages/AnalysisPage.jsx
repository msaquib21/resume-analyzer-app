import { useState } from 'react'
import { analyzeStream, exportPdf } from '../api.js'

const STAGE_LABELS = {
  started: 'Initializing pipeline…',
  extracting: 'Parsing resume & building embeddings…',
  retrieved: 'Multi-query retrieval complete',
  analyzing_gaps: 'Evaluating candidate against requirements…',
  gaps_found: 'Gaps classified',
  scoring: 'Synthesizing score & roadmap…',
  complete: 'Analysis complete',
}

const SAMPLE_JD = `Role: Senior AI / MLOps Engineer
Requirements:
- 3+ years experience with Python, FastAPI, and asynchronous backend microservices.
- Hands-on production experience building RAG pipelines using LangChain, LangGraph, and vector databases (ChromaDB, Pinecone, FAISS).
- Experience fine-tuning and deploying open-source LLMs (Llama, Qwen, Mistral) on AWS SageMaker or GCP.
- Strong software engineering fundamentals: automated testing, Docker, CI/CD.
- Strong communication skills and cross-functional leadership.`

export function ErrorView({ message }) {
  return (
    <div className="error-view">
      <strong>Analysis failed</strong>
      <p>{message}</p>
    </div>
  )
}

function ScoreGauge({ score }) {
  const radius = 60
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - score / 100)
  const color = score >= 80 ? 'var(--accent)' : score >= 50 ? 'var(--warn)' : 'var(--danger)'
  return (
    <div className="gauge">
      <svg viewBox="0 0 160 160">
        <circle className="gauge-bg" cx="80" cy="80" r={radius} />
        <circle
          cx="80" cy="80" r={radius}
          stroke={color}
          strokeWidth="10"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 80 80)"
        />
      </svg>
      <div className="gauge-label">
        <span className="gauge-score" style={{ color }}>{score}</span>
        <span className="gauge-max">/100</span>
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function ListPanel({ items, empty }) {
  if (!items || items.length === 0) return <p className="empty-msg">{empty}</p>
  return (
    <ol className="item-list">
      {items.map((text, i) => <li key={i}>{text}</li>)}
    </ol>
  )
}

function KeywordGrid({ keywords }) {
  const found = keywords.filter((k) => k.found_in_resume)
  const missing = keywords.filter((k) => !k.found_in_resume)
  return (
    <div className="keyword-grid">
      <div>
        <h4>Verified ({found.length})</h4>
        <div className="chips">{found.map((k, i) => <span key={i} className="chip chip-found">{k.keyword}</span>)}</div>
      </div>
      <div>
        <h4>Missing ({missing.length})</h4>
        <div className="chips">{missing.map((k, i) => <span key={i} className="chip chip-missing">{k.keyword}</span>)}</div>
      </div>
    </div>
  )
}

function ResultsView({ result }) {
  const [tab, setTab] = useState('gaps')
  const [downloading, setDownloading] = useState(false)
  const keywords = result.keywords || []
  const kwTotal = keywords.length
  const kwFound = keywords.filter((k) => k.found_in_resume).length

  async function handleDownload() {
    setDownloading(true)
    try {
      const blob = await exportPdf(result.analysis_id)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `resume_analysis_${result.analysis_id}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      alert(err.message)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="results">
      <ScoreGauge score={result.score} />
      <div className="stats-row">
        <Stat label="Gaps" value={result.gaps.length} />
        <Stat label="Keyword match" value={`${kwTotal ? Math.round((kwFound / kwTotal) * 100) : 0}%`} />
        <Stat label="Action items" value={result.improvements.length} />
        <Stat label="Prep steps" value={result.preparation.length} />
      </div>

      <div className="tabs">
        {['gaps', 'improvements', 'preparation', 'keywords'].map((t) => (
          <button key={t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
            {t === 'gaps' ? 'Skill gaps' : t === 'improvements' ? 'Improvements' : t === 'preparation' ? 'Interview prep' : 'Keywords'}
          </button>
        ))}
      </div>

      <div className="tab-body">
        {tab === 'gaps' && <ListPanel items={result.gaps} empty="No unevidenced requirements found." />}
        {tab === 'improvements' && <ListPanel items={result.improvements} empty="Nothing to add — the resume already covers this role." />}
        {tab === 'preparation' && <ListPanel items={result.preparation} empty="No prep items generated." />}
        {tab === 'keywords' && <KeywordGrid keywords={keywords} />}
      </div>

      <button className="btn-secondary" onClick={handleDownload} disabled={downloading}>
        {downloading ? 'Preparing PDF…' : 'Download PDF report'}
      </button>
    </div>
  )
}

function ProgressView({ stage }) {
  return (
    <div className="progress-view">
      <div className="spinner" />
      <p>{STAGE_LABELS[stage] || 'Working…'}</p>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <h3>Multi-agent resume matching</h3>
      <p>Paste a job description and upload a resume PDF to run a grounded, evidence-based match analysis.</p>
      <ul>
        <li>0–100 evidence-based match score</li>
        <li>Skill gaps linked to exact JD requirements</li>
        <li>Actionable resume bullet suggestions</li>
        <li>Interview prep roadmap</li>
        <li>ATS keyword coverage</li>
      </ul>
    </div>
  )
}

export default function AnalysisPage() {
  const [jobDescription, setJobDescription] = useState('')
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const canSubmit = jobDescription.trim().length > 0 && file && !loading

  async function handleSubmit(e) {
    e.preventDefault()
    if (!canSubmit) return
    setLoading(true)
    setError(null)
    setResult(null)
    setStage('started')
    try {
      const res = await analyzeStream({ jobDescription, file }, (stageKey) => setStage(stageKey))
      setResult(res)
    } catch (err) {
      setError(err.message || 'Analysis failed.')
    } finally {
      setLoading(false)
      setStage(null)
    }
  }

  return (
    <div className="analysis-grid">
      <form className="panel input-panel" onSubmit={handleSubmit}>
        <div className="panel-header">
          <h2>Target job description</h2>
          <span className="step-tag">Step 1 of 2</span>
        </div>
        <p className="panel-hint">Paste the requirements, skills, and responsibilities for semantic matching.</p>
        <div className="row-actions">
          <button type="button" className="btn-ghost" onClick={() => setJobDescription(SAMPLE_JD)}>Load sample JD</button>
          <button type="button" className="btn-ghost" onClick={() => setJobDescription('')}>Clear</button>
        </div>
        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          placeholder="Paste the target job description here…"
          rows={10}
        />

        <div className="panel-header">
          <h2>Candidate resume (PDF)</h2>
          <span className="step-tag">Step 2 of 2</span>
        </div>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />

        <button type="submit" className="btn-primary" disabled={!canSubmit}>
          {loading ? 'Running analysis…' : 'Run agentic analysis'}
        </button>
      </form>

      <div className="panel results-panel">
        {loading && <ProgressView stage={stage} />}
        {!loading && error && <ErrorView message={error} />}
        {!loading && !error && result && <ResultsView result={result} />}
        {!loading && !error && !result && <EmptyState />}
      </div>
    </div>
  )
}
