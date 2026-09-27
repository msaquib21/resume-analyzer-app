import { useEffect, useState } from 'react'
import { getHistory, getHistoryTrend, deleteHistoryItem } from '../api.js'
import { ErrorView } from './AnalysisPage.jsx'

function TrendChart({ points }) {
  const width = 600
  const height = 140
  const pad = 20
  const max = 100
  const stepX = points.length > 1 ? (width - pad * 2) / (points.length - 1) : 0
  const coords = points
    .map((p, i) => {
      const x = pad + i * stepX
      const y = height - pad - (p.score / max) * (height - pad * 2)
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="trend-chart">
      <h4>Match score over time</h4>
      <svg viewBox={`0 0 ${width} ${height}`} className="trend-svg">
        <polyline points={coords} fill="none" stroke="var(--accent)" strokeWidth="2" />
      </svg>
    </div>
  )
}

export default function HistoryPage() {
  const [items, setItems] = useState([])
  const [trend, setTrend] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => { load() }, [])

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const [historyData, trendData] = await Promise.all([getHistory(), getHistoryTrend()])
      setItems(historyData)
      setTrend(trendData)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete(id) {
    await deleteHistoryItem(id)
    load()
  }

  if (loading) return <p className="empty-msg">Loading history…</p>
  if (error) return <ErrorView message={error} />

  return (
    <div className="history-page">
      {trend.length >= 2 && <TrendChart points={trend} />}
      {items.length === 0 ? (
        <p className="empty-msg">No analyses yet — run one from the New Analysis tab.</p>
      ) : (
        <ul className="history-list">
          {items.map((item) => (
            <li key={item.id} className="history-item">
              <div>
                <div className="history-file">{item.resume_filename}</div>
                <div className="history-meta">{new Date(item.timestamp).toLocaleString()} · {item.gap_count} gaps · {item.keyword_match_pct}% keywords</div>
                <div className="history-jd">{item.jd_snippet}…</div>
              </div>
              <div className="history-score">{item.score}/100</div>
              <button className="btn-ghost" onClick={() => handleDelete(item.id)}>Delete</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
