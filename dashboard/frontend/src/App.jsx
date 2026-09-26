import { useEffect, useRef, useState } from 'react'
import UploadZone from './components/UploadZone.jsx'
import Feed from './components/Feed.jsx'
import Stats from './components/Stats.jsx'
import { fetchHealth, fetchModelInfo, scoreCsv } from './api.js'

const STREAM_INTERVAL_MS = 120

export default function App() {
  const [model, setModel] = useState(null)
  const [health, setHealth] = useState(null)
  const [threshold, setThreshold] = useState(0.15)
  const [result, setResult] = useState(null)
  const [visible, setVisible] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('ff-theme') || 'dark')
  const timerRef = useRef(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('ff-theme', theme)
  }, [theme])

  useEffect(() => {
    fetchModelInfo()
      .then((m) => {
        setModel(m)
        setThreshold(m.threshold)
      })
      .catch(() => setModel(null))
    fetchHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }))
  }, [])

  // Stream scored rows into the feed one by one so it reads like a live monitor.
  useEffect(() => {
    if (!result) return
    clearInterval(timerRef.current)
    setVisible([])
    setStreaming(true)
    let i = 0
    timerRef.current = setInterval(() => {
      if (i >= result.rows.length) {
        clearInterval(timerRef.current)
        setStreaming(false)
        return
      }
      const next = result.rows[i]
      setVisible((v) => [next, ...v].slice(0, 200))
      i += 1
    }, STREAM_INTERVAL_MS)
    return () => clearInterval(timerRef.current)
  }, [result])

  async function handleFile(file) {
    setError(null)
    setLoading(true)
    try {
      const res = await scoreCsv(file, threshold)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function skipStream() {
    clearInterval(timerRef.current)
    if (result) setVisible([...result.rows].reverse().slice(0, 200))
    setStreaming(false)
  }

  function downloadResults() {
    if (!result) return
    const header = 'index,time,amount,fraud_probability,prediction\n'
    const lines = result.rows
      .map((r) => `${r.index},${r.time},${r.amount},${r.probability},${r.is_fraud ? 1 : 0}`)
      .join('\n')
    const blob = new Blob([header + lines], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `fraudflux_${(result.filename || 'results').replace(/\.csv$/i, '')}_scored.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const online = health?.status === 'ok'

  return (
    <div className="fraudflux">
      <header className="fx-head">
        <div className="fx-brand">
          <span className="fx-logo">FF</span> FraudFlux
        </div>
        <div className="fx-head-right">
          <div className={`fx-live ${online ? '' : 'off'}`}>
            <span className="d"></span>
            {online ? (streaming ? 'SCORING STREAM' : 'MODEL READY') : health?.status === 'degraded' ? 'MODEL NOT LOADED' : 'API OFFLINE'}
          </div>
          <button className="fx-icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Toggle theme">
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <section className="fx-controls">
        <UploadZone onFile={handleFile} loading={loading} disabled={!online} />
        <div className="fx-threshold">
          <label>
            <span className="k">THRESHOLD</span>
            <span className="v a">{threshold.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="0.99"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
          <p className="hint">Applies to the next upload. Default 0.15 is the deployed setting.</p>
        </div>
      </section>

      {error && <div className="fx-error">{error}</div>}

      <Stats model={model} result={result} />

      <Feed
        rows={visible}
        total={result?.rows.length || 0}
        streaming={streaming}
        onSkip={skipStream}
        onDownload={downloadResults}
        hasResult={!!result}
      />

      {model && (
        <footer className="fx-foot">
          Held-out test split ({model.test_transactions.toLocaleString()} transactions): recall {(model.recall * 100).toFixed(2)}%,
          precision {(model.precision * 100).toFixed(2)}%, F1 {(model.f1 * 100).toFixed(2)}%. Confusion matrix TN {model.confusion_matrix.tn.toLocaleString()} / FP {model.confusion_matrix.fp} / FN {model.confusion_matrix.fn} / TP {model.confusion_matrix.tp}.
        </footer>
      )}
    </div>
  )
}
