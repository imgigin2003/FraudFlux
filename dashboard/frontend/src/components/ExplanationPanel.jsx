import { useEffect, useState } from 'react'

const DEFAULT_SHOWN = 10

function money(x) {
  return `\u20ac ${x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function signed(x, digits = 3) {
  return `${x > 0 ? '+' : ''}${x.toFixed(digits)}`
}

export default function ExplanationPanel({ row, threshold, explanation, loading, error, onRetry, onClose }) {
  const [showAll, setShowAll] = useState(false)

  // Reset the expanded state whenever a different transaction is selected.
  useEffect(() => setShowAll(false), [row?.index])

  const feats = explanation?.features || []
  const shown = showAll ? feats : feats.slice(0, DEFAULT_SHOWN)
  const maxAbs = feats.length ? Math.max(...feats.map((f) => Math.abs(f.contribution))) || 1 : 1
  const thr = explanation?.threshold ?? threshold

  return (
    <aside className="fx-panel" aria-label={`Explanation for transaction ${row.index}`}>
      <div className="fx-panel-head">
        <div>
          <div className="k">WHY THIS VERDICT</div>
          <div className="fx-panel-title">
            #{row.index.toLocaleString()} <span className="dim">{money(row.amount)}</span>
          </div>
        </div>
        <button className="fx-icon-btn" onClick={onClose} aria-label="Close explanation">
          Close
        </button>
      </div>

      <div className="fx-panel-verdict">
        <span className={row.is_fraud ? 'v-fraud' : 'v-ok'}>{row.is_fraud ? '! FRAUD' : 'NOT FRAUD'}</span>
        <span className="dim">
          P(fraud) {row.probability.toFixed(3)} {row.is_fraud ? '\u2265' : '<'} threshold {thr != null ? Number(thr).toFixed(2) : '...'}
        </span>
      </div>

      {loading && <div className="fx-panel-state">Computing feature contributions...</div>}

      {!loading && error && (
        <div className="fx-panel-state error">
          <div>Could not produce an explanation.</div>
          <div className="dim small">{error}</div>
          <button className="fx-btn" onClick={onRetry}>Retry</button>
        </div>
      )}

      {!loading && explanation && (
        <>
          <p className="fx-summary">{explanation.summary}</p>

          <div className="fx-baseline">
            <span className="k">BASELINE</span>
            <span>{explanation.baseline.toFixed(4)}</span>
            <span className="dim small">model's expected fraud probability before this row's features</span>
          </div>

          <div className="fx-contrib-head">
            <span className="k">FEATURE</span>
            <span className="k">VALUE</span>
            <span className="k right">CONTRIBUTION</span>
          </div>

          <ul className="fx-contribs">
            {shown.map((f) => {
              const pct = (Math.abs(f.contribution) / maxAbs) * 100
              const up = f.contribution > 0
              return (
                <li key={f.feature} className={`fx-contrib ${up ? 'up' : 'down'}`}>
                  <span className="name">{f.feature}</span>
                  <span className="val">{f.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}</span>
                  <span className="contrib">
                    <span className="cbar">
                      <span className="cfill" style={{ width: `${Math.max(2, pct)}%` }}></span>
                    </span>
                    <span className="num">{signed(f.contribution)}</span>
                  </span>
                </li>
              )
            })}
          </ul>

          {feats.length > DEFAULT_SHOWN && (
            <button className="fx-btn" onClick={() => setShowAll((s) => !s)}>
              {showAll ? `Show top ${DEFAULT_SHOWN}` : `Show all ${feats.length} features`}
            </button>
          )}

          <p className="hint">
            Positive contributions pushed toward fraud; negative pulled away. Contributions sum from the baseline to this row's probability.
            V1-V28 are anonymized PCA components from the source dataset.
          </p>
        </>
      )}
    </aside>
  )
}
