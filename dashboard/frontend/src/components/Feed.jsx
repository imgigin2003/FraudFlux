function money(x) {
  return `€ ${x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

export default function Feed({ rows, total, streaming, onSkip, onDownload, hasResult }) {
  return (
    <section className="fx-feed">
      <div className="fx-feed-bar">
        <span className="k">
          {hasResult ? `${Math.min(rows.length, total).toLocaleString()} / ${total.toLocaleString()} shown` : 'No transactions scored yet'}
          {rows.length >= 200 && total > 200 ? ' (feed keeps the latest 200; download for all)' : ''}
        </span>
        <span className="fx-feed-actions">
          {streaming && (
            <button className="fx-btn" onClick={onSkip}>
              Skip animation
            </button>
          )}
          {hasResult && (
            <button className="fx-btn primary" onClick={onDownload}>
              Download scored CSV
            </button>
          )}
        </span>
      </div>

      <div className="fx-cols">
        <span className="c-id">TXN</span>
        <span className="c-time">TIME (s)</span>
        <span className="c-amt">AMOUNT</span>
        <span className="c-pr">P(FRAUD)</span>
        <span className="c-v">VERDICT</span>
      </div>

      <div className="fx-rows">
        {rows.length === 0 && (
          <div className="fx-empty">Upload a CSV to start the scoring stream.</div>
        )}
        {rows.map((r) => (
          <div key={r.index} className={`fx-row ${r.is_fraud ? 'hot' : ''}`}>
            <span className="c-id">#{r.index.toLocaleString()}</span>
            <span className="c-time">{r.time.toLocaleString()}</span>
            <span className="c-amt">{money(r.amount)}</span>
            <span className="c-pr">
              <span className="bar">
                <span className="fill" style={{ width: `${Math.max(2, r.probability * 100)}%` }}></span>
              </span>
              {r.probability.toFixed(3)}
            </span>
            <span className={`c-v ${r.is_fraud ? 'v-fraud' : 'v-ok'}`}>
              {r.is_fraud ? '! FRAUD' : 'NOT FRAUD'}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}
