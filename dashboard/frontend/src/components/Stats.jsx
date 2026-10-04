function pct(x, digits = 2) {
  return `${(x * 100).toFixed(digits)}%`
}

export default function Stats({ model, result }) {
  const s = result?.summary
  return (
    <section className="fx-stats">
      <div className="fx-stat">
        <div className="k">MODEL</div>
        <div className="v small">{model ? model.model.toUpperCase() : '...'}</div>
      </div>
      <div className="fx-stat">
        <div className="k">RECALL</div>
        <div className="v g">{model ? pct(model.recall) : '...'}</div>
      </div>
      <div className="fx-stat">
        <div className="k">PRECISION</div>
        <div className="v g">{model ? pct(model.precision) : '...'}</div>
      </div>
      <div className="fx-stat">
        <div className="k">SCORED</div>
        <div className="v">{s ? s.total.toLocaleString() : '0'}</div>
      </div>
      <div className="fx-stat">
        <div className="k">FLAGGED</div>
        <div className={`v ${s && s.fraud > 0 ? 'r' : ''}`}>{s ? s.fraud.toLocaleString() : '0'}</div>
      </div>
      <div className="fx-stat">
        <div className="k">FRAUD RATE</div>
        <div className="v a">{s ? pct(s.fraud_rate, 3) : '0.000%'}</div>
      </div>
      <div className="fx-stat">
        <div className="k">FLAGGED AMOUNT</div>
        <div className={`v ${s && s.flagged_amount > 0 ? 'r' : ''}`}>
          {s ? `€ ${s.flagged_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '€ 0.00'}
        </div>
      </div>
    </section>
  )
}
