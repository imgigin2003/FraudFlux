export async function fetchModelInfo() {
  const res = await fetch('/api/model')
  if (!res.ok) throw new Error('Could not load model info')
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error('API unreachable')
  return res.json()
}

function detailToMessage(body, status) {
  const d = body?.detail
  if (typeof d === 'string') return d
  if (d && d.message) {
    const parts = [d.message]
    if (d.missing?.length) parts.push(`Missing: ${d.missing.join(', ')}`)
    if (d.unexpected?.length) parts.push(`Unexpected: ${d.unexpected.join(', ')}`)
    if (d.columns?.length) parts.push(`Columns: ${d.columns.join(', ')}`)
    return parts.join(' ')
  }
  return `Request failed (${status})`
}

// Serverless functions cap request bodies (4.5 MB on Vercel), so large CSVs are
// scored in chunks of rows and merged here.
const CHUNK_BYTES = 3_000_000

async function postForm(url, fields) {
  const form = new FormData()
  for (const [k, v] of Object.entries(fields)) {
    if (v !== undefined && v !== null) form.append(k, v instanceof Blob ? v : String(v))
  }
  const res = await fetch(url, { method: 'POST', body: form })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(detailToMessage(body, res.status))
  return body
}

function csvBlob(header, lines) {
  return new Blob([[header, ...lines].join('\n')], { type: 'text/csv' })
}

// Split into header + data lines, skipping blank lines like pandas does, so
// line i is the row the API reports as index i.
async function readCsv(file) {
  const all = (await file.text()).split(/\r?\n/).filter((l) => l.trim() !== '')
  return { header: all[0] ?? '', lines: all.slice(1) }
}

export async function scoreCsv(file, threshold) {
  const source = await readCsv(file)
  const { header, lines } = source

  const chunks = []
  let start = 0
  while (start < lines.length || chunks.length === 0) {
    let end = start
    let bytes = header.length
    while (end < lines.length && (end === start || bytes + lines[end].length < CHUNK_BYTES)) {
      bytes += lines[end].length + 1
      end += 1
    }
    chunks.push([start, end])
    start = end
  }

  const rows = []
  let total = 0, fraud = 0, flagged = 0, probSum = 0, maxProb = 0, result = null
  for (const [from, to] of chunks) {
    const part = await postForm('/api/predict', {
      file: new File([csvBlob(header, lines.slice(from, to))], file.name, { type: 'text/csv' }),
      threshold,
    })
    for (const r of part.rows) rows.push({ ...r, index: r.index + from })
    const s = part.summary
    total += s.total
    fraud += s.fraud
    flagged += s.flagged_amount
    probSum += s.mean_probability * s.total
    maxProb = Math.max(maxProb, s.max_probability)
    result = part
  }

  return {
    ...result,
    filename: file.name,
    source,
    summary: {
      total,
      fraud,
      not_fraud: total - fraud,
      fraud_rate: +(fraud / total).toFixed(6),
      flagged_amount: +flagged.toFixed(2),
      mean_probability: +(probSum / total).toFixed(6),
      max_probability: maxProb,
    },
    rows,
  }
}

export async function explainRow(source, index, threshold) {
  return postForm('/api/explain', {
    file: csvBlob(source.header, [source.lines[index]]),
    index,
    threshold,
  })
}
