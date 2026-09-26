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

export async function scoreCsv(file, threshold) {
  const form = new FormData()
  form.append('file', file)
  if (threshold !== undefined && threshold !== null) form.append('threshold', String(threshold))

  const res = await fetch('/api/predict', { method: 'POST', body: form })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(detailToMessage(body, res.status))
  return body
}

export async function explainRow(jobId, index) {
  const res = await fetch(`/api/explain/${jobId}/${index}`)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(detailToMessage(body, res.status))
  return body
}
