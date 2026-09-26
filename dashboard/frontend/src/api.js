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

export async function scoreCsv(file, threshold) {
  const form = new FormData()
  form.append('file', file)
  if (threshold !== undefined && threshold !== null) form.append('threshold', String(threshold))

  const res = await fetch('/api/predict', { method: 'POST', body: form })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const d = body.detail
    if (typeof d === 'string') throw new Error(d)
    if (d && d.message) {
      const parts = [d.message]
      if (d.missing?.length) parts.push(`Missing: ${d.missing.join(', ')}`)
      if (d.unexpected?.length) parts.push(`Unexpected: ${d.unexpected.join(', ')}`)
      if (d.columns?.length) parts.push(`Columns: ${d.columns.join(', ')}`)
      throw new Error(parts.join(' '))
    }
    throw new Error(`Request failed (${res.status})`)
  }
  return body
}
