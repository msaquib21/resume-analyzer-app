export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

export async function getHealth(timeoutMs = 8000) {
  const controller = new AbortController()
  const t = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { signal: controller.signal })
    if (!res.ok) throw new Error('bad status')
    return await res.json()
  } finally {
    clearTimeout(t)
  }
}

/**
 * Calls POST /analyze/stream and reads the Server-Sent Events response body
 * manually via the Fetch Streams API (EventSource only supports GET requests,
 * and this endpoint needs a multipart POST body for the PDF upload).
 */
export async function analyzeStream({ jobDescription, file, model }, onEvent) {
  const form = new FormData()
  form.append('job_description', jobDescription)
  form.append('resume_pdf', file)
  if (model) form.append('model', model)

  const res = await fetch(`${BACKEND_URL}/analyze/stream`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok || !res.body) {
    throw new Error(`Analysis request failed (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult = null
  let errorMessage = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sepIndex
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)

      const line = rawEvent.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const payload = line.slice(5).trim()
      if (!payload) continue

      let evt
      try {
        evt = JSON.parse(payload)
      } catch {
        continue
      }

      if (evt.event === 'result') {
        finalResult = evt.data
      } else if (evt.event === 'error') {
        errorMessage = evt.message
      } else {
        onEvent?.(evt.event, evt.message)
      }
    }
  }

  if (errorMessage) throw new Error(errorMessage)
  if (!finalResult) throw new Error('The analysis stream ended without a result.')
  return finalResult
}

export async function getHistory() {
  const res = await fetch(`${BACKEND_URL}/history`)
  if (!res.ok) throw new Error('Could not load history.')
  return res.json()
}

export async function getHistoryTrend() {
  const res = await fetch(`${BACKEND_URL}/history/trend`)
  if (!res.ok) throw new Error('Could not load trend data.')
  return res.json()
}

export async function deleteHistoryItem(id) {
  const res = await fetch(`${BACKEND_URL}/history/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Could not delete this analysis.')
  return res.json()
}

export async function exportPdf(analysisId) {
  const form = new FormData()
  form.append('analysis_id', analysisId)
  const res = await fetch(`${BACKEND_URL}/export/pdf`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Could not generate the PDF report.')
  return res.blob()
}
