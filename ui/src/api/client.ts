const API_BASE = ''

interface RequestOptions extends RequestInit {
  params?: Record<string, string>
}

async function request<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...init } = options

  let url = `${API_BASE}${endpoint}`
  if (params) {
    const searchParams = new URLSearchParams(params)
    url += `?${searchParams.toString()}`
  }

  const response = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

export const api = {
  get: <T>(endpoint: string, params?: Record<string, string>) =>
    request<T>(endpoint, { method: 'GET', params }),

  post: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    }),

  put: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    }),

  delete: <T>(endpoint: string) =>
    request<T>(endpoint, { method: 'DELETE' }),

  upload: async <T>(endpoint: string, formData: FormData): Promise<T> => {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  },

  stream: (endpoint: string, data: unknown, onChunk: (chunk: string) => void) => {
    const controller = new AbortController()

    const fetchStream = async () => {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Process complete SSE events (separated by double newlines)
        const events = buffer.split('\n\n')
        buffer = events.pop() || '' // Keep incomplete event in buffer

        for (const event of events) {
          if (!event.trim()) continue

          const lines = event.split('\n')
          let eventType = ''
          let eventData = ''

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7)
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6)
            }
          }

          if (eventData === '[DONE]') continue

          // Send the event data with event type info
          if (eventData) {
            try {
              const parsed = JSON.parse(eventData)
              // Normalize the response format based on event type
              if (eventType === 'chunk' && parsed.text) {
                onChunk(JSON.stringify({ type: 'token', token: parsed.text }))
              } else if (eventType === 'sources') {
                onChunk(JSON.stringify({ type: 'sources', sources: parsed }))
              } else if (eventType === 'queries') {
                // Multi-query variations
                onChunk(JSON.stringify({ type: 'queries', original: parsed.original, variations: parsed.variations }))
              } else if (eventType === 'step') {
                // Pipeline step progress
                onChunk(JSON.stringify({ type: 'step', step: parsed }))
              } else if (eventType === 'thinking') {
                // LLM reasoning (Gemini 3)
                onChunk(JSON.stringify({ type: 'thinking', thinking: parsed.content }))
              } else if (eventType === 'done') {
                // Final event with metadata
                onChunk(JSON.stringify({ type: 'done', done: true, provider: parsed.provider, model: parsed.model, confidence: parsed.confidence }))
              } else if (eventType === 'error') {
                onChunk(JSON.stringify({ type: 'error', error: parsed.error }))
              } else {
                onChunk(eventData)
              }
            } catch {
              onChunk(eventData)
            }
          }
        }
      }
    }

    return {
      promise: fetchStream(),
      cancel: () => controller.abort(),
    }
  },
}
