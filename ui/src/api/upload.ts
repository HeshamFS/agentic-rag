import type { UploadResponse } from '../types'

// Backend api.py upload endpoint expects:
// - file: UploadFile (single file)
// - collection: str = Form(default="default")
// - chunk_size: int = Form(default=512)
export interface UploadRequest {
  files: File[] // UI sends multiple, we upload one at a time
  collection: string
  chunkSize?: number // Maps to chunk_size
}

export interface UploadProgress {
  step: 'reading' | 'checking' | 'parsing' | 'chunking' | 'embedding' | 'storing'
  message: string
  percent: number
}

export const uploadApi = {
  // Streaming upload with progress updates
  uploadFileStream: async (
    request: UploadRequest,
    onProgress: (progress: UploadProgress) => void,
  ): Promise<UploadResponse> => {
    if (request.files.length === 0) {
      throw new Error('No files provided')
    }

    const formData = new FormData()
    formData.append('file', request.files[0])
    formData.append('collection', request.collection)
    if (request.chunkSize) {
      formData.append('chunk_size', request.chunkSize.toString())
    }

    const response = await fetch('/upload/stream', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''
    let result: UploadResponse | null = null

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Process SSE events
      const events = buffer.split('\n\n')
      buffer = events.pop() || ''

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

        if (eventData) {
          try {
            const parsed = JSON.parse(eventData)
            if (eventType === 'progress') {
              onProgress(parsed as UploadProgress)
            } else if (eventType === 'done') {
              result = parsed as UploadResponse
            } else if (eventType === 'error') {
              throw new Error(parsed.error)
            }
          } catch (e) {
            if (e instanceof Error && e.message !== eventData) {
              throw e
            }
          }
        }
      }
    }

    if (!result) {
      throw new Error('Upload completed without response')
    }

    return result
  },

  // Legacy non-streaming upload (fallback)
  uploadFiles: async (request: UploadRequest): Promise<UploadResponse> => {
    if (request.files.length === 0) {
      throw new Error('No files provided')
    }

    const formData = new FormData()
    formData.append('file', request.files[0])
    formData.append('collection', request.collection)
    if (request.chunkSize) {
      formData.append('chunk_size', request.chunkSize.toString())
    }

    const response = await fetch('/upload', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  },
}
