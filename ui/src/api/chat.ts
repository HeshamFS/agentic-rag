import { api } from './client'
import type { QueryResponse, PipelineConfig } from '../types'

export interface QueryRequest {
  question: string
  collection: string
  // Retrieval settings
  top_k?: number
  use_hyde?: boolean
  use_multi_query?: boolean
  use_reranking?: boolean
  // Generation settings
  temperature?: number
  // Full config override (optional)
  config?: Partial<PipelineConfig>
}

export const chatApi = {
  query: (request: QueryRequest) =>
    api.post<QueryResponse>('/query', request),

  queryStream: (
    request: QueryRequest,
    onChunk: (chunk: string) => void
  ) => api.stream('/query/stream', request, onChunk),

  // Search endpoint for retrieval-only queries
  search: (query: string, collection: string, top_k: number = 10) =>
    api.post<{
      chunks: Array<{
        content: string
        source: string
        score: number
        metadata: Record<string, unknown>
      }>
    }>('/search', { query, collection, top_k }),
}
