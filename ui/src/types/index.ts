// Collection types
export interface Collection {
  name: string
  document_count: number
  chunk_count: number
  created_at?: string
  updated_at?: string
}

// Pipeline execution types
export interface PipelineStep {
  name: string
  status: 'started' | 'completed' | 'failed'
  duration_ms?: number
  details?: Record<string, unknown>
}

export interface QueryVariations {
  original: string
  variations: string[]
}

// Message types
export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  evaluation?: EvaluationResult
  timestamp: Date
  isStreaming?: boolean
  // Pipeline transparency
  pipelineSteps?: PipelineStep[]
  queryVariations?: QueryVariations
  thinking?: string  // LLM reasoning (Gemini 3)
  provider?: string
  model?: string
  confidence?: number
}

export interface Source {
  citation?: number  // Citation number [1], [2], etc.
  content: string
  source: string
  filename?: string
  score: number
  chunk_id?: string
  metadata?: Record<string, unknown>
}

// Evaluation types
export interface EvaluationResult {
  context_precision?: number
  faithfulness?: number
  answer_relevancy?: number
  ragas_score?: number
  self_rag?: SelfRAGResult
}

export interface SelfRAGResult {
  isrel: 'FULLY' | 'PARTIALLY' | 'NOT'
  issup: 'FULLY' | 'PARTIALLY' | 'NOT'
  isuse: 'FULLY' | 'PARTIALLY' | 'NOT'
  overall_score: number
  should_regenerate: boolean
}

// Pipeline configuration types
export interface PipelineConfig {
  // Chunking
  chunking: {
    strategy: 'semantic' | 'hierarchical' | 'raptor' | 'contextual'
    chunk_size: number
    chunk_overlap: number
    raptor_levels?: number
    raptor_clustering?: 'gmm' | 'kmeans'
  }
  // Retrieval
  retrieval: {
    strategy: 'dense' | 'sparse' | 'hybrid'
    top_k: number
    use_hyde: boolean
    use_multi_query: boolean
    use_rrf: boolean
    sparse_weight: number
  }
  // Reranking (model names from config.py)
  reranking: {
    enabled: boolean
    model: string // e.g., 'jinaai/jina-reranker-v2-base-multilingual', 'jinaai/jina-colbert-v2'
    top_k: number
  }
  // Compression
  compression: {
    enabled: boolean
    method: 'extractive' | 'longllmlingua' | 'sentence'
    ratio: number
  }
  // Generation
  generation: {
    provider: 'claude' | 'openai' | 'gemini' | 'local'
    model: string
    temperature: number
    max_tokens: number
    reasoning_effort?: 'none' | 'low' | 'medium' | 'high' | 'xhigh' // GPT-5 only
  }
  // Caching
  caching: {
    backend: 'memory' | 'redis' | 'disabled'
    similarity_threshold: number
    ttl_seconds: number
  }
  // Agentic
  agentic: {
    enabled: boolean
    self_rag: boolean
    crag: boolean
    planning: boolean
    web_fallback: boolean
  }
}

// API response types
export interface QueryResponse {
  response: string
  sources: Source[]
  evaluation?: EvaluationResult
  cached?: boolean
  processing_time_ms?: number
}

// Matches backend UploadResponse in api.py
export interface UploadResponse {
  success: boolean
  file_id: string
  filename: string
  chunks_created: number
  collection: string
  processing_time_sec: number
  entities_extracted: number
  relationships_extracted: number
  cached: boolean
  // For UI convenience (derived from filename)
  documents_processed?: number
}

// UI state types
export type ViewMode = 'chat' | 'collections' | 'upload' | 'config' | 'evaluation'
