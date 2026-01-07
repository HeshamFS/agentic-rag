import { api } from './client'

export interface PipelineConfigResponse {
  provider: string
  model: string
  temperature: number
  reasoning_effort: string // GPT-5: none/low/medium/high/xhigh
  use_hyde: boolean
  use_multi_query: boolean
  use_reranking: boolean
  retrieval_strategy: string
  enable_self_rag: boolean
  enable_planning: boolean
}

export interface PipelineConfigRequest {
  provider?: string
  model?: string
  temperature?: number
  reasoning_effort?: string // GPT-5: none/low/medium/high/xhigh
  use_hyde?: boolean
  use_multi_query?: boolean
  use_reranking?: boolean
  retrieval_strategy?: string
  enable_self_rag?: boolean
  enable_planning?: boolean
}

export const configApi = {
  // Get current pipeline configuration
  get: () => api.get<PipelineConfigResponse>('/config'),

  // Update pipeline configuration
  update: (config: PipelineConfigRequest) =>
    api.put<PipelineConfigResponse>('/config', config),
}
