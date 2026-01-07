import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { PipelineConfig, ViewMode } from '../types'
import { configApi } from '../api/config'

interface SettingsState {
  // UI state
  sidebarOpen: boolean
  configPanelOpen: boolean
  viewMode: ViewMode

  // Pipeline configuration
  config: PipelineConfig

  // Sync state
  isSyncing: boolean
  lastSyncError: string | null

  // Actions
  setSidebarOpen: (open: boolean) => void
  setConfigPanelOpen: (open: boolean) => void
  setViewMode: (mode: ViewMode) => void
  updateConfig: <K extends keyof PipelineConfig>(
    section: K,
    updates: Partial<PipelineConfig[K]>
  ) => void
  resetConfig: () => void
  syncToBackend: () => Promise<void>
  fetchFromBackend: () => Promise<void>
}

// Defaults matching backend config.py
const defaultConfig: PipelineConfig = {
  chunking: {
    strategy: 'semantic', // with_chunking("semantic", ...)
    chunk_size: 512, // default_chunk_size = 512
    chunk_overlap: 50, // default_chunk_overlap = 50
    raptor_levels: 3, // raptor_max_levels = 3
    raptor_clustering: 'gmm', // raptor_clustering = "gmm"
  },
  retrieval: {
    strategy: 'hybrid', // with_retrieval("hybrid", ...)
    top_k: 5, // Reduced from 10 to 5 for better answer quality (matches old UI)
    use_hyde: false, // use_hyde=False in api.py
    use_multi_query: true, // use_multi_query=True, num_queries=4
    use_rrf: true, // use_rrf default
    sparse_weight: 0.3, // hybrid_sparse_weight = 0.3
  },
  reranking: {
    enabled: true,
    model: 'jinaai/jina-reranker-v2-base-multilingual', // reranker_model from config.py
    top_k: 5, // default_rerank_top_k = 5
  },
  compression: {
    enabled: false, // enable_compression = False
    method: 'extractive', // compression_type = "extractive"
    ratio: 0.5, // compression_ratio = 0.5
  },
  generation: {
    provider: 'claude', // llm_provider = "claude"
    model: 'claude-sonnet-4-5-20250929', // llm_model from config.py
    temperature: 0.3, // default_temperature = 0.3
    max_tokens: 4096, // default_max_tokens = 4096
    reasoning_effort: 'medium', // GPT-5 reasoning effort (none/low/medium/high/xhigh)
  },
  caching: {
    backend: 'memory', // cache_backend = "memory"
    similarity_threshold: 0.95, // cache_similarity_threshold = 0.95
    ttl_seconds: 3600, // cache_ttl_seconds = 3600
  },
  agentic: {
    enabled: true, // enable_reflection and enable_planning are true
    self_rag: true, // enable_reflection = True (Self-RAG)
    crag: false, // CRAG not enabled by default
    planning: true, // enable_planning = True
    web_fallback: false,
  },
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      sidebarOpen: true,
      configPanelOpen: false,
      viewMode: 'chat',
      config: defaultConfig,
      isSyncing: false,
      lastSyncError: null,

      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),

      setConfigPanelOpen: (configPanelOpen) => set({ configPanelOpen }),

      setViewMode: (viewMode) => set({ viewMode }),

      updateConfig: (section, updates) => {
        set((state) => ({
          config: {
            ...state.config,
            [section]: {
              ...state.config[section],
              ...updates,
            },
          },
        }))
        // Auto-sync to backend after update
        get().syncToBackend()
      },

      resetConfig: () => {
        set({ config: defaultConfig })
        get().syncToBackend()
      },

      // Sync current config to backend
      syncToBackend: async () => {
        const { config, isSyncing } = get()
        if (isSyncing) return

        set({ isSyncing: true, lastSyncError: null })

        try {
          await configApi.update({
            provider: config.generation.provider,
            model: config.generation.model,
            temperature: config.generation.temperature,
            reasoning_effort: config.generation.reasoning_effort, // GPT-5 reasoning effort
            use_hyde: config.retrieval.use_hyde,
            use_multi_query: config.retrieval.use_multi_query,
            use_reranking: config.reranking.enabled,
            retrieval_strategy: config.retrieval.strategy,
            enable_self_rag: config.agentic.self_rag,
            enable_planning: config.agentic.planning,
          })
          console.log('Config synced to backend')
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Failed to sync config'
          console.error('Failed to sync config:', message)
          set({ lastSyncError: message })
        } finally {
          set({ isSyncing: false })
        }
      },

      // Fetch config from backend and update local state
      fetchFromBackend: async () => {
        set({ isSyncing: true, lastSyncError: null })

        try {
          const backendConfig = await configApi.get()

          set((state) => ({
            config: {
              ...state.config,
              generation: {
                ...state.config.generation,
                provider: backendConfig.provider as 'claude' | 'openai' | 'gemini' | 'local',
                model: backendConfig.model,
                temperature: backendConfig.temperature,
                reasoning_effort: (backendConfig.reasoning_effort || 'medium') as 'none' | 'low' | 'medium' | 'high' | 'xhigh',
              },
              retrieval: {
                ...state.config.retrieval,
                use_hyde: backendConfig.use_hyde,
                use_multi_query: backendConfig.use_multi_query,
                strategy: backendConfig.retrieval_strategy as 'dense' | 'sparse' | 'hybrid',
              },
              reranking: {
                ...state.config.reranking,
                enabled: backendConfig.use_reranking,
              },
              agentic: {
                ...state.config.agentic,
                self_rag: backendConfig.enable_self_rag,
                planning: backendConfig.enable_planning,
              },
            },
          }))
          console.log('Config fetched from backend')
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Failed to fetch config'
          console.error('Failed to fetch config:', message)
          set({ lastSyncError: message })
        } finally {
          set({ isSyncing: false })
        }
      },
    }),
    {
      name: 'rag-optimizer-settings',
      partialize: (state) => ({ config: state.config }),
    }
  )
)
