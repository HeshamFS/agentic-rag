import { X, RotateCcw, Loader2, Check, AlertCircle, RefreshCw } from 'lucide-react'
import { useSettingsStore } from '../../stores/settingsStore'
import { Button } from '../common/Button'
import { ChunkingConfig } from './ChunkingConfig'
import { RetrievalConfig } from './RetrievalConfig'
import { RerankingConfig } from './RerankingConfig'
import { CompressionConfig } from './CompressionConfig'
import { GenerationConfig } from './GenerationConfig'
import { CachingConfig } from './CachingConfig'
import { AgenticConfig } from './AgenticConfig'

interface PipelineConfigProps {
  onClose?: () => void
}

export function PipelineConfig({ onClose }: PipelineConfigProps) {
  const { resetConfig, isSyncing, lastSyncError, fetchFromBackend } = useSettingsStore()

  const handleReset = () => {
    if (confirm('Reset all pipeline settings to defaults?')) {
      resetConfig()
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-surface sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-foreground">Pipeline Configuration</h2>
          {/* Sync status indicator */}
          <div className="flex items-center gap-1.5">
            {isSyncing ? (
              <Loader2 className="h-4 w-4 text-accent animate-spin" />
            ) : lastSyncError ? (
              <span title={lastSyncError}>
                <AlertCircle className="h-4 w-4 text-error" />
              </span>
            ) : (
              <Check className="h-4 w-4 text-success" />
            )}
            <span className="text-xs text-muted">
              {isSyncing ? 'Syncing...' : lastSyncError ? 'Sync failed' : 'Synced'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchFromBackend()}
            className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
            title="Refresh from backend"
            disabled={isSyncing}
          >
            <RefreshCw className="h-4 w-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
            title="Reset to defaults"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {/* Chunking */}
        <ConfigSection title="Chunking" description="How documents are split into chunks">
          <ChunkingConfig />
        </ConfigSection>

        {/* Retrieval */}
        <ConfigSection title="Retrieval" description="How relevant chunks are found">
          <RetrievalConfig />
        </ConfigSection>

        {/* Reranking */}
        <ConfigSection title="Reranking" description="Re-order retrieved chunks by relevance">
          <RerankingConfig />
        </ConfigSection>

        {/* Compression */}
        <ConfigSection title="Context Compression" description="Reduce token usage while preserving information">
          <CompressionConfig />
        </ConfigSection>

        {/* Generation */}
        <ConfigSection title="Generation" description="LLM provider and model settings">
          <GenerationConfig />
        </ConfigSection>

        {/* Caching */}
        <ConfigSection title="Semantic Caching" description="Cache similar queries for faster responses">
          <CachingConfig />
        </ConfigSection>

        {/* Agentic */}
        <ConfigSection title="Agentic Features" description="Self-correction and advanced reasoning">
          <AgenticConfig />
        </ConfigSection>
      </div>

      {/* Footer */}
      {onClose && (
        <div className="px-6 py-4 border-t border-border bg-surface">
          <Button variant="primary" onClick={onClose} className="w-full">
            Apply Changes
          </Button>
        </div>
      )}
    </div>
  )
}

interface ConfigSectionProps {
  title: string
  description: string
  children: React.ReactNode
}

function ConfigSection({ title, description, children }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        <p className="text-xs text-muted mt-0.5">{description}</p>
      </div>
      <div className="space-y-4 pl-0">
        {children}
      </div>
    </div>
  )
}
