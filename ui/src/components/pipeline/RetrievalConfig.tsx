import { useSettingsStore } from '../../stores/settingsStore'
import { Select } from '../common/Select'
import { Slider } from '../common/Slider'
import { Toggle } from '../common/Toggle'

export function RetrievalConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { retrieval } = config

  return (
    <div className="space-y-4">
      <Select
        label="Strategy"
        value={retrieval.strategy}
        onChange={(value) => updateConfig('retrieval', { strategy: value as typeof retrieval.strategy })}
        options={[
          { value: 'hybrid', label: 'Hybrid', description: 'Dense + Sparse (recommended)' },
          { value: 'dense', label: 'Dense', description: 'Semantic search only' },
          { value: 'sparse', label: 'Sparse (BM25)', description: 'Keyword matching' },
        ]}
      />

      <Slider
        label="Top K Results"
        value={retrieval.top_k}
        onChange={(value) => updateConfig('retrieval', { top_k: value })}
        min={1}
        max={50}
        step={1}
      />

      {retrieval.strategy === 'hybrid' && (
        <>
          <Slider
            label="Sparse Weight"
            value={retrieval.sparse_weight}
            onChange={(value) => updateConfig('retrieval', { sparse_weight: value })}
            min={0}
            max={1}
            step={0.1}
            valueFormatter={(v) => v.toFixed(1)}
          />

          <Toggle
            checked={retrieval.use_rrf}
            onChange={(checked) => updateConfig('retrieval', { use_rrf: checked })}
            label="Reciprocal Rank Fusion"
            description="Better score combination method"
          />
        </>
      )}

      <Toggle
        checked={retrieval.use_hyde}
        onChange={(checked) => updateConfig('retrieval', { use_hyde: checked })}
        label="HyDE (Hypothetical Document Embeddings)"
        description="Generate hypothetical answer to improve retrieval"
      />

      <Toggle
        checked={retrieval.use_multi_query}
        onChange={(checked) => updateConfig('retrieval', { use_multi_query: checked })}
        label="Multi-Query Expansion"
        description="Generate multiple query variations"
      />
    </div>
  )
}
