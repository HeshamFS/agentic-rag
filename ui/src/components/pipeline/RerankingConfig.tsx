import { useSettingsStore } from '../../stores/settingsStore'
import { Select } from '../common/Select'
import { Slider } from '../common/Slider'
import { Toggle } from '../common/Toggle'

export function RerankingConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { reranking } = config

  return (
    <div className="space-y-4">
      <Toggle
        checked={reranking.enabled}
        onChange={(checked) => updateConfig('reranking', { enabled: checked })}
        label="Enable Reranking"
        description="Re-score and re-order retrieved chunks"
      />

      {reranking.enabled && (
        <>
          <Select
            label="Reranker Model"
            value={reranking.model}
            onChange={(value) => updateConfig('reranking', { model: value as typeof reranking.model })}
            options={[
              { value: 'jinaai/jina-reranker-v2-base-multilingual', label: 'Jina Reranker v2', description: 'Multilingual (Default)' },
              { value: 'jinaai/jina-colbert-v2', label: 'ColBERT v2', description: 'Late interaction, highest quality' },
            ]}
          />

          <Slider
            label="Rerank Top K"
            value={reranking.top_k}
            onChange={(value) => updateConfig('reranking', { top_k: value })}
            min={1}
            max={20}
            step={1}
          />
        </>
      )}
    </div>
  )
}
