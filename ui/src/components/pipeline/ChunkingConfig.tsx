import { useSettingsStore } from '../../stores/settingsStore'
import { Select } from '../common/Select'
import { Slider } from '../common/Slider'

export function ChunkingConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { chunking } = config

  return (
    <div className="space-y-4">
      <Select
        label="Strategy"
        value={chunking.strategy}
        onChange={(value) => updateConfig('chunking', { strategy: value as typeof chunking.strategy })}
        options={[
          { value: 'semantic', label: 'Semantic', description: 'Smart boundary detection' },
          { value: 'hierarchical', label: 'Hierarchical', description: 'Multi-level chunking' },
          { value: 'raptor', label: 'RAPTOR', description: 'Tree-organized retrieval' },
          { value: 'contextual', label: 'Contextual', description: 'With document context' },
        ]}
      />

      <Slider
        label="Chunk Size"
        value={chunking.chunk_size}
        onChange={(value) => updateConfig('chunking', { chunk_size: value })}
        min={128}
        max={2048}
        step={64}
        valueFormatter={(v) => `${v} tokens`}
      />

      <Slider
        label="Chunk Overlap"
        value={chunking.chunk_overlap}
        onChange={(value) => updateConfig('chunking', { chunk_overlap: value })}
        min={0}
        max={200}
        step={10}
        valueFormatter={(v) => `${v} tokens`}
      />

      {chunking.strategy === 'raptor' && (
        <>
          <Slider
            label="RAPTOR Levels"
            value={chunking.raptor_levels || 3}
            onChange={(value) => updateConfig('chunking', { raptor_levels: value })}
            min={1}
            max={5}
            step={1}
          />

          <Select
            label="Clustering Algorithm"
            value={chunking.raptor_clustering || 'gmm'}
            onChange={(value) => updateConfig('chunking', { raptor_clustering: value as 'gmm' | 'kmeans' })}
            options={[
              { value: 'gmm', label: 'GMM', description: 'Gaussian Mixture Model' },
              { value: 'kmeans', label: 'K-Means', description: 'Simpler clustering' },
            ]}
          />
        </>
      )}
    </div>
  )
}
