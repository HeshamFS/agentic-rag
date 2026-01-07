import { useSettingsStore } from '../../stores/settingsStore'
import { Select } from '../common/Select'
import { Slider } from '../common/Slider'

export function CachingConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { caching } = config

  const isEnabled = caching.backend !== 'disabled'

  return (
    <div className="space-y-4">
      <Select
        label="Cache Backend"
        value={caching.backend}
        onChange={(value) => updateConfig('caching', { backend: value as typeof caching.backend })}
        options={[
          { value: 'memory', label: 'In-Memory', description: 'Fast, single instance' },
          { value: 'redis', label: 'Redis', description: 'Distributed, persistent' },
          { value: 'disabled', label: 'Disabled', description: 'No caching' },
        ]}
      />

      {isEnabled && (
        <>
          <Slider
            label="Similarity Threshold"
            value={caching.similarity_threshold}
            onChange={(value) => updateConfig('caching', { similarity_threshold: value })}
            min={0.8}
            max={1.0}
            step={0.01}
            valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />

          <Slider
            label="TTL (Time to Live)"
            value={caching.ttl_seconds}
            onChange={(value) => updateConfig('caching', { ttl_seconds: value })}
            min={60}
            max={86400}
            step={60}
            valueFormatter={(v) => {
              if (v < 3600) return `${Math.floor(v / 60)} min`
              return `${Math.floor(v / 3600)} hr`
            }}
          />

          <div className="p-3 rounded-lg bg-success-light text-sm">
            <p className="text-success font-medium">Cache Benefits</p>
            <p className="text-success/80 text-xs mt-1">
              Similar queries (≥{(caching.similarity_threshold * 100).toFixed(0)}% match) return instant cached responses, saving API costs and latency.
            </p>
          </div>
        </>
      )}
    </div>
  )
}
