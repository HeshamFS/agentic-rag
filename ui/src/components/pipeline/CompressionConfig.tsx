import { useSettingsStore } from '../../stores/settingsStore'
import { Select } from '../common/Select'
import { Slider } from '../common/Slider'
import { Toggle } from '../common/Toggle'

export function CompressionConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { compression } = config

  return (
    <div className="space-y-4">
      <Toggle
        checked={compression.enabled}
        onChange={(checked) => updateConfig('compression', { enabled: checked })}
        label="Enable Context Compression"
        description="Reduce tokens while preserving key information"
      />

      {compression.enabled && (
        <>
          <Select
            label="Compression Method"
            value={compression.method}
            onChange={(value) => updateConfig('compression', { method: value as typeof compression.method })}
            options={[
              { value: 'extractive', label: 'Extractive', description: 'Select important sentences' },
              { value: 'longllmlingua', label: 'LongLLMLingua', description: 'Perplexity-based (best quality)' },
              { value: 'sentence', label: 'Sentence-level', description: 'Score each sentence' },
            ]}
          />

          <Slider
            label="Compression Ratio"
            value={compression.ratio}
            onChange={(value) => updateConfig('compression', { ratio: value })}
            min={0.1}
            max={0.9}
            step={0.1}
            valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />

          <div className="p-3 rounded-lg bg-accent-light text-sm">
            <p className="text-accent font-medium">Token Savings Estimate</p>
            <p className="text-accent/80 text-xs mt-1">
              With {(compression.ratio * 100).toFixed(0)}% compression, you'll use ~{((1 - compression.ratio) * 100).toFixed(0)}% of original context tokens
            </p>
          </div>
        </>
      )}
    </div>
  )
}
