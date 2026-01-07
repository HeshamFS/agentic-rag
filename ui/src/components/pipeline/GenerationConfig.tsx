import { useSettingsStore } from '../../stores/settingsStore'
import { Select } from '../common/Select'
import { Slider } from '../common/Slider'

// Models from backend generators (claude_generator.py, gemini_generator.py, openai_generator.py, local_generator.py)
const MODEL_OPTIONS: Record<string, { value: string; label: string }[]> = {
  claude: [
    { value: 'claude-sonnet-4-5-20250929', label: 'Claude Sonnet 4.5 (Recommended)' },
    { value: 'claude-opus-4-5-20251101', label: 'Claude Opus 4.5 (Highest Quality)' },
    { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
    { value: 'claude-opus-4-20250514', label: 'Claude Opus 4' },
  ],
  openai: [
    { value: 'gpt-5.2', label: 'GPT-5.2 (Most Capable)' },
    { value: 'gpt-5.1', label: 'GPT-5.1 (High Quality)' },
    { value: 'gpt-5', label: 'GPT-5 (Base)' },
    { value: 'gpt-5-mini', label: 'GPT-5 Mini (Recommended)' },
    { value: 'gpt-5-nano', label: 'GPT-5 Nano (Fast)' },
  ],
  gemini: [
    { value: 'gemini-3-pro-preview', label: 'Gemini 3 Pro (With Thinking)' },
    { value: 'gemini-3-flash-preview', label: 'Gemini 3 Flash (With Thinking)' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (Stable)' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (Recommended)' },
    { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite (Efficient)' },
  ],
  local: [
    { value: 'qwen2.5:7b', label: 'Qwen 2.5 7B (Recommended)' },
    { value: 'llama3.3:70b', label: 'Llama 3.3 70B (High Quality)' },
    { value: 'mistral:7b', label: 'Mistral 7B (Fast)' },
  ],
}

// GPT-5 reasoning effort options
const REASONING_EFFORT_OPTIONS = [
  { value: 'none', label: 'None', description: 'No reasoning, fastest responses' },
  { value: 'low', label: 'Low', description: 'Light reasoning for simple tasks' },
  { value: 'medium', label: 'Medium (Recommended)', description: 'Balanced reasoning' },
  { value: 'high', label: 'High', description: 'Deep reasoning for complex tasks' },
  { value: 'xhigh', label: 'Extra High', description: 'Maximum reasoning depth' },
]

export function GenerationConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { generation } = config

  const availableModels = MODEL_OPTIONS[generation.provider] || []

  // When provider changes, update to first model of that provider
  const handleProviderChange = (provider: string) => {
    const models = MODEL_OPTIONS[provider]
    updateConfig('generation', {
      provider: provider as typeof generation.provider,
      model: models?.[0]?.value || '',
    })
  }

  // GPT-5 uses reasoning effort instead of temperature
  const isOpenAI = generation.provider === 'openai'

  return (
    <div className="space-y-4">
      <Select
        label="LLM Provider"
        value={generation.provider}
        onChange={handleProviderChange}
        options={[
          { value: 'claude', label: 'Anthropic Claude', description: 'Sonnet 4.5 / Opus 4.5 (Default)' },
          { value: 'gemini', label: 'Google Gemini', description: 'Gemini 3 with thinking' },
          { value: 'openai', label: 'OpenAI', description: 'GPT-5 series (no temperature)' },
          { value: 'local', label: 'Local (Ollama)', description: 'Qwen, Llama, Mistral' },
        ]}
      />

      <Select
        label="Model"
        value={generation.model}
        onChange={(value) => updateConfig('generation', { model: value })}
        options={availableModels}
      />

      {/* GPT-5 uses reasoning effort, other providers use temperature */}
      {isOpenAI ? (
        <Select
          label="Reasoning Effort"
          value={generation.reasoning_effort || 'medium'}
          onChange={(value) => updateConfig('generation', { reasoning_effort: value as 'none' | 'low' | 'medium' | 'high' | 'xhigh' })}
          options={REASONING_EFFORT_OPTIONS}
        />
      ) : (
        <Slider
          label="Temperature"
          value={generation.temperature}
          onChange={(value) => updateConfig('generation', { temperature: value })}
          min={0}
          max={1}
          step={0.1}
          valueFormatter={(v) => v.toFixed(1)}
        />
      )}

      <Slider
        label="Max Output Tokens"
        value={generation.max_tokens}
        onChange={(value) => updateConfig('generation', { max_tokens: value })}
        min={256}
        max={8192}
        step={256}
        valueFormatter={(v) => v.toLocaleString()}
      />

      {/* Note about GPT-5 parameters */}
      {isOpenAI && (
        <p className="text-xs text-muted">
          GPT-5 models use reasoning effort instead of temperature. Higher effort = deeper thinking.
        </p>
      )}
    </div>
  )
}
