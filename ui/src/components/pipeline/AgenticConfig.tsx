import { useSettingsStore } from '../../stores/settingsStore'
import { Toggle } from '../common/Toggle'

export function AgenticConfig() {
  const { config, updateConfig } = useSettingsStore()
  const { agentic } = config

  return (
    <div className="space-y-4">
      <Toggle
        checked={agentic.enabled}
        onChange={(checked) => updateConfig('agentic', { enabled: checked })}
        label="Enable Agentic Features"
        description="Self-correction and advanced reasoning capabilities"
      />

      {agentic.enabled && (
        <div className="space-y-3 pl-4 border-l-2 border-accent-light">
          <Toggle
            checked={agentic.self_rag}
            onChange={(checked) => updateConfig('agentic', { self_rag: checked })}
            label="Self-RAG"
            description="Reflect on response quality and regenerate if needed"
          />

          <Toggle
            checked={agentic.crag}
            onChange={(checked) => updateConfig('agentic', { crag: checked })}
            label="CRAG (Corrective RAG)"
            description="Evaluate retrieval quality and correct if poor"
          />

          <Toggle
            checked={agentic.planning}
            onChange={(checked) => updateConfig('agentic', { planning: checked })}
            label="Multi-Step Planning"
            description="Break complex queries into sub-tasks"
          />

          <Toggle
            checked={agentic.web_fallback}
            onChange={(checked) => updateConfig('agentic', { web_fallback: checked })}
            label="Web Search Fallback"
            description="Search the web when retrieval fails"
          />
        </div>
      )}

      {agentic.enabled && (agentic.self_rag || agentic.crag) && (
        <div className="p-3 rounded-lg bg-warning-light text-sm">
          <p className="text-warning font-medium">Note</p>
          <p className="text-warning/80 text-xs mt-1">
            Agentic features may increase latency and API costs due to additional LLM calls for reflection and correction.
          </p>
        </div>
      )}
    </div>
  )
}
