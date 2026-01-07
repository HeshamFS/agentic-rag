import { Settings, HelpCircle, Database } from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSettingsStore } from '../../stores/settingsStore'

export function Header() {
  const { activeCollection } = useCollectionStore()
  const { viewMode, setConfigPanelOpen } = useSettingsStore()

  const getTitle = () => {
    switch (viewMode) {
      case 'chat':
        return activeCollection ? `Chat with ${activeCollection}` : 'Select a Collection'
      case 'collections':
        return 'Collections'
      case 'upload':
        return 'Upload Documents'
      case 'config':
        return 'Pipeline Configuration'
      case 'evaluation':
        return 'Evaluation Metrics'
      default:
        return 'RAG Optimizer'
    }
  }

  return (
    <header className="h-14 border-b border-border bg-surface flex items-center justify-between px-6">
      {/* Left side - Title */}
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold text-foreground">{getTitle()}</h1>
        {viewMode === 'chat' && activeCollection && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-accent-light text-accent text-xs font-medium">
            <Database className="h-3 w-3" />
            {activeCollection}
          </span>
        )}
      </div>

      {/* Right side - Actions */}
      <div className="flex items-center gap-2">
        {viewMode === 'chat' && (
          <button
            onClick={() => setConfigPanelOpen(true)}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
          >
            <Settings className="h-4 w-4" />
            <span className="hidden sm:inline">Configure</span>
          </button>
        )}
        <button
          className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
          title="Help"
        >
          <HelpCircle className="h-5 w-5" />
        </button>
      </div>
    </header>
  )
}
