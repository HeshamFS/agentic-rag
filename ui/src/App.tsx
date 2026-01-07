import { useEffect } from 'react'
import { Layout } from './components/layout'
import { useSettingsStore } from './stores/settingsStore'
import { ChatContainer } from './components/chat/ChatContainer'
import { CollectionManager } from './components/collections/CollectionManager'
import { UploadZone } from './components/upload/UploadZone'
import { PipelineConfig } from './components/pipeline/PipelineConfig'
import { EvaluationPanel } from './components/evaluation/EvaluationPanel'

export default function App() {
  const { viewMode, configPanelOpen, setConfigPanelOpen, fetchFromBackend } = useSettingsStore()

  // Fetch config from backend on initial load
  useEffect(() => {
    fetchFromBackend()
  }, [fetchFromBackend])

  const renderContent = () => {
    switch (viewMode) {
      case 'chat':
        return <ChatContainer />
      case 'collections':
        return <CollectionManager />
      case 'upload':
        return <UploadZone />
      case 'config':
        return <PipelineConfig />
      case 'evaluation':
        return <EvaluationPanel />
      default:
        return <ChatContainer />
    }
  }

  return (
    <Layout>
      {renderContent()}

      {/* Pipeline config overlay - accessible from any view */}
      {configPanelOpen && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setConfigPanelOpen(false)}
          />
          <div className="absolute right-0 top-0 h-full w-full max-w-md bg-surface shadow-dropdown overflow-y-auto animate-slide-down">
            <PipelineConfig onClose={() => setConfigPanelOpen(false)} />
          </div>
        </div>
      )}
    </Layout>
  )
}
