import { useState } from 'react'
import { Plus, Search, Database, FolderOpen } from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { CollectionCard } from './CollectionCard'
import { CreateCollectionModal } from './CreateCollectionModal'

export function CollectionManager() {
  const { collections, isLoading } = useCollectionStore()
  const { setViewMode } = useSettingsStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)

  const filteredCollections = collections.filter((c) =>
    c && c.name && c.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-foreground">Collections</h1>
            <p className="text-muted text-sm mt-1">
              Manage your document collections for RAG queries
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary"
          >
            <Plus className="h-4 w-4" />
            New Collection
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
          <input
            type="text"
            placeholder="Search collections..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input pl-10"
          />
        </div>

        {/* Collections grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="card p-6 animate-pulse">
                <div className="h-10 w-10 bg-surface-hover rounded-lg mb-4" />
                <div className="h-5 w-32 bg-surface-hover rounded mb-2" />
                <div className="h-4 w-24 bg-surface-hover rounded" />
              </div>
            ))}
          </div>
        ) : filteredCollections.length === 0 ? (
          <div className="text-center py-12">
            {collections.length === 0 ? (
              <>
                <div className="w-16 h-16 rounded-2xl bg-surface-hover flex items-center justify-center mx-auto mb-4">
                  <FolderOpen className="h-8 w-8 text-muted" />
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  No collections yet
                </h3>
                <p className="text-muted text-sm mb-6 max-w-md mx-auto">
                  Create your first collection to start uploading documents and chatting with your data.
                </p>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="btn btn-primary"
                >
                  <Plus className="h-4 w-4" />
                  Create Collection
                </button>
              </>
            ) : (
              <>
                <Database className="h-12 w-12 text-muted mx-auto mb-4" />
                <p className="text-muted">No collections matching "{searchQuery}"</p>
              </>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredCollections.map((collection) => (
              <CollectionCard
                key={collection.name}
                collection={collection}
                onChat={() => {
                  useCollectionStore.getState().setActiveCollection(collection.name)
                  setViewMode('chat')
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Create modal */}
      <CreateCollectionModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />
    </div>
  )
}
