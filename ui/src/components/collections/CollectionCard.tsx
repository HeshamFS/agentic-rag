import { useState } from 'react'
import { Database, MessageSquare, Upload, Trash2, MoreVertical, FileText, Layers } from 'lucide-react'
import { clsx } from 'clsx'
import type { Collection } from '../../types'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { collectionsApi } from '../../api/collections'

interface CollectionCardProps {
  collection: Collection
  onChat: () => void
}

export function CollectionCard({ collection, onChat }: CollectionCardProps) {
  const [showMenu, setShowMenu] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const { removeCollection } = useCollectionStore()
  const { setViewMode } = useSettingsStore()

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete "${collection.name}"? This action cannot be undone.`)) {
      return
    }

    setIsDeleting(true)
    try {
      await collectionsApi.delete(collection.name)
      removeCollection(collection.name)
    } catch (error) {
      console.error('Failed to delete collection:', error)
      alert('Failed to delete collection')
    } finally {
      setIsDeleting(false)
      setShowMenu(false)
    }
  }

  const handleUpload = () => {
    useCollectionStore.getState().setActiveCollection(collection.name)
    setViewMode('upload')
  }

  return (
    <div className="card group hover:shadow-medium transition-shadow">
      <div className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="w-12 h-12 rounded-xl bg-accent-light flex items-center justify-center">
            <Database className="h-6 w-6 text-accent" />
          </div>
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-1.5 rounded-md text-muted hover:text-foreground hover:bg-surface-hover transition-colors opacity-0 group-hover:opacity-100"
            >
              <MoreVertical className="h-4 w-4" />
            </button>
            {showMenu && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowMenu(false)}
                />
                <div className="absolute right-0 top-full mt-1 w-36 bg-surface border border-border rounded-lg shadow-dropdown z-20 py-1">
                  <button
                    onClick={handleDelete}
                    disabled={isDeleting}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-error hover:bg-error-light transition-colors"
                  >
                    <Trash2 className="h-4 w-4" />
                    {isDeleting ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Name */}
        <h3 className="text-lg font-semibold text-foreground mb-1 truncate">
          {collection.name}
        </h3>

        {/* Stats */}
        <div className="flex items-center gap-4 text-sm text-muted mb-4">
          <span className="flex items-center gap-1.5">
            <FileText className="h-4 w-4" />
            {collection.document_count} docs
          </span>
          <span className="flex items-center gap-1.5">
            <Layers className="h-4 w-4" />
            {collection.chunk_count.toLocaleString()} chunks
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={onChat}
            className={clsx(
              'flex-1 inline-flex items-center justify-center gap-2',
              'px-3 py-2 rounded-lg text-sm font-medium',
              'bg-accent text-white hover:bg-accent-hover transition-colors'
            )}
          >
            <MessageSquare className="h-4 w-4" />
            Chat
          </button>
          <button
            onClick={handleUpload}
            className={clsx(
              'inline-flex items-center justify-center',
              'px-3 py-2 rounded-lg text-sm font-medium',
              'border border-border text-muted hover:text-foreground hover:bg-surface-hover transition-colors'
            )}
          >
            <Upload className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
