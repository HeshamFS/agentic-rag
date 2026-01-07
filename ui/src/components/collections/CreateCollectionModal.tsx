import { useState } from 'react'
import { Database } from 'lucide-react'
import { Modal } from '../common/Modal'
import { Button } from '../common/Button'
import { useCollectionStore } from '../../stores/collectionStore'
import { collectionsApi } from '../../api/collections'

interface CreateCollectionModalProps {
  isOpen: boolean
  onClose: () => void
}

export function CreateCollectionModal({ isOpen, onClose }: CreateCollectionModalProps) {
  const [name, setName] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState('')
  const { addCollection, setActiveCollection } = useCollectionStore()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const trimmedName = name.trim()
    if (!trimmedName) {
      setError('Collection name is required')
      return
    }

    // Validate name format
    if (!/^[a-zA-Z0-9_-]+$/.test(trimmedName)) {
      setError('Name can only contain letters, numbers, hyphens, and underscores')
      return
    }

    setIsCreating(true)
    setError('')

    try {
      await collectionsApi.create(trimmedName)
      addCollection({
        name: trimmedName,
        document_count: 0,
        chunk_count: 0,
      })
      setActiveCollection(trimmedName)
      setName('')
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create collection')
    } finally {
      setIsCreating(false)
    }
  }

  const handleClose = () => {
    setName('')
    setError('')
    onClose()
  }

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Create Collection" size="sm">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 rounded-2xl bg-accent-light flex items-center justify-center">
            <Database className="h-8 w-8 text-accent" />
          </div>
        </div>

        <div className="space-y-2">
          <label htmlFor="name" className="block text-sm font-medium text-foreground">
            Collection Name
          </label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setError('')
            }}
            placeholder="e.g., research-papers"
            className="input"
            autoFocus
          />
          {error && (
            <p className="text-sm text-error">{error}</p>
          )}
          <p className="text-xs text-muted">
            Use letters, numbers, hyphens, and underscores only.
          </p>
        </div>

        <div className="flex gap-3 pt-2">
          <Button
            type="button"
            variant="secondary"
            onClick={handleClose}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            type="submit"
            variant="primary"
            isLoading={isCreating}
            className="flex-1"
          >
            Create
          </Button>
        </div>
      </form>
    </Modal>
  )
}
