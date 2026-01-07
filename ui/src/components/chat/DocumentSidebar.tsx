import { useState, useEffect } from 'react'
import { FileText, ChevronRight, ChevronLeft, Loader2, Layers } from 'lucide-react'
import { clsx } from 'clsx'
import { collectionsApi } from '../../api/collections'
import { useCollectionStore } from '../../stores/collectionStore'

interface Document {
  filename: string
  file_id: string
  source_file: string
  chunk_count: number
}

interface DocumentSidebarProps {
  collection: string
  isOpen: boolean
  onToggle: () => void
}

export function DocumentSidebar({ collection, isOpen, onToggle }: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [totalChunks, setTotalChunks] = useState(0)

  const { getCachedDocuments, setCachedDocuments } = useCollectionStore()

  // Fetch documents when collection changes or sidebar opens - with caching
  useEffect(() => {
    if (!isOpen || !collection) return

    const fetchDocuments = async () => {
      // Check cache first
      const cached = getCachedDocuments(collection)
      if (cached) {
        setDocuments(cached.documents)
        setTotalChunks(cached.total_chunks)
        return
      }

      setIsLoading(true)
      setError(null)

      try {
        const response = await collectionsApi.getDocuments(collection)
        const docs = response.documents || []
        const chunks = response.total_chunks || 0

        setDocuments(docs)
        setTotalChunks(chunks)

        // Cache the results
        setCachedDocuments(collection, docs, chunks)
      } catch (err) {
        console.error('Failed to fetch documents:', err)
        setError('Failed to load documents')
        setDocuments([])
      } finally {
        setIsLoading(false)
      }
    }

    fetchDocuments()
  }, [collection, isOpen, getCachedDocuments, setCachedDocuments])

  return (
    <>
      {/* Toggle button (always visible) */}
      <button
        onClick={onToggle}
        className={clsx(
          'absolute right-0 top-4 z-10 p-2 rounded-l-lg border border-r-0 border-border',
          'bg-surface hover:bg-surface-hover transition-colors',
          'flex items-center gap-1 text-sm text-muted hover:text-foreground',
          isOpen && 'hidden'
        )}
        title="Show documents"
      >
        <FileText className="h-4 w-4" />
        <ChevronLeft className="h-3 w-3" />
      </button>

      {/* Sidebar panel */}
      <div
        className={clsx(
          'border-l border-border bg-surface transition-all duration-200 overflow-hidden',
          isOpen ? 'w-72' : 'w-0'
        )}
      >
        {isOpen && (
          <div className="w-72 h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-muted" />
                <span className="font-medium text-sm text-foreground">Documents</span>
              </div>
              <button
                onClick={onToggle}
                className="p-1 rounded-md text-muted hover:text-foreground hover:bg-surface-hover"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>

            {/* Stats */}
            <div className="px-4 py-2 border-b border-border bg-surface-hover/50">
              <div className="flex items-center justify-between text-xs text-muted">
                <span>{documents.length} documents</span>
                <span className="flex items-center gap-1">
                  <Layers className="h-3 w-3" />
                  {totalChunks.toLocaleString()} chunks
                </span>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-2">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-5 w-5 text-muted animate-spin" />
                </div>
              ) : error ? (
                <div className="text-center py-8">
                  <p className="text-sm text-error">{error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="mt-2 text-xs text-muted hover:text-foreground"
                  >
                    Retry
                  </button>
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-8">
                  <FileText className="h-8 w-8 text-muted mx-auto mb-2" />
                  <p className="text-sm text-muted">No documents yet</p>
                  <p className="text-xs text-muted mt-1">Upload files to get started</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {documents.map((doc) => (
                    <DocumentItem key={doc.file_id || doc.filename} document={doc} />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  )
}

function DocumentItem({ document }: { document: Document }) {
  const [isExpanded, setIsExpanded] = useState(false)

  // Get file extension for icon styling
  const ext = document.filename.split('.').pop()?.toLowerCase() || ''
  const extColors: Record<string, string> = {
    pdf: 'text-red-500',
    docx: 'text-blue-500',
    doc: 'text-blue-500',
    txt: 'text-gray-500',
    md: 'text-purple-500',
  }

  return (
    <div
      className={clsx(
        'rounded-lg border border-border bg-surface p-2',
        'hover:border-accent/50 transition-colors cursor-pointer'
      )}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div className="flex items-start gap-2">
        <FileText className={clsx('h-4 w-4 mt-0.5 flex-shrink-0', extColors[ext] || 'text-muted')} />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground truncate" title={document.filename}>
            {document.filename}
          </p>
          <p className="text-xs text-muted">
            {document.chunk_count} chunks
          </p>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted space-y-1">
            <p><span className="text-foreground">File ID:</span> {document.file_id}</p>
            <p><span className="text-foreground">Source:</span> {document.source_file}</p>
          </div>
        </div>
      )}
    </div>
  )
}
