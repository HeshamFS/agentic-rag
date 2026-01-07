import { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import {
  MessageSquare,
  FolderOpen,
  Upload,
  Settings,
  BarChart3,
  Plus,
  ChevronLeft,
  ChevronRight,
  Database,
} from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { collectionsApi } from '../../api/collections'
import type { ViewMode } from '../../types'

interface NavItemProps {
  icon: React.ElementType
  label: string
  active?: boolean
  collapsed?: boolean
  onClick?: () => void
  badge?: number
}

function NavItem({ icon: Icon, label, active, collapsed, onClick, badge }: NavItemProps) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
        active
          ? 'bg-accent-light text-accent'
          : 'text-muted hover:text-foreground hover:bg-surface-hover'
      )}
    >
      <Icon className="h-5 w-5 flex-shrink-0" />
      {!collapsed && (
        <>
          <span className="flex-1 text-left truncate">{label}</span>
          {badge !== undefined && badge > 0 && (
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-accent-muted text-accent">
              {badge}
            </span>
          )}
        </>
      )}
    </button>
  )
}

interface CollectionItemProps {
  name: string
  documentCount: number
  active?: boolean
  collapsed?: boolean
  onClick?: () => void
}

function CollectionItem({ name, documentCount, active, collapsed, onClick }: CollectionItemProps) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
        active
          ? 'bg-accent-light text-accent font-medium'
          : 'text-muted hover:text-foreground hover:bg-surface-hover'
      )}
    >
      <Database className="h-4 w-4 flex-shrink-0" />
      {!collapsed && (
        <>
          <span className="flex-1 text-left truncate">{name}</span>
          <span className="text-xs text-muted-foreground">{documentCount}</span>
        </>
      )}
    </button>
  )
}

export function Sidebar() {
  const { sidebarOpen, setSidebarOpen, viewMode, setViewMode } = useSettingsStore()
  const { collections, activeCollection, setActiveCollection, setCollections, setLoading } = useCollectionStore()
  const [isHovered, setIsHovered] = useState(false)
  const [hasFetched, setHasFetched] = useState(false)

  // Fetch collections ONCE on mount - no dependencies to prevent refetching
  useEffect(() => {
    // Skip if already fetched
    if (hasFetched) return

    const fetchCollections = async () => {
      setLoading(true)
      setHasFetched(true)

      try {
        // Step 1: Get collection names (API returns { collections: string[] })
        const response = await collectionsApi.list()
        const names = response.collections || []

        // Step 2: Convert names to Collection objects with initial values
        const collectionObjects = names
          .filter((name: string) => name && name.trim() !== '')
          .map((name: string) => ({
            name,
            document_count: 0,
            chunk_count: 0,
          }))
        setCollections(collectionObjects)

        // Auto-select first collection if none selected
        const store = useCollectionStore.getState()
        if (!store.activeCollection && collectionObjects.length > 0) {
          setActiveCollection(collectionObjects[0].name)
        }

        // Step 3: Fetch documents for each collection (includes chunk counts)
        // This populates both document_count and chunk_count, and caches the results
        // Use Promise.all for parallel fetching
        const docsPromises = names
          .filter((n: string) => n && n.trim() !== '')
          .map(async (name: string) => {
            try {
              const response = await collectionsApi.getDocuments(name)
              const docs = response.documents || []
              const chunks = response.total_chunks || 0

              // Cache the documents
              useCollectionStore.getState().setCachedDocuments(name, docs, chunks)

              return { name, document_count: docs.length, chunk_count: chunks }
            } catch {
              // Fallback to just chunk count from info endpoint
              try {
                const info = await collectionsApi.getInfo(name)
                return { name, document_count: 0, chunk_count: info?.chunk_count || 0 }
              } catch {
                return { name, document_count: 0, chunk_count: 0 }
              }
            }
          })

        const results = await Promise.all(docsPromises)
        results.forEach(({ name, document_count, chunk_count }) => {
          useCollectionStore.getState().updateCollection(name, { document_count, chunk_count })
        })
      } catch (error) {
        console.error('Failed to fetch collections:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchCollections()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasFetched])

  const collapsed = !sidebarOpen && !isHovered

  const handleViewChange = (mode: ViewMode) => {
    setViewMode(mode)
  }

  const handleCollectionSelect = (name: string) => {
    setActiveCollection(name)
    setViewMode('chat')
  }

  return (
    <aside
      className={clsx(
        'flex flex-col bg-sidebar border-r border-border transition-all duration-200',
        collapsed ? 'w-16' : 'w-64'
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 h-14 border-b border-border">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
              <MessageSquare className="h-4 w-4 text-white" />
            </div>
            <span className="font-semibold text-foreground">RAG Optimizer</span>
          </div>
        )}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-1.5 rounded-md text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
        >
          {sidebarOpen ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        <NavItem
          icon={MessageSquare}
          label="Chat"
          active={viewMode === 'chat'}
          collapsed={collapsed}
          onClick={() => handleViewChange('chat')}
        />
        <NavItem
          icon={FolderOpen}
          label="Collections"
          active={viewMode === 'collections'}
          collapsed={collapsed}
          onClick={() => handleViewChange('collections')}
        />
        <NavItem
          icon={Upload}
          label="Upload"
          active={viewMode === 'upload'}
          collapsed={collapsed}
          onClick={() => handleViewChange('upload')}
        />

        {/* Collections list */}
        {!collapsed && collections.filter(c => c && c.name).length > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex items-center justify-between px-3 mb-2">
              <span className="text-xs font-medium text-muted uppercase tracking-wider">
                Collections
              </span>
              <button
                onClick={() => handleViewChange('collections')}
                className="p-1 rounded-md text-muted hover:text-foreground hover:bg-surface-hover"
              >
                <Plus className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="space-y-0.5">
              {collections.filter(c => c && c.name).slice(0, 5).map((collection) => (
                <CollectionItem
                  key={collection.name}
                  name={collection.name}
                  documentCount={collection.document_count || 0}
                  active={activeCollection === collection.name && viewMode === 'chat'}
                  collapsed={collapsed}
                  onClick={() => handleCollectionSelect(collection.name)}
                />
              ))}
              {collections.filter(c => c && c.name).length > 5 && (
                <button
                  onClick={() => handleViewChange('collections')}
                  className="w-full px-3 py-1.5 text-xs text-muted hover:text-foreground"
                >
                  View all ({collections.filter(c => c && c.name).length})
                </button>
              )}
            </div>
          </div>
        )}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-border space-y-1">
        <NavItem
          icon={Settings}
          label="Pipeline Config"
          active={viewMode === 'config'}
          collapsed={collapsed}
          onClick={() => handleViewChange('config')}
        />
        <NavItem
          icon={BarChart3}
          label="Evaluation"
          active={viewMode === 'evaluation'}
          collapsed={collapsed}
          onClick={() => handleViewChange('evaluation')}
        />
      </div>
    </aside>
  )
}
