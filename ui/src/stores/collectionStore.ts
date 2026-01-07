import { create } from 'zustand'
import type { Collection } from '../types'

// Document info from the API
interface DocumentInfo {
  filename: string
  file_id: string
  source_file: string
  chunk_count: number
}

// Cache entry with timestamp
interface DocumentsCache {
  documents: DocumentInfo[]
  total_chunks: number
  fetchedAt: number
}

interface CollectionState {
  collections: Collection[]
  activeCollection: string | null
  isLoading: boolean
  error: string | null

  // Documents cache - keyed by collection name
  documentsCache: Record<string, DocumentsCache>

  // Actions
  setCollections: (collections: Collection[]) => void
  // Set collections from names (API returns string[])
  setCollectionsFromNames: (names: string[]) => void
  setActiveCollection: (name: string | null) => void
  addCollection: (collection: Collection) => void
  removeCollection: (name: string) => void
  updateCollection: (name: string, updates: Partial<Collection>) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  // Documents cache actions
  setCachedDocuments: (collection: string, documents: DocumentInfo[], total_chunks: number) => void
  getCachedDocuments: (collection: string) => DocumentsCache | null
  clearDocumentsCache: (collection?: string) => void
}

// Cache TTL: 5 minutes
const CACHE_TTL_MS = 5 * 60 * 1000

export const useCollectionStore = create<CollectionState>((set, get) => ({
  collections: [],
  activeCollection: null,
  isLoading: false,
  error: null,
  documentsCache: {},

  setCollections: (collections) => set({ collections }),

  // Convert string names from API to Collection objects (with default values)
  // Detailed info will be fetched separately via collectionsApi.getInfo()
  setCollectionsFromNames: (names) =>
    set({
      collections: names.filter(name => name && name.trim() !== '').map((name) => ({
        name,
        document_count: 0,
        chunk_count: 0,
      })),
    }),

  setActiveCollection: (name) => set({ activeCollection: name }),

  addCollection: (collection) =>
    set((state) => ({
      collections: [...state.collections, collection],
    })),

  removeCollection: (name) =>
    set((state) => ({
      collections: state.collections.filter((c) => c.name !== name),
      activeCollection: state.activeCollection === name ? null : state.activeCollection,
      // Also clear cache for removed collection
      documentsCache: Object.fromEntries(
        Object.entries(state.documentsCache).filter(([key]) => key !== name)
      ),
    })),

  updateCollection: (name, updates) =>
    set((state) => ({
      collections: state.collections.map((c) =>
        c.name === name ? { ...c, ...updates } : c
      ),
    })),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error }),

  // Cache documents for a collection
  setCachedDocuments: (collection, documents, total_chunks) =>
    set((state) => ({
      documentsCache: {
        ...state.documentsCache,
        [collection]: {
          documents,
          total_chunks,
          fetchedAt: Date.now(),
        },
      },
      // Also update collection document_count
      collections: state.collections.map((c) =>
        c.name === collection
          ? { ...c, document_count: documents.length, chunk_count: total_chunks }
          : c
      ),
    })),

  // Get cached documents if not expired
  getCachedDocuments: (collection) => {
    const cache = get().documentsCache[collection]
    if (!cache) return null

    // Check if cache is expired
    if (Date.now() - cache.fetchedAt > CACHE_TTL_MS) {
      return null // Expired
    }

    return cache
  },

  // Clear cache for a specific collection or all
  clearDocumentsCache: (collection) =>
    set((state) => ({
      documentsCache: collection
        ? Object.fromEntries(
            Object.entries(state.documentsCache).filter(([key]) => key !== collection)
          )
        : {},
    })),
}))
