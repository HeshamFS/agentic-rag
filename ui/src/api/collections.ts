import { api } from './client'

// Backend returns just collection names as strings
export interface CollectionListResponse {
  collections: string[]
}

// GET /collections/{name} returns this
export interface CollectionInfo {
  name: string
  chunk_count: number
  created_at: string | null
  metadata: Record<string, unknown>
}

// GET /collections/{name}/documents returns this
export interface DocumentListResponse {
  documents: Array<{
    filename: string
    file_id: string
    source_file: string
    chunk_count: number
  }>
  collection: string
  total_chunks: number
}

export const collectionsApi = {
  // Get list of collection names
  list: () =>
    api.get<CollectionListResponse>('/collections'),

  // Get collection info (chunk count, etc)
  getInfo: (name: string) =>
    api.get<CollectionInfo>(`/collections/${name}`),

  // Get documents in a collection
  getDocuments: (name: string) =>
    api.get<DocumentListResponse>(`/collections/${name}/documents`),

  // Create a new collection
  // Note: In the backend, collections are created implicitly on first upload
  // This is a UI convenience method - it just adds to local state
  // The actual collection will be created when documents are uploaded
  create: async (name: string): Promise<{ success: boolean; name: string }> => {
    // Validate name exists (will create collection on first upload)
    return { success: true, name }
  },

  // Delete collection
  delete: (name: string) =>
    api.delete<{ success: boolean; deleted: string }>(`/collections/${name}`),
}
