import { create } from 'zustand'
import type { Message } from '../types'

interface ChatState {
  messages: Record<string, Message[]>  // Keyed by collection name
  isLoading: boolean
  currentStreamingId: string | null

  // Actions
  addMessage: (collection: string, message: Message) => void
  updateMessage: (collection: string, id: string, updates: Partial<Message>) => void
  appendToMessage: (collection: string, id: string, content: string) => void
  clearMessages: (collection: string) => void
  setLoading: (loading: boolean) => void
  setCurrentStreamingId: (id: string | null) => void
  getMessages: (collection: string) => Message[]
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: {},
  isLoading: false,
  currentStreamingId: null,

  addMessage: (collection, message) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [collection]: [...(state.messages[collection] || []), message],
      },
    })),

  updateMessage: (collection, id, updates) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [collection]: (state.messages[collection] || []).map((m) =>
          m.id === id ? { ...m, ...updates } : m
        ),
      },
    })),

  appendToMessage: (collection, id, content) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [collection]: (state.messages[collection] || []).map((m) =>
          m.id === id ? { ...m, content: m.content + content } : m
        ),
      },
    })),

  clearMessages: (collection) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [collection]: [],
      },
    })),

  setLoading: (isLoading) => set({ isLoading }),

  setCurrentStreamingId: (currentStreamingId) => set({ currentStreamingId }),

  getMessages: (collection) => get().messages[collection] || [],
}))
