import { useState, useRef, useEffect } from 'react'
import { Send, Paperclip, Settings2 } from 'lucide-react'
import { clsx } from 'clsx'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { chatApi } from '../../api/chat'
import type { Message, Source, PipelineStep, QueryVariations } from '../../types'

interface ChatInputProps {
  collection: string
}

export function ChatInput({ collection }: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { addMessage, updateMessage, appendToMessage, isLoading, setLoading, setCurrentStreamingId } = useChatStore()
  const { config, setConfigPanelOpen } = useSettingsStore()

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()

    const trimmedInput = input.trim()
    if (!trimmedInput || isLoading) return

    // Add user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    }
    addMessage(collection, userMessage)
    setInput('')

    // Add placeholder assistant message
    const assistantId = crypto.randomUUID()
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    }
    addMessage(collection, assistantMessage)
    setLoading(true)
    setCurrentStreamingId(assistantId)

    try {
      // Streaming with full pipeline visibility
      let streamedSources: Source[] = []
      let pipelineSteps: PipelineStep[] = []
      let queryVariations: QueryVariations | undefined
      let thinking = ''
      let provider = ''
      let model = ''
      let confidence = 0

      const { promise } = chatApi.queryStream(
        {
          question: trimmedInput,
          collection,
          // Retrieval settings - critical for answer quality
          top_k: config.retrieval.top_k,
          use_hyde: config.retrieval.use_hyde,
          use_multi_query: config.retrieval.use_multi_query,
          use_reranking: config.reranking.enabled,
          // Generation settings
          temperature: config.generation.temperature,
        },
        (chunk) => {
          try {
            const parsed = JSON.parse(chunk)

            // Handle different event types
            switch (parsed.type) {
              case 'token':
                // Text token from stream
                appendToMessage(collection, assistantId, parsed.token)
                break

              case 'sources':
                // Sources with citation numbers and relevance scores
                streamedSources = parsed.sources.map((s: {
                  citation?: number
                  content: string
                  document_id: string
                  filename?: string
                  score?: number
                  metadata?: Record<string, unknown>
                }) => ({
                  citation: s.citation,
                  content: s.content,
                  source: s.document_id,
                  filename: s.filename,
                  score: s.score ?? 0,  // Preserve score from backend
                  metadata: s.metadata,
                }))
                // Update message with sources immediately for display
                updateMessage(collection, assistantId, { sources: streamedSources })
                break

              case 'queries':
                // Multi-query variations
                queryVariations = {
                  original: parsed.original,
                  variations: parsed.variations,
                }
                updateMessage(collection, assistantId, { queryVariations })
                break

              case 'step':
                // Pipeline step progress
                const step = parsed.step as PipelineStep
                if (step.status === 'completed') {
                  pipelineSteps = [...pipelineSteps, step]
                  updateMessage(collection, assistantId, { pipelineSteps })
                }
                break

              case 'thinking':
                // LLM reasoning (Gemini 3)
                thinking = parsed.thinking
                updateMessage(collection, assistantId, { thinking })
                break

              case 'done':
                // Stream complete with metadata
                provider = parsed.provider || ''
                model = parsed.model || ''
                confidence = parsed.confidence || 0
                updateMessage(collection, assistantId, {
                  sources: streamedSources,
                  pipelineSteps,
                  queryVariations,
                  thinking,
                  provider,
                  model,
                  confidence,
                  isStreaming: false,
                })
                break

              case 'error':
                // Error from stream
                updateMessage(collection, assistantId, {
                  content: `Error: ${parsed.error}`,
                  isStreaming: false,
                })
                break

              default:
                // Legacy format fallback
                if (parsed.token) {
                  appendToMessage(collection, assistantId, parsed.token)
                } else if (parsed.response) {
                  updateMessage(collection, assistantId, {
                    content: parsed.response,
                    sources: parsed.sources as Source[],
                    evaluation: parsed.evaluation,
                    isStreaming: false,
                  })
                }
            }
          } catch {
            // Plain text chunk
            appendToMessage(collection, assistantId, chunk)
          }
        }
      )

      await promise

      // Ensure streaming is marked as done
      updateMessage(collection, assistantId, { isStreaming: false })
    } catch (error) {
      // Fallback to non-streaming
      try {
        const response = await chatApi.query({
          question: trimmedInput,
          collection,
          // Retrieval settings - critical for answer quality
          top_k: config.retrieval.top_k,
          use_hyde: config.retrieval.use_hyde,
          use_multi_query: config.retrieval.use_multi_query,
          use_reranking: config.reranking.enabled,
          // Generation settings
          temperature: config.generation.temperature,
        })

        updateMessage(collection, assistantId, {
          content: response.response,
          sources: response.sources,
          evaluation: response.evaluation,
          isStreaming: false,
        })
      } catch (err) {
        updateMessage(collection, assistantId, {
          content: `Error: ${err instanceof Error ? err.message : 'Failed to get response'}`,
          isStreaming: false,
        })
      }
    } finally {
      setLoading(false)
      setCurrentStreamingId(null)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-border bg-surface p-4">
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            disabled={isLoading}
            className={clsx(
              'w-full resize-none rounded-xl border border-border bg-background',
              'px-4 py-3 pr-24 text-sm text-foreground',
              'placeholder:text-muted-foreground',
              'focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent',
              'disabled:cursor-not-allowed disabled:opacity-50',
              'transition-colors duration-150'
            )}
          />
          <div className="absolute right-2 bottom-2 flex items-center gap-1">
            <button
              type="button"
              onClick={() => setConfigPanelOpen(true)}
              className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
              title="Configure pipeline"
            >
              <Settings2 className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
              title="Attach file"
            >
              <Paperclip className="h-4 w-4" />
            </button>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                input.trim() && !isLoading
                  ? 'bg-accent text-white hover:bg-accent-hover'
                  : 'text-muted-foreground cursor-not-allowed'
              )}
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground text-center mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </form>
    </div>
  )
}
