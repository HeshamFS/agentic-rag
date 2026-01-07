import { useRef, useEffect, useState } from 'react'
import { MessageSquare } from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { useChatStore } from '../../stores/chatStore'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { DocumentSidebar } from './DocumentSidebar'

export function ChatContainer() {
  const { activeCollection } = useCollectionStore()
  const { getMessages } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showDocuments, setShowDocuments] = useState(false)

  const messages = activeCollection ? getMessages(activeCollection) : []

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (!activeCollection) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 rounded-2xl bg-surface-hover flex items-center justify-center mx-auto mb-4">
            <MessageSquare className="h-8 w-8 text-muted" />
          </div>
          <h2 className="text-xl font-semibold text-foreground mb-2">
            Select a Collection
          </h2>
          <p className="text-muted text-sm">
            Choose a document collection from the sidebar to start chatting.
            Upload documents first if you haven't created a collection yet.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full relative">
      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center p-8">
              <div className="text-center max-w-lg">
                <div className="w-16 h-16 rounded-2xl bg-accent-light flex items-center justify-center mx-auto mb-4">
                  <MessageSquare className="h-8 w-8 text-accent" />
                </div>
                <h2 className="text-xl font-semibold text-foreground mb-2">
                  Talk to Your Documents
                </h2>
                <p className="text-muted text-sm mb-6">
                  Ask questions about your documents in <strong>{activeCollection}</strong>.
                  The AI will search through your content and provide accurate answers with sources.
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  <SuggestionChip text="What are the main topics covered?" />
                  <SuggestionChip text="Summarize the key findings" />
                  <SuggestionChip text="What are the conclusions?" />
                </div>
              </div>
            </div>
          ) : (
            <MessageList messages={messages} />
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <ChatInput collection={activeCollection} />
      </div>

      {/* Document sidebar */}
      <DocumentSidebar
        collection={activeCollection}
        isOpen={showDocuments}
        onToggle={() => setShowDocuments(!showDocuments)}
      />
    </div>
  )
}

function SuggestionChip({ text }: { text: string }) {
  const { activeCollection } = useCollectionStore()
  const { addMessage, setLoading } = useChatStore()

  const handleClick = async () => {
    if (!activeCollection) return

    // Add user message
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user' as const,
      content: text,
      timestamp: new Date(),
    }
    addMessage(activeCollection, userMessage)
    setLoading(true)

    // TODO: Trigger actual query
  }

  return (
    <button
      onClick={handleClick}
      className="px-3 py-1.5 rounded-full border border-border text-sm text-muted hover:text-foreground hover:border-accent hover:bg-accent-light transition-colors"
    >
      {text}
    </button>
  )
}
