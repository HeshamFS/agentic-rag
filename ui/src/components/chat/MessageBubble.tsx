import { useState } from 'react'
import { clsx } from 'clsx'
import ReactMarkdown from 'react-markdown'
import {
  User,
  Bot,
  Copy,
  Check,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Search,
  FileText,
  Zap,
  Brain,
  Clock,
} from 'lucide-react'
import type { Message } from '../../types'
import { SourceCards } from './SourceCards'
import { EvaluationBadge } from './EvaluationBadge'
import { LoadingDots } from '../common/LoadingSpinner'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const [showSources, setShowSources] = useState(true) // Default open
  const [showPipeline, setShowPipeline] = useState(false)
  const [showThinking, setShowThinking] = useState(false)

  const isUser = message.role === 'user'
  const hasSources = message.sources && message.sources.length > 0
  const hasQueryVariations = message.queryVariations && message.queryVariations.variations.length > 1
  const hasPipelineSteps = message.pipelineSteps && message.pipelineSteps.length > 0
  const hasThinking = message.thinking && message.thinking.length > 0

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Format step name for display
  const formatStepName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
  }

  // Get step icon
  const getStepIcon = (name: string) => {
    if (name.includes('query') || name.includes('expansion')) return Search
    if (name.includes('retriev') || name.includes('embed')) return FileText
    if (name.includes('rerank')) return Zap
    if (name.includes('generat')) return Sparkles
    return Clock
  }

  return (
    <div
      className={clsx(
        'group animate-slide-up',
        isUser ? 'flex justify-end' : ''
      )}
    >
      <div
        className={clsx(
          'max-w-full',
          isUser ? 'max-w-[85%]' : ''
        )}
      >
        {/* Avatar and role */}
        <div className={clsx('flex items-center gap-2 mb-2', isUser && 'justify-end')}>
          <div
            className={clsx(
              'w-6 h-6 rounded-full flex items-center justify-center',
              isUser ? 'bg-accent' : 'bg-surface-hover order-first'
            )}
          >
            {isUser ? (
              <User className="h-3.5 w-3.5 text-white" />
            ) : (
              <Bot className="h-3.5 w-3.5 text-muted" />
            )}
          </div>
          <span className="text-xs font-medium text-muted">
            {isUser ? 'You' : 'Assistant'}
          </span>
          {!isUser && message.provider && (
            <span className="text-xs text-muted-foreground bg-surface-hover px-1.5 py-0.5 rounded">
              {message.model || message.provider}
            </span>
          )}
          {!isUser && message.evaluation?.ragas_score && (
            <EvaluationBadge score={message.evaluation.ragas_score} />
          )}
          {!isUser && message.confidence !== undefined && message.confidence > 0 && (
            <span className="text-xs text-muted-foreground">
              {(message.confidence * 100).toFixed(0)}% confidence
            </span>
          )}
        </div>

        {/* Query Variations (Multi-Query) */}
        {!isUser && hasQueryVariations && (
          <div className="mb-3 p-3 bg-surface-hover/50 rounded-lg border border-border">
            <div className="flex items-center gap-2 mb-2">
              <Search className="h-4 w-4 text-accent" />
              <span className="text-xs font-medium text-foreground">Multi-Query Expansion</span>
              <span className="text-xs text-muted">({message.queryVariations!.variations.length} queries)</span>
            </div>
            <div className="space-y-1">
              {message.queryVariations!.variations.map((query, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-xs text-muted shrink-0 w-4">{idx + 1}.</span>
                  <span className={clsx(
                    'text-xs',
                    idx === 0 ? 'text-foreground font-medium' : 'text-muted'
                  )}>
                    {query}
                    {idx === 0 && <span className="text-accent ml-1">(original)</span>}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pipeline Steps (collapsible) */}
        {!isUser && hasPipelineSteps && (
          <div className="mb-3">
            <button
              onClick={() => setShowPipeline(!showPipeline)}
              className="flex items-center gap-2 text-xs text-muted hover:text-foreground transition-colors"
            >
              <Zap className="h-3.5 w-3.5" />
              <span>Pipeline execution</span>
              {showPipeline ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              <span className="text-muted-foreground">
                ({message.pipelineSteps!.reduce((sum, s) => sum + (s.duration_ms || 0), 0).toFixed(0)}ms total)
              </span>
            </button>
            {showPipeline && (
              <div className="mt-2 pl-2 border-l-2 border-border space-y-1">
                {message.pipelineSteps!.map((step, idx) => {
                  const Icon = getStepIcon(step.name)
                  return (
                    <div key={idx} className="flex items-center gap-2 text-xs">
                      <Icon className="h-3 w-3 text-muted" />
                      <span className="text-foreground">{formatStepName(step.name)}</span>
                      {step.duration_ms && (
                        <span className="text-muted-foreground">{step.duration_ms.toFixed(0)}ms</span>
                      )}
                      {step.status === 'completed' && (
                        <Check className="h-3 w-3 text-success" />
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* LLM Thinking (Gemini 3) */}
        {!isUser && hasThinking && (
          <div className="mb-3">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="flex items-center gap-2 text-xs text-muted hover:text-foreground transition-colors"
            >
              <Brain className="h-3.5 w-3.5" />
              <span>LLM Reasoning</span>
              {showThinking ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            {showThinking && (
              <div className="mt-2 p-2 bg-surface-hover/30 rounded border border-border text-xs text-muted italic">
                {message.thinking}
              </div>
            )}
          </div>
        )}

        {/* Message content */}
        <div
          className={clsx(
            'rounded-2xl px-4 py-3',
            isUser
              ? 'bg-accent text-white rounded-tr-md'
              : 'bg-surface border border-border rounded-tl-md'
          )}
        >
          {message.isStreaming && message.content === '' ? (
            <LoadingDots />
          ) : (
            <div className={clsx('prose-chat', isUser && 'text-white')}>
              <ReactMarkdown
                components={{
                  // Render citation links
                  a: ({ href, children }) => {
                    // Check if it's a citation link like [1]
                    const text = String(children)
                    if (/^\[\d+\]$/.test(text)) {
                      const citationNum = parseInt(text.slice(1, -1))
                      return (
                        <button
                          onClick={() => {
                            const el = document.getElementById(`source-${citationNum}`)
                            el?.scrollIntoView({ behavior: 'smooth' })
                          }}
                          className="inline-flex items-center justify-center w-5 h-5 text-xs font-medium bg-accent/10 text-accent rounded hover:bg-accent/20 transition-colors mx-0.5"
                        >
                          {citationNum}
                        </button>
                      )
                    }
                    return <a href={href} className="text-accent hover:underline">{children}</a>
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Actions bar (for assistant messages) */}
        {!isUser && message.content && !message.isStreaming && (
          <div className="flex items-center gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-md text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
              title="Copy"
            >
              {copied ? (
                <Check className="h-4 w-4 text-success" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </button>
            <button
              className="p-1.5 rounded-md text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
              title="Regenerate"
            >
              <RotateCcw className="h-4 w-4" />
            </button>

            {hasSources && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-muted hover:text-foreground hover:bg-surface-hover transition-colors ml-2"
              >
                <FileText className="h-3.5 w-3.5" />
                {showSources ? (
                  <ChevronUp className="h-3.5 w-3.5" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5" />
                )}
                {message.sources?.length} sources
              </button>
            )}
          </div>
        )}

        {/* Sources with Citations */}
        {hasSources && showSources && !message.isStreaming && (
          <div className="mt-3">
            <SourceCards sources={message.sources!} />
          </div>
        )}
      </div>
    </div>
  )
}
