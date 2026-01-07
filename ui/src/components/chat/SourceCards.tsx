import { useState } from 'react'
import { clsx } from 'clsx'
import { FileText, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'
import type { Source } from '../../types'

interface SourceCardsProps {
  sources: Source[]
}

export function SourceCards({ sources }: SourceCardsProps) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted uppercase tracking-wider">
        Sources ({sources.length})
      </p>
      <div className="space-y-2">
        {sources.map((source, index) => (
          <SourceCard key={index} source={source} index={index + 1} />
        ))}
      </div>
    </div>
  )
}

interface SourceCardProps {
  source: Source
  index: number
}

function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false)

  // Use citation number from source, fallback to index
  const citationNum = source.citation || index

  // Get display name: prefer filename, then extract from source path
  const displayName = source.filename
    || source.source?.split('/').pop()?.split('\\').pop()
    || `Source ${citationNum}`

  // Get file extension for styling
  const ext = displayName.split('.').pop()?.toLowerCase() || ''
  const extColors: Record<string, string> = {
    pdf: 'text-red-500',
    docx: 'text-blue-500',
    doc: 'text-blue-500',
    txt: 'text-gray-500',
    md: 'text-purple-500',
  }

  // Score color based on relevance (if score > 0)
  const hasScore = source.score > 0
  const scoreColor = source.score >= 0.8
    ? 'text-success'
    : source.score >= 0.6
      ? 'text-warning'
      : 'text-muted'

  // Truncate content for preview
  const preview = source.content.length > 200
    ? source.content.substring(0, 200) + '...'
    : source.content

  return (
    <div
      id={`source-${citationNum}`}
      className="rounded-lg border border-border bg-surface-hover/50 overflow-hidden scroll-mt-4"
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-3 text-left hover:bg-surface-hover transition-colors"
      >
        <div className="flex-shrink-0 w-6 h-6 rounded bg-accent text-white flex items-center justify-center text-xs font-bold">
          {citationNum}
        </div>
        <FileText className={clsx('h-4 w-4 flex-shrink-0', extColors[ext] || 'text-muted')} />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground truncate" title={displayName}>
            {displayName}
          </p>
        </div>
        {hasScore && (
          <span className={clsx('text-xs font-mono', scoreColor)}>
            {(source.score * 100).toFixed(0)}%
          </span>
        )}
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted" />
        )}
      </button>

      {/* Content preview */}
      {!expanded && (
        <div className="px-3 pb-3 pt-0">
          <p className="text-xs text-muted line-clamp-2">
            {preview}
          </p>
        </div>
      )}

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-border p-3 space-y-3">
          <div className="text-sm text-foreground whitespace-pre-wrap">
            {source.content}
          </div>
          {source.metadata && Object.keys(source.metadata).length > 0 && (
            <div className="pt-2 border-t border-border">
              <p className="text-xs font-medium text-muted mb-1">Metadata</p>
              <div className="flex flex-wrap gap-1">
                {Object.entries(source.metadata).map(([key, value]) => (
                  <span
                    key={key}
                    className="px-2 py-0.5 rounded-full bg-surface text-xs text-muted"
                  >
                    {key}: {String(value)}
                  </span>
                ))}
              </div>
            </div>
          )}
          {source.source && (
            <a
              href={source.source}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
            >
              View source
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      )}
    </div>
  )
}
