import { BarChart3, CheckCircle, AlertCircle, XCircle, Info } from 'lucide-react'
import { clsx } from 'clsx'

// Mock data for demonstration - in real app, this would come from API/store
const mockMetrics = {
  context_precision: 0.82,
  context_recall: 0.75,
  faithfulness: 0.91,
  answer_relevancy: 0.78,
  ragas_score: 0.815,
}

const mockSelfRAG = {
  isrel: 'FULLY' as const,
  issup: 'FULLY' as const,
  isuse: 'PARTIALLY' as const,
  overall_score: 0.83,
}

export function EvaluationPanel() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Evaluation Metrics</h1>
          <p className="text-muted text-sm mt-1">
            Monitor RAG pipeline quality using RAGAS framework and Self-RAG reflection
          </p>
        </div>

        {/* RAGAS Overview */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-accent-light flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-accent" />
              </div>
              <div>
                <h2 className="font-semibold text-foreground">RAGAS Score</h2>
                <p className="text-xs text-muted">Aggregated quality metric</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-accent">
                {(mockMetrics.ragas_score * 100).toFixed(0)}%
              </div>
              <p className="text-xs text-muted">Overall score</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <MetricCard
              name="Context Precision"
              value={mockMetrics.context_precision}
              description="Relevance of retrieved context"
            />
            <MetricCard
              name="Context Recall"
              value={mockMetrics.context_recall}
              description="Coverage of ground truth"
            />
            <MetricCard
              name="Faithfulness"
              value={mockMetrics.faithfulness}
              description="Grounding in retrieved context"
            />
            <MetricCard
              name="Answer Relevancy"
              value={mockMetrics.answer_relevancy}
              description="How well response addresses query"
            />
          </div>
        </div>

        {/* Self-RAG Reflection */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-accent-light flex items-center justify-center">
              <Info className="h-5 w-5 text-accent" />
            </div>
            <div>
              <h2 className="font-semibold text-foreground">Self-RAG Reflection</h2>
              <p className="text-xs text-muted">Quality assessment from reflection tokens</p>
            </div>
          </div>

          <div className="space-y-4">
            <ReflectionToken
              name="ISREL (Relevance)"
              value={mockSelfRAG.isrel}
              description="Is the retrieved context relevant to the query?"
            />
            <ReflectionToken
              name="ISSUP (Support)"
              value={mockSelfRAG.issup}
              description="Is the response supported by the context?"
            />
            <ReflectionToken
              name="ISUSE (Usefulness)"
              value={mockSelfRAG.isuse}
              description="Is the response useful to the user?"
            />
          </div>

          <div className="mt-6 pt-4 border-t border-border">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted">Overall Self-RAG Score</span>
              <span className="text-lg font-semibold text-foreground">
                {(mockSelfRAG.overall_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Interpretation Guide */}
        <div className="card p-6 bg-surface-hover/50">
          <h3 className="font-semibold text-foreground mb-4">Interpretation Guide</h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-success mt-1.5" />
              <div>
                <p className="font-medium text-foreground">High Scores (≥80%)</p>
                <p className="text-muted">Good performance, reliable responses</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-warning mt-1.5" />
              <div>
                <p className="font-medium text-foreground">Moderate Scores (60-79%)</p>
                <p className="text-muted">Acceptable but room for improvement</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-error mt-1.5" />
              <div>
                <p className="font-medium text-foreground">Low Scores (&lt;60%)</p>
                <p className="text-muted">Consider adjusting pipeline settings</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

interface MetricCardProps {
  name: string
  value: number
  description: string
}

function MetricCard({ name, value, description }: MetricCardProps) {
  const percentage = Math.round(value * 100)
  const color = value >= 0.8 ? 'text-success' : value >= 0.6 ? 'text-warning' : 'text-error'
  const bgColor = value >= 0.8 ? 'bg-success' : value >= 0.6 ? 'bg-warning' : 'bg-error'

  return (
    <div className="p-4 rounded-lg bg-surface-hover/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-foreground">{name}</span>
        <span className={clsx('text-lg font-bold', color)}>{percentage}%</span>
      </div>
      <div className="h-2 rounded-full bg-surface-active overflow-hidden mb-2">
        <div
          className={clsx('h-full rounded-full transition-all', bgColor)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <p className="text-xs text-muted">{description}</p>
    </div>
  )
}

interface ReflectionTokenProps {
  name: string
  value: 'FULLY' | 'PARTIALLY' | 'NOT'
  description: string
}

function ReflectionToken({ name, value, description }: ReflectionTokenProps) {
  const getIcon = () => {
    switch (value) {
      case 'FULLY':
        return <CheckCircle className="h-5 w-5 text-success" />
      case 'PARTIALLY':
        return <AlertCircle className="h-5 w-5 text-warning" />
      case 'NOT':
        return <XCircle className="h-5 w-5 text-error" />
    }
  }

  const getBadgeColor = () => {
    switch (value) {
      case 'FULLY':
        return 'bg-success-light text-success'
      case 'PARTIALLY':
        return 'bg-warning-light text-warning'
      case 'NOT':
        return 'bg-error-light text-error'
    }
  }

  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-surface-hover/50">
      {getIcon()}
      <div className="flex-1">
        <p className="text-sm font-medium text-foreground">{name}</p>
        <p className="text-xs text-muted">{description}</p>
      </div>
      <span className={clsx('px-2 py-1 rounded text-xs font-medium', getBadgeColor())}>
        {value}
      </span>
    </div>
  )
}
