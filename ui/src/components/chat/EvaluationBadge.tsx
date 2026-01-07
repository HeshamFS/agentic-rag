import { clsx } from 'clsx'
import { CheckCircle, AlertCircle, XCircle } from 'lucide-react'

interface EvaluationBadgeProps {
  score: number
  showLabel?: boolean
}

export function EvaluationBadge({ score, showLabel = true }: EvaluationBadgeProps) {
  const percentage = Math.round(score * 100)

  const getStatus = () => {
    if (score >= 0.8) return { icon: CheckCircle, color: 'text-success', bg: 'bg-success-light', label: 'High quality' }
    if (score >= 0.6) return { icon: AlertCircle, color: 'text-warning', bg: 'bg-warning-light', label: 'Moderate' }
    return { icon: XCircle, color: 'text-error', bg: 'bg-error-light', label: 'Low quality' }
  }

  const status = getStatus()
  const Icon = status.icon

  return (
    <div
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium',
        status.bg,
        status.color
      )}
      title={`RAGAS Score: ${percentage}%`}
    >
      <Icon className="h-3 w-3" />
      <span className="font-mono">{percentage}%</span>
      {showLabel && <span className="hidden sm:inline">{status.label}</span>}
    </div>
  )
}
