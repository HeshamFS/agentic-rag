import { clsx } from 'clsx'

interface ToggleProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label?: string
  description?: string
  disabled?: boolean
  className?: string
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  className,
}: ToggleProps) {
  return (
    <label
      className={clsx(
        'flex items-start gap-3 cursor-pointer',
        disabled && 'cursor-not-allowed opacity-50',
        className
      )}
    >
      <div className="relative flex-shrink-0 pt-0.5">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="sr-only"
        />
        <div
          className={clsx(
            'h-5 w-9 rounded-full transition-colors duration-200',
            checked ? 'bg-accent' : 'bg-surface-active'
          )}
        >
          <div
            className={clsx(
              'absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-soft transition-transform duration-200',
              checked ? 'translate-x-4' : 'translate-x-0.5'
            )}
          />
        </div>
      </div>
      {(label || description) && (
        <div className="flex-1 min-w-0">
          {label && (
            <span className="block text-sm font-medium text-foreground">
              {label}
            </span>
          )}
          {description && (
            <span className="block text-xs text-muted mt-0.5">
              {description}
            </span>
          )}
        </div>
      )}
    </label>
  )
}
