import { clsx } from 'clsx'

interface SliderProps {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  label?: string
  showValue?: boolean
  valueFormatter?: (value: number) => string
  disabled?: boolean
  className?: string
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step = 1,
  label,
  showValue = true,
  valueFormatter = (v) => v.toString(),
  disabled = false,
  className,
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className={clsx('space-y-2', className)}>
      {(label || showValue) && (
        <div className="flex items-center justify-between">
          {label && (
            <label className="text-sm font-medium text-foreground">
              {label}
            </label>
          )}
          {showValue && (
            <span className="text-sm text-muted font-mono">
              {valueFormatter(value)}
            </span>
          )}
        </div>
      )}
      <div className="relative h-2">
        <div className="absolute inset-0 rounded-full bg-surface-active" />
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-accent transition-all"
          style={{ width: `${percentage}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className={clsx(
            'absolute inset-0 w-full cursor-pointer appearance-none bg-transparent',
            'disabled:cursor-not-allowed disabled:opacity-50',
            // Thumb styles
            '[&::-webkit-slider-thumb]:appearance-none',
            '[&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4',
            '[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent',
            '[&::-webkit-slider-thumb]:shadow-medium [&::-webkit-slider-thumb]:cursor-pointer',
            '[&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-110',
            '[&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4',
            '[&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-accent',
            '[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:shadow-medium'
          )}
        />
      </div>
    </div>
  )
}
