import { useState, type ReactNode } from 'react'
import { clsx } from 'clsx'

interface TooltipProps {
  content: string
  children: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
}

export function Tooltip({
  content,
  children,
  position = 'top',
  delay = 200,
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [timeoutId, setTimeoutId] = useState<ReturnType<typeof setTimeout> | null>(null)

  const showTooltip = () => {
    const id = setTimeout(() => setIsVisible(true), delay)
    setTimeoutId(id)
  }

  const hideTooltip = () => {
    if (timeoutId) clearTimeout(timeoutId)
    setIsVisible(false)
  }

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
    >
      {children}
      {isVisible && (
        <div
          className={clsx(
            'absolute z-50 px-2 py-1 text-xs font-medium text-white bg-foreground rounded-md',
            'whitespace-nowrap animate-fade-in pointer-events-none',
            {
              'bottom-full left-1/2 -translate-x-1/2 mb-2': position === 'top',
              'top-full left-1/2 -translate-x-1/2 mt-2': position === 'bottom',
              'right-full top-1/2 -translate-y-1/2 mr-2': position === 'left',
              'left-full top-1/2 -translate-y-1/2 ml-2': position === 'right',
            }
          )}
        >
          {content}
          {/* Arrow */}
          <div
            className={clsx(
              'absolute w-2 h-2 bg-foreground rotate-45',
              {
                'top-full left-1/2 -translate-x-1/2 -mt-1': position === 'top',
                'bottom-full left-1/2 -translate-x-1/2 -mb-1': position === 'bottom',
                'left-full top-1/2 -translate-y-1/2 -ml-1': position === 'left',
                'right-full top-1/2 -translate-y-1/2 -mr-1': position === 'right',
              }
            )}
          />
        </div>
      )}
    </div>
  )
}
