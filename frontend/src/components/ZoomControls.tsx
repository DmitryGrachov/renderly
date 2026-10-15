import { Minus, Plus } from 'lucide-react'
import { ToolButton } from './ToolButton'

interface ZoomControlsProps {
  zoom: number
  min: number
  max: number
  onZoomIn: () => void
  onZoomOut: () => void
  onFit: () => void
}

/** Масштаб канваса: +, текущий процент (клик — вписать), − */
export function ZoomControls({ zoom, min, max, onZoomIn, onZoomOut, onFit }: ZoomControlsProps) {
  return (
    <div className="glass flex flex-col items-center gap-1 rounded-[18px] p-1.5 shadow-float ring-1 ring-fg/5">
      <ToolButton label="Приблизить" shortcut="= · колесо" onClick={onZoomIn} disabled={zoom >= max} side="left">
        <Plus className="size-[18px]" strokeWidth={1.75} />
      </ToolButton>
      <ToolButton label="Вписать" shortcut="0" onClick={onFit} side="left">
        <span className="text-[11px] font-semibold tabular-nums">{Math.round(zoom * 100)}%</span>
      </ToolButton>
      <ToolButton label="Отдалить" shortcut="−" onClick={onZoomOut} disabled={zoom <= min} side="left">
        <Minus className="size-[18px]" strokeWidth={1.75} />
      </ToolButton>
    </div>
  )
}
