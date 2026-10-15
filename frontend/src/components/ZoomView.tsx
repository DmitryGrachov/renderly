import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useZoomPan, ZOOM_MAX, ZOOM_MIN } from '../lib/zoom'
import { ZoomControls } from './ZoomControls'

interface Props {
  /** Пропорция изображения (ширина / высота) — для границ перемещения */
  aspect: number
  /** Левая кнопка тоже перемещает при приближении (выключено, если левая занята, например ползунком сравнения) */
  panWithLeft?: boolean
  /** Изображения внутри получают style — масштаб и сдвиг (картинки object-contain на всю сцену) */
  children: (imageStyle: CSSProperties) => ReactNode
}

/** Сцена просмотра с масштабом и перемещением: колесо, кнопки справа, пробел / средняя кнопка / перетаскивание. */
export function ZoomView({ aspect, panWithLeft = true, children }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const baseW = Math.min(size.w, size.h * aspect)
  const { view, zoomIn, zoomOut, zoomFit, panning, panTool, panHandlers } = useZoomPan(
    ref,
    { baseW, baseH: baseW / aspect, availW: size.w, availH: size.h, cx: size.w / 2, cy: size.h / 2 },
    { panWithLeft },
  )
  const imageStyle: CSSProperties = { transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})` }

  return (
    <div
      ref={ref}
      className={`absolute inset-0 touch-none overflow-hidden ${panning ? 'cursor-grabbing' : panTool ? 'cursor-grab' : ''}`}
      {...panHandlers}
    >
      {children(imageStyle)}
      <div className="absolute right-4 top-1/2 z-10 -translate-y-1/2">
        <ZoomControls zoom={view.zoom} min={ZOOM_MIN} max={ZOOM_MAX} onZoomIn={zoomIn} onZoomOut={zoomOut} onFit={zoomFit} />
      </div>
    </div>
  )
}
