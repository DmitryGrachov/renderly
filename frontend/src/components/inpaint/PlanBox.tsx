import { useRef } from 'react'
import { motion } from 'motion/react'
import { ArrowUp, X } from 'lucide-react'

/** Рамка в долях изображения: x1, y1, x2, y2 */
export type Box = [number, number, number, number]

type Handle = 'move' | 'nw' | 'ne' | 'sw' | 'se'

interface Props {
  box: Box
  onChange: (b: Box) => void
  onConfirm: () => void
  onCancel: () => void
  /** Размер изображения на экране — для перевода сдвига мыши в доли */
  width: number
  height: number
}

const MIN = 0.02 // минимальная сторона рамки, доля изображения
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v))

/**
 * Рамка будущего объекта (план правки) поверх изображения: где он встанет и какого размера.
 * Её можно двигать и растягивать за углы, затем — «Сгенерировать». Низ рамки — точка, где объект касается земли.
 */
export function PlanBox({ box, onChange, onConfirm, onCancel, width, height }: Props) {
  const drag = useRef<{ handle: Handle; x: number; y: number; box: Box } | null>(null)

  const down = (handle: Handle) => (e: React.PointerEvent) => {
    e.stopPropagation()
    e.preventDefault()
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* синтетические события */ }
    drag.current = { handle, x: e.clientX, y: e.clientY, box }
  }
  const move = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    e.stopPropagation()
    const dx = (e.clientX - d.x) / Math.max(1, width)
    const dy = (e.clientY - d.y) / Math.max(1, height)
    let [x1, y1, x2, y2] = d.box
    if (d.handle === 'move') {
      const w = x2 - x1
      const h = y2 - y1
      x1 = clamp(x1 + dx, 0, 1 - w)
      y1 = clamp(y1 + dy, 0, 1 - h)
      x2 = x1 + w
      y2 = y1 + h
    } else {
      if (d.handle.includes('w')) x1 = clamp(x1 + dx, 0, x2 - MIN)
      if (d.handle.includes('e')) x2 = clamp(x2 + dx, x1 + MIN, 1)
      if (d.handle.includes('n')) y1 = clamp(y1 + dy, 0, y2 - MIN)
      if (d.handle.includes('s')) y2 = clamp(y2 + dy, y1 + MIN, 1)
    }
    onChange([x1, y1, x2, y2])
  }
  const up = (e: React.PointerEvent) => {
    if (!drag.current) return
    e.stopPropagation()
    drag.current = null
  }

  const [x1, y1, x2, y2] = box
  const corners: [Handle, number, number][] = [['nw', x1, y1], ['ne', x2, y1], ['sw', x1, y2], ['se', x2, y2]]
  const below = y2 < 0.82 // кнопки под рамкой, если снизу есть место

  return (
    <div className="absolute inset-0 z-20" data-no-zoom onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
      <div
        onPointerDown={down('move')}
        className="absolute cursor-move rounded-[10px] border-2 border-dashed border-white bg-white/10 shadow-[0_0_0_1px_rgba(0,0,0,0.4),0_0_0_9999px_rgba(0,0,0,0.25)]"
        style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }}
      >
        {/* земля — где объект касается поверхности */}
        <div className="pointer-events-none absolute inset-x-2 -bottom-px h-[3px] rounded-full bg-accent" />
      </div>
      {corners.map(([h, x, y]) => (
        <div
          key={h}
          onPointerDown={down(h)}
          className={`absolute size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-accent shadow ${
            h === 'nw' || h === 'se' ? 'cursor-nwse-resize' : 'cursor-nesw-resize'
          }`}
          style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
        />
      ))}
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass absolute flex -translate-x-1/2 items-center gap-1 rounded-full p-1 shadow-float ring-1 ring-fg/5"
        style={{
          left: `${((x1 + x2) / 2) * 100}%`,
          top: below ? `calc(${y2 * 100}% + 12px)` : `calc(${y1 * 100}% - 52px)`,
        }}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <span className="whitespace-nowrap px-2.5 text-[12px] text-ink-2">Размер и место объекта</span>
        <button
          onClick={onCancel}
          aria-label="Отмена"
          className="grid size-8 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink"
        >
          <X className="size-4" />
        </button>
        <button
          onClick={onConfirm}
          className="flex items-center gap-1.5 rounded-full bg-accent px-3.5 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-accent-hover"
        >
          Сгенерировать <ArrowUp className="size-3.5" strokeWidth={2.5} />
        </button>
      </motion.div>
    </div>
  )
}
