import { useCallback, useRef, useState } from 'react'
import { ChevronsLeftRight } from 'lucide-react'

interface Props {
  before: string
  after: string
}

export function CompareSlider({ before, after }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState(50)
  const [dragging, setDragging] = useState(false)

  const move = useCallback((clientX: number) => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)))
  }, [])

  return (
    <div
      ref={ref}
      className="relative size-full cursor-ew-resize touch-none select-none overflow-hidden"
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDragging(true); move(e.clientX) }}
      onPointerMove={(e) => dragging && move(e.clientX)}
      onPointerUp={() => setDragging(false)}
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 2))
        if (e.key === 'ArrowRight') setPos((p) => Math.min(100, p + 2))
      }}
      tabIndex={0}
      role="slider"
      aria-label="Сравнение до и после"
      aria-valuenow={Math.round(pos)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <img src={after} alt="После" className="absolute inset-0 size-full object-contain" draggable={false} />
      <img
        src={before}
        alt="До"
        className="absolute inset-0 size-full object-contain"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        draggable={false}
      />

      <Label className="left-4">До</Label>
      <Label className="right-4">После</Label>

      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -translate-x-1/2 w-[2px] bg-white shadow-[0_0_12px_rgba(0,0,0,0.25)]" />
        <div
          className={`glass absolute top-1/2 grid size-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full shadow-float ring-1 ring-fg/5 transition-transform duration-300 ease-apple ${
            dragging ? 'scale-110' : ''
          }`}
        >
          <ChevronsLeftRight className="size-[18px] text-ink" strokeWidth={2} />
        </div>
      </div>
    </div>
  )
}

function Label({ children, className }: { children: React.ReactNode; className: string }) {
  return (
    <span className={`glass pointer-events-none absolute top-4 rounded-full px-3 py-1 text-[12px] font-medium text-ink shadow-soft ${className}`}>
      {children}
    </span>
  )
}
