import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { lockAxis, segment, type Pt } from '../lib/brush'
import { combineRegion, type RegionEdits } from '../lib/colorAdjust'

export type RegionMode = 'add' | 'remove'

export interface RegionBrushHandle {
  undo: () => void
}

interface Props {
  /** область, найденная автоматически (разница с кадром «до») */
  auto: HTMLCanvasElement
  /** правки кистью — рисуются сюда (живут у владельца, переживают пересчёт области) */
  edits: RegionEdits
  /** рисовать: добавить в область / убрать из неё; null — только показывать область */
  mode: RegionMode | null
  /** диаметр кисти в пикселях экрана */
  size: number
  /** масштаб и сдвиг (ZoomView); холст — object-contain на месте картинки */
  imageStyle?: CSSProperties
  /** мазок закончен — пересобрать цвет по новой области */
  onCommit: () => void
  onHistory?: (undoable: number) => void
  className?: string
}

const HISTORY = 15
const HARDNESS = 0.6 // мягкий край мазка: граница области не режет цвет ступенькой
const TINT = '#ff2bd6'

/** Кисть области цвета: подсветка итоговой области и мазки «добавить» / «убрать».
 *  Shift+клик — прямая от конца прошлого мазка, Shift в мазке — по горизонтали / вертикали. */
export const RegionBrush = forwardRef<RegionBrushHandle, Props>(function RegionBrush(
  { auto, edits, mode, size, imageStyle, onCommit, onHistory, className = '' },
  ref,
) {
  const view = useRef<HTMLCanvasElement>(null)
  const history = useRef<[ImageData, ImageData][]>([])
  const last = useRef<Pt | null>(null)
  const lastEnd = useRef<Pt | null>(null)
  const anchor = useRef<Pt | null>(null)
  const axis = useRef<'x' | 'y' | null>(null)
  const frame = useRef(0)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const W = auto.width
  const H = auto.height

  function render() {
    const c = view.current?.getContext('2d')
    if (!c) return
    c.clearRect(0, 0, W, H)
    c.globalCompositeOperation = 'source-over'
    c.fillStyle = TINT
    c.globalAlpha = 0.45
    c.fillRect(0, 0, W, H)
    c.globalAlpha = 1
    c.globalCompositeOperation = 'destination-in'
    c.drawImage(combineRegion(auto, edits), 0, 0)
    c.globalCompositeOperation = 'source-over'
  }
  const schedule = () => { cancelAnimationFrame(frame.current); frame.current = requestAnimationFrame(render) }

  useEffect(() => { render(); return () => cancelAnimationFrame(frame.current) }, [auto, edits]) // eslint-disable-line react-hooks/exhaustive-deps

  useImperativeHandle(ref, () => ({
    undo() {
      const snap = history.current.pop()
      if (!snap) return
      edits.add.getContext('2d')!.putImageData(snap[0], 0, 0)
      edits.remove.getContext('2d')!.putImageData(snap[1], 0, 0)
      onHistory?.(history.current.length)
      render()
      onCommit()
    },
  }))

  function toImage(e: React.PointerEvent) {
    const r = view.current!.getBoundingClientRect()
    const k = Math.min(r.width / W, r.height / H)
    return { p: { x: (e.clientX - r.left - (r.width - W * k) / 2) / k, y: (e.clientY - r.top - (r.height - H * k) / 2) / k }, k }
  }

  // добавить: мазок в «добавлено» и стирание из «убрано» — и наоборот; так поверх прошлой правки можно передумать
  function stroke(a: Pt, b: Pt, k: number) {
    const [into, outOf] = mode === 'add' ? [edits.add, edits.remove] : [edits.remove, edits.add]
    const r = size / 2 / k
    const i = into.getContext('2d')!
    i.globalCompositeOperation = 'source-over'
    segment(i, a, b, r, 'circle', 'rgba(0,0,0,1)', HARDNESS)
    const o = outOf.getContext('2d')!
    o.globalCompositeOperation = 'destination-out'
    segment(o, a, b, r, 'circle', 'rgba(0,0,0,1)', HARDNESS)
    o.globalCompositeOperation = 'source-over'
    schedule()
  }

  const active = mode !== null
  return (
    <>
      <canvas
        ref={view}
        width={W}
        height={H}
        className={`absolute inset-0 size-full object-contain ${active ? 'cursor-none' : 'pointer-events-none'} ${className}`}
        style={imageStyle}
        onPointerDown={(e) => {
          if (!active || e.button !== 0) return
          e.currentTarget.setPointerCapture(e.pointerId)
          const { p, k } = toImage(e)
          history.current.push([
            edits.add.getContext('2d')!.getImageData(0, 0, edits.add.width, edits.add.height),
            edits.remove.getContext('2d')!.getImageData(0, 0, edits.remove.width, edits.remove.height),
          ])
          if (history.current.length > HISTORY) history.current.shift()
          onHistory?.(history.current.length)
          stroke(e.shiftKey && lastEnd.current ? lastEnd.current : p, p, k)
          last.current = p
          anchor.current = p
          axis.current = null
        }}
        onPointerMove={(e) => {
          if (!active) return
          setCursor({ x: e.clientX, y: e.clientY })
          if (!(e.buttons & 1) || !last.current) return
          let { p } = toImage(e)
          const { k } = toImage(e)
          if (e.shiftKey && anchor.current) {
            const locked = lockAxis(anchor.current, p, axis.current, 4 / k)
            p = locked.p
            axis.current = locked.axis
          } else {
            axis.current = null
          }
          stroke(last.current, p, k)
          last.current = p
        }}
        onPointerUp={() => {
          if (!last.current) return
          lastEnd.current = last.current
          last.current = null
          onCommit()
        }}
        onPointerLeave={() => setCursor(null)}
      />
      {active && cursor && createPortal(
        <div
          aria-hidden
          className={`pointer-events-none fixed z-50 grid place-items-center rounded-full border-2 text-[14px] font-bold text-white shadow-[0_0_0_1px_rgba(0,0,0,0.45)] ${mode === 'add' ? 'border-white' : 'border-dashed border-white'}`}
          style={{ left: cursor.x - size / 2, top: cursor.y - size / 2, width: size, height: size, textShadow: '0 0 3px #000' }}
        >
          {mode === 'add' ? '+' : '−'}
        </div>,
        document.body,
      )}
    </>
  )
})
