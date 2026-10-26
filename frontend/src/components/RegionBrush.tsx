import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { fillPolygon, lockAxis, rectPoints, segment, type DrawTool, type Pt } from '../lib/brush'
import { combineRegion, REGION_HARDNESS, type RegionEdits } from '../lib/colorAdjust'

export type RegionMode = 'add' | 'remove'

export interface RegionBrushHandle {
  undo: () => void
}

interface Props {
  /** область, найденная автоматически (разница с кадром «до») */
  auto: HTMLCanvasElement
  /** правки — рисуются сюда (живут у владельца, переживают пересчёт области) */
  edits: RegionEdits
  /** рисовать: добавить в область / убрать из неё; null — только показывать область */
  mode: RegionMode | null
  /** кисть, прямоугольник или многоугольник — прямые границы области */
  tool?: DrawTool
  /** жёсткость края всей области 0…1 */
  hardness?: number
  /** сдвиг края всей области, px кадра: > 0 — расширить, < 0 — поджать */
  offset?: number
  /** диаметр кисти в пикселях экрана */
  size: number
  /** масштаб и сдвиг (ZoomView); холст — object-contain на месте картинки */
  imageStyle?: CSSProperties
  /** правка закончена — пересобрать цвет по новой области */
  onCommit: () => void
  onHistory?: (undoable: number) => void
  className?: string
}

const HISTORY = 15
const TINT = '#ff2bd6'
const POLY_CLOSE = 10 // многоугольник замыкается кликом не дальше стольких экранных px от первой точки
const INK = 'rgba(0,0,0,1)'

/** Область цвета: подсветка итоговой области и правки «добавить» / «убрать».
 *  Кисть: Shift+клик — прямая от конца прошлого мазка, Shift в мазке — по горизонтали / вертикали.
 *  Прямоугольник — протянуть (Shift — квадрат). Многоугольник — точки кликами; Enter, двойной клик или клик в первую
 *  точку замыкают, Backspace убирает точку, Esc отменяет. Край мазков жёсткий — мягкость задаёт «Край» всей области. */
export const RegionBrush = forwardRef<RegionBrushHandle, Props>(function RegionBrush(
  { auto, edits, mode, tool = 'brush', hardness = REGION_HARDNESS, offset = 0, size, imageStyle, onCommit, onHistory, className = '' },
  ref,
) {
  const view = useRef<HTMLCanvasElement>(null)
  const history = useRef<[ImageData, ImageData][]>([])
  const last = useRef<Pt | null>(null)
  const lastEnd = useRef<Pt | null>(null)
  const anchor = useRef<Pt | null>(null)
  const axis = useRef<'x' | 'y' | null>(null)
  const rectStart = useRef<Pt | null>(null)
  const scale = useRef(1)
  const frame = useRef(0)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [outline, setOutline] = useState<Pt[]>([])
  const [rubber, setRubber] = useState<Pt | null>(null)
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
    c.drawImage(combineRegion(auto, edits, hardness, offset), 0, 0)
    c.globalCompositeOperation = 'source-over'
  }
  const schedule = () => { cancelAnimationFrame(frame.current); frame.current = requestAnimationFrame(render) }

  useEffect(() => { render(); return () => cancelAnimationFrame(frame.current) }, [auto, edits, hardness, offset]) // eslint-disable-line react-hooks/exhaustive-deps
  // смена инструмента или режима — незаконченный контур и «прямая от конца мазка» не переносятся
  useEffect(() => { setOutline([]); setRubber(null); lastEnd.current = null }, [tool, mode])

  function snapshot() {
    history.current.push([
      edits.add.getContext('2d')!.getImageData(0, 0, edits.add.width, edits.add.height),
      edits.remove.getContext('2d')!.getImageData(0, 0, edits.remove.width, edits.remove.height),
    ])
    if (history.current.length > HISTORY) history.current.shift()
    onHistory?.(history.current.length)
  }

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
    scale.current = k
    return { x: (e.clientX - r.left - (r.width - W * k) / 2) / k, y: (e.clientY - r.top - (r.height - H * k) / 2) / k }
  }

  // добавить: в «добавлено» и стереть из «убрано» — и наоборот; так поверх прошлой правки можно передумать
  function paint(draw: (c: CanvasRenderingContext2D) => void) {
    const [into, outOf] = mode === 'add' ? [edits.add, edits.remove] : [edits.remove, edits.add]
    const i = into.getContext('2d')!
    i.globalCompositeOperation = 'source-over'
    draw(i)
    const o = outOf.getContext('2d')!
    o.globalCompositeOperation = 'destination-out'
    draw(o)
    o.globalCompositeOperation = 'source-over'
    schedule()
  }
  const stroke = (a: Pt, b: Pt) => paint((c) => segment(c, a, b, size / 2 / scale.current, 'circle', INK))

  function fillOutline(pts: Pt[]) {
    if (pts.length < 3) return
    snapshot()
    paint((c) => fillPolygon(c, pts, INK))
    onCommit()
  }

  function closePoly() {
    fillOutline(outline)
    setOutline([])
    setRubber(null)
  }

  // многоугольник: Enter — замкнуть, Backspace — убрать точку, Esc — отменить
  useEffect(() => {
    if (tool !== 'poly' || !outline.length) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter') closePoly()
      else if (e.key === 'Backspace') setOutline((o) => o.slice(0, -1))
      else if (e.key === 'Escape') { setOutline([]); setRubber(null) }
      else return
      e.preventDefault()
      e.stopPropagation()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  const active = mode !== null
  return (
    <>
      <canvas
        ref={view}
        width={W}
        height={H}
        className={`absolute inset-0 size-full object-contain ${active ? (tool === 'brush' ? 'cursor-none' : 'cursor-crosshair') : 'pointer-events-none'} ${className}`}
        style={imageStyle}
        onPointerDown={(e) => {
          if (!active || e.button !== 0) return
          const p = toImage(e)
          if (tool === 'poly') {
            if (outline.length >= 3 && (e.detail >= 2 || Math.hypot(p.x - outline[0].x, p.y - outline[0].y) <= POLY_CLOSE / scale.current)) closePoly()
            else if (e.detail < 2) setOutline((o) => [...o, p])
            return
          }
          e.currentTarget.setPointerCapture(e.pointerId)
          if (tool === 'rect') {
            rectStart.current = p
            setOutline(rectPoints(p, p))
            return
          }
          snapshot()
          stroke(e.shiftKey && lastEnd.current ? lastEnd.current : p, p)
          last.current = p
          anchor.current = p
          axis.current = null
        }}
        onPointerMove={(e) => {
          if (!active) return
          setCursor({ x: e.clientX, y: e.clientY })
          let p = toImage(e)
          if (tool === 'poly') {
            if (outline.length) setRubber(p)
            return
          }
          if (tool === 'rect') {
            const a = rectStart.current
            if (!a || !(e.buttons & 1)) return
            if (e.shiftKey) {
              const d = Math.max(Math.abs(p.x - a.x), Math.abs(p.y - a.y))
              p = { x: a.x + Math.sign(p.x - a.x || 1) * d, y: a.y + Math.sign(p.y - a.y || 1) * d }
            }
            setOutline(rectPoints(a, p))
            return
          }
          if (!(e.buttons & 1) || !last.current) return
          if (e.shiftKey && anchor.current) {
            const locked = lockAxis(anchor.current, p, axis.current, 4 / scale.current)
            p = locked.p
            axis.current = locked.axis
          } else {
            axis.current = null
          }
          stroke(last.current, p)
          last.current = p
        }}
        onPointerUp={() => {
          if (tool === 'rect' && rectStart.current) {
            const [a, , b] = outline
            if (a && b && Math.abs(b.x - a.x) * scale.current > 2 && Math.abs(b.y - a.y) * scale.current > 2) fillOutline(outline)
            rectStart.current = null
            setOutline([])
            return
          }
          if (!last.current) return
          lastEnd.current = last.current
          last.current = null
          onCommit()
        }}
        onPointerLeave={() => setCursor(null)}
      />
      {/* контур прямоугольника / многоугольника — в тех же координатах, что и холст */}
      {outline.length > 0 && (
        <svg
          viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
          className={`pointer-events-none absolute inset-0 size-full ${className}`} style={imageStyle}
        >
          <path
            d={`M${[...outline, ...(rubber && tool === 'poly' ? [rubber] : [])].map((q) => `${q.x},${q.y}`).join('L')}Z`}
            fill={mode === 'add' ? TINT : '#ffffff'} fillOpacity={0.25} stroke="#fff" strokeWidth={1.5}
            strokeDasharray="6 6" vectorEffect="non-scaling-stroke" className="animate-ants"
          />
          {tool === 'poly' && outline.map((q, n) => (
            <circle key={n} cx={q.x} cy={q.y} r={(n ? 3 : 5) / scale.current} fill={n ? '#fff' : TINT} stroke="#000" strokeOpacity={0.4} vectorEffect="non-scaling-stroke" />
          ))}
        </svg>
      )}
      {active && cursor && tool === 'brush' && createPortal(
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
