import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { fillPolygon, lockAxis, rectPoints, segment, type DrawTool, type Pt, type Shape } from '../lib/brush'

export interface RestoreBrushHandle {
  /** Отменить последний мазок */
  undo: () => void
  /** Стереть все мазки — снова весь результат */
  clear: () => void
  /** Итог: результат с прежним кадром под мазками, в размере результата */
  toBlob: () => Promise<Blob>
  /** Копия маски мазков (жёсткий край) — чтобы открыть тот же откат снова и поменять край или мазки */
  mask: () => HTMLCanvasElement | null
}

interface Props {
  /** прежний кадр — проявляется под кистью */
  before: string
  /** результат генерации */
  after: string
  /** масштаб и сдвиг ZoomView */
  imageStyle: CSSProperties
  /** диаметр кисти в пикселях экрана */
  size: number
  /** сколько мазков можно отменить */
  onHistory?: (undoable: number) => void
  /** сдвиг прежнего кадра в пикселях кадра — дотянуть остаток рассогласования вручную (стрелки) */
  offset?: { x: number; y: number }
  /** кисть, прямоугольник или многоугольник */
  tool?: DrawTool
  shape?: Shape
  /** жёсткость края 0…1 — для всей маски сразу, в том числе уже нарисованных мазков */
  hardness?: number
  /** маска прошлого отката по этому результату — открыть его снова */
  initialMask?: HTMLCanvasElement | null
  /** показать прежний кадр целиком, закрашенное — подсветкой: вести границы по нему (рисовать можно) */
  peek?: boolean
}

const HISTORY = 15 // снимков маски для отмены: кадр ~2 Мп — 8 МБ на снимок
// мягкий край: мазки в маске — с жёстким краем, при показе и сохранении маска размывается на эту долю длинной
// стороны кадра при нулевой жёсткости. Край — свойство всего отката, а не мазка: его меняют и после «Применить»
const FEATHER = 0.012
const POLY_CLOSE = 10 // многоугольник замыкается кликом не дальше стольких экранных px от первой точки
const MASK = 'rgba(0,0,0,1)'
const PEEK_TINT = '#f59e0b' // «Показать прошлый»: чем подсвечено закрашенное

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

/** Кисть отката: под мазками вместо результата — прежний кадр. Холст на месте картинки (object-contain), рисует
 *  левая кнопка, Alt — возвращает результат; пробел и средняя кнопка двигают кадр (ZoomView).
 *  Кисть: Shift+клик — прямая от конца прошлого мазка, Shift в мазке — по горизонтали / вертикали.
 *  Прямоугольник — протянуть (Shift — квадрат). Многоугольник — точки кликами; Enter, двойной клик или клик в первую
 *  точку замыкают, Backspace убирает точку, Esc отменяет. */
export const RestoreBrush = forwardRef<RestoreBrushHandle, Props>(function RestoreBrush(
  { before, after, imageStyle, size, onHistory, offset = { x: 0, y: 0 }, tool = 'brush', shape = 'circle', hardness = 0.55, initialMask, peek = false },
  ref,
) {
  const view = useRef<HTMLCanvasElement>(null)
  const layers = useRef<{ after: HTMLImageElement; before: HTMLCanvasElement; mask: HTMLCanvasElement; tmp: HTMLCanvasElement } | null>(null)
  const history = useRef<ImageData[]>([])
  const last = useRef<{ x: number; y: number } | null>(null)
  const frame = useRef(0)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [restoring, setRestoring] = useState(true)
  const lastEnd = useRef<Pt | null>(null)
  const anchor = useRef<Pt | null>(null)
  const axis = useRef<'x' | 'y' | null>(null)
  const scale = useRef(1) // экранных px на пиксель кадра — для толщин и допусков
  // контур прямоугольника / многоугольника в работе (пиксели кадра) и «резинка» до курсора
  const [outline, setOutline] = useState<Pt[]>([])
  const [rubber, setRubber] = useState<Pt | null>(null)
  const rectStart = useRef<Pt | null>(null)
  const [size2, setSize2] = useState({ w: 1, h: 1 })
  const shift = useRef(offset)
  shift.current = offset

  useEffect(() => { render() }, [offset.x, offset.y, hardness, peek]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let alive = true
    Promise.all([loadImage(after), loadImage(before)]).then(([a, b]) => {
      if (!alive || !view.current) return
      const { naturalWidth: w, naturalHeight: h } = a
      const canvas = (cw = w, ch = h) => Object.assign(document.createElement('canvas'), { width: cw, height: ch })
      // прежний кадр — в размере результата (бэкенд отдаёт результат в размере исходника, подогнанным по краям)
      const prev = canvas()
      prev.getContext('2d')!.drawImage(b, 0, 0, w, h)
      const mask = canvas()
      if (initialMask) mask.getContext('2d')!.drawImage(initialMask, 0, 0, w, h)
      layers.current = { after: a, before: prev, mask, tmp: canvas() }
      view.current.width = w
      view.current.height = h
      setSize2({ w, h })
      history.current = []
      onHistory?.(0)
      render()
    })
    return () => { alive = false; cancelAnimationFrame(frame.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [before, after])

  /** peekView — прежний кадр целиком и закрашенное подсветкой (только для глаз; в итог всегда идёт обычный вид) */
  function render(peekView = peek) {
    const l = layers.current
    const ctx = view.current?.getContext('2d')
    if (!l || !ctx) return
    const t = l.tmp.getContext('2d')!
    const blur = (1 - hardness) * FEATHER * Math.max(l.mask.width, l.mask.height)
    if (peekView) {
      ctx.drawImage(l.after, 0, 0)
      ctx.drawImage(l.before, shift.current.x, shift.current.y)
      t.globalCompositeOperation = 'copy'
      t.fillStyle = PEEK_TINT
      t.fillRect(0, 0, l.tmp.width, l.tmp.height)
      t.globalCompositeOperation = 'destination-in'
      t.filter = blur > 0.5 ? `blur(${blur}px)` : 'none'
      t.drawImage(l.mask, 0, 0)
      t.filter = 'none'
      ctx.globalAlpha = 0.4
      ctx.drawImage(l.tmp, 0, 0)
      ctx.globalAlpha = 1
      return
    }
    // прежний кадр, вырезанный маской, поверх результата
    t.globalCompositeOperation = 'copy'
    t.drawImage(l.before, shift.current.x, shift.current.y)
    t.globalCompositeOperation = 'destination-in'
    t.filter = blur > 0.5 ? `blur(${blur}px)` : 'none'
    t.drawImage(l.mask, 0, 0)
    t.filter = 'none'
    ctx.drawImage(l.after, 0, 0)
    ctx.drawImage(l.tmp, 0, 0)
  }

  function schedule() {
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => render())
  }

  /** точка экрана → пиксели кадра и масштаб (холст object-contain внутри своего прямоугольника после transform) */
  function toImage(e: React.PointerEvent) {
    const c = view.current!
    const r = c.getBoundingClientRect()
    const k = Math.min(r.width / c.width, r.height / c.height)
    const ox = r.left + (r.width - c.width * k) / 2
    const oy = r.top + (r.height - c.height * k) / 2
    return { x: (e.clientX - ox) / k, y: (e.clientY - oy) / k, k }
  }

  function snapshot() {
    const m = layers.current!.mask
    history.current.push(m.getContext('2d')!.getImageData(0, 0, m.width, m.height))
    if (history.current.length > HISTORY) history.current.shift()
    onHistory?.(history.current.length)
  }

  const radius = () => size / 2 / scale.current
  const maskCtx = (erase: boolean) => {
    const m = layers.current!.mask.getContext('2d')!
    m.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    return m
  }

  /** контур → маска (Alt — вернуть результат); край смягчается при показе, по всей маске */
  function fillOutline(pts: Pt[], erase: boolean) {
    if (pts.length < 3 || !layers.current) return
    snapshot()
    fillPolygon(maskCtx(erase), pts, MASK)
    schedule()
  }

  function closePoly(erase: boolean) {
    fillOutline(outline, erase)
    setOutline([])
    setRubber(null)
  }

  // многоугольник: Enter — замкнуть, Backspace — убрать точку, Esc — отменить
  useEffect(() => {
    if (tool !== 'poly' || !outline.length) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter') closePoly(e.altKey)
      else if (e.key === 'Backspace') setOutline((o) => o.slice(0, -1))
      else if (e.key === 'Escape') { setOutline([]); setRubber(null) }
      else return
      e.preventDefault()
      e.stopPropagation()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  // смена инструмента — незаконченный контур и «прямая от конца мазка» не переносятся
  useEffect(() => { setOutline([]); setRubber(null); lastEnd.current = null }, [tool])

  useImperativeHandle(ref, () => ({
    undo() {
      const l = layers.current
      const snap = history.current.pop()
      if (!l || !snap) return
      l.mask.getContext('2d')!.putImageData(snap, 0, 0)
      onHistory?.(history.current.length)
      render()
    },
    clear() {
      const l = layers.current
      if (!l) return
      const m = l.mask.getContext('2d')!
      history.current.push(m.getImageData(0, 0, l.mask.width, l.mask.height))
      if (history.current.length > HISTORY) history.current.shift()
      m.clearRect(0, 0, l.mask.width, l.mask.height)
      onHistory?.(history.current.length)
      render()
    },
    mask() {
      const m = layers.current?.mask
      if (!m) return null
      const copy = Object.assign(document.createElement('canvas'), { width: m.width, height: m.height })
      copy.getContext('2d')!.drawImage(m, 0, 0)
      return copy
    },
    toBlob() {
      render(false)
      return new Promise<Blob>((resolve, reject) =>
        view.current!.toBlob((b) => {
          if (peek) render()
          b ? resolve(b) : reject(new Error('Не удалось собрать кадр'))
        }, 'image/jpeg', 0.95),
      )
    },
  }))

  return (
    <>
      <canvas
        ref={view}
        className={`absolute inset-0 size-full object-contain ${tool === 'brush' ? 'cursor-none' : 'cursor-crosshair'}`}
        style={imageStyle}
        onPointerDown={(e) => {
          if (e.button !== 0 || !layers.current) return
          const { x, y, k } = toImage(e)
          scale.current = k
          const p = { x, y }
          if (tool === 'poly') {
            if (outline.length >= 3 && (e.detail >= 2 || Math.hypot(x - outline[0].x, y - outline[0].y) <= POLY_CLOSE / k)) {
              closePoly(e.altKey)
            } else if (e.detail < 2) {
              setOutline((o) => [...o, p])
            }
            return
          }
          e.currentTarget.setPointerCapture(e.pointerId)
          if (tool === 'rect') {
            rectStart.current = p
            setOutline(rectPoints(p, p))
            return
          }
          snapshot()
          // Shift+клик — прямая от конца прошлого мазка
          const from = e.shiftKey && lastEnd.current ? lastEnd.current : p
          segment(maskCtx(e.altKey), from, p, radius(), shape, MASK)
          last.current = p
          anchor.current = p
          axis.current = null
          schedule()
        }}
        onPointerMove={(e) => {
          setCursor({ x: e.clientX, y: e.clientY })
          setRestoring(!e.altKey)
          const { x, y, k } = toImage(e)
          scale.current = k
          let p = { x, y }
          if (tool === 'poly') {
            if (outline.length) setRubber(p)
            return
          }
          if (tool === 'rect') {
            const a = rectStart.current
            if (!a || !(e.buttons & 1)) return
            if (e.shiftKey) {
              const d = Math.max(Math.abs(x - a.x), Math.abs(y - a.y))
              p = { x: a.x + Math.sign(x - a.x || 1) * d, y: a.y + Math.sign(y - a.y || 1) * d }
            }
            setOutline(rectPoints(a, p))
            return
          }
          if (!(e.buttons & 1) || !last.current) return
          if (e.shiftKey && anchor.current) {
            const locked = lockAxis(anchor.current, p, axis.current, 4 / k)
            p = locked.p
            axis.current = locked.axis
          } else {
            axis.current = null
          }
          segment(maskCtx(e.altKey), last.current, p, radius(), shape, MASK)
          last.current = p
          schedule()
        }}
        onPointerUp={(e) => {
          if (tool === 'rect' && rectStart.current) {
            const [a, , b] = outline
            if (a && b && Math.abs(b.x - a.x) * scale.current > 2 && Math.abs(b.y - a.y) * scale.current > 2) fillOutline(outline, e.altKey)
            rectStart.current = null
            setOutline([])
            return
          }
          if (last.current) lastEnd.current = last.current
          last.current = null
        }}
        onPointerLeave={() => setCursor(null)}
      />
      {/* контур прямоугольника / многоугольника — в тех же координатах, что и холст (object-contain) */}
      {outline.length > 0 && (
        <svg
          viewBox={`0 0 ${size2.w} ${size2.h}`} preserveAspectRatio="xMidYMid meet"
          className="pointer-events-none absolute inset-0 size-full" style={imageStyle}
        >
          <path
            d={`M${[...outline, ...(rubber && tool === 'poly' ? [rubber] : [])].map((q) => `${q.x},${q.y}`).join('L')}Z`}
            fill={restoring ? '#ffffff' : '#f59e0b'} fillOpacity={0.18} stroke="#fff" strokeWidth={1.5}
            strokeDasharray="6 6" vectorEffect="non-scaling-stroke" className="animate-ants"
          />
          {tool === 'poly' && outline.map((q, n) => (
            <circle key={n} cx={q.x} cy={q.y} r={(n ? 3 : 5) / scale.current} fill={n ? '#fff' : '#f59e0b'} stroke="#000" strokeOpacity={0.4} vectorEffect="non-scaling-stroke" />
          ))}
        </svg>
      )}
      {/* в body: предки с transform (анимации motion) сбили бы position: fixed */}
      {cursor && tool === 'brush' && createPortal(
        <div
          aria-hidden
          className={`pointer-events-none fixed z-50 border-2 shadow-[0_0_0_1px_rgba(0,0,0,0.45)] ${shape === 'circle' ? 'rounded-full' : 'rounded-[3px]'} ${restoring ? 'border-white' : 'border-[#f59e0b]'}`}
          style={{ left: cursor.x - size / 2, top: cursor.y - size / 2, width: size, height: size }}
        />,
        document.body,
      )}
    </>
  )
})
