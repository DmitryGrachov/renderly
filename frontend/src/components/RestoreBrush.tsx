import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'

export interface RestoreBrushHandle {
  /** Отменить последний мазок */
  undo: () => void
  /** Стереть все мазки — снова весь результат */
  clear: () => void
  /** Итог: результат с прежним кадром под мазками, в размере результата */
  toBlob: () => Promise<Blob>
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
}

const HISTORY = 15 // снимков маски для отмены: кадр ~2 Мп — 8 МБ на снимок
const HARDNESS = 0.55 // доля радиуса без спада: край мазка мягкий, шов не виден

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

/** Кисть отката: под мазками вместо результата — прежний кадр. Холст на месте картинки (object-contain), рисует
 *  левая кнопка, Alt — возвращает результат; пробел и средняя кнопка двигают кадр (ZoomView). */
export const RestoreBrush = forwardRef<RestoreBrushHandle, Props>(function RestoreBrush(
  { before, after, imageStyle, size, onHistory, offset = { x: 0, y: 0 } },
  ref,
) {
  const view = useRef<HTMLCanvasElement>(null)
  const layers = useRef<{ after: HTMLImageElement; before: HTMLCanvasElement; mask: HTMLCanvasElement; tmp: HTMLCanvasElement } | null>(null)
  const history = useRef<ImageData[]>([])
  const last = useRef<{ x: number; y: number } | null>(null)
  const frame = useRef(0)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [restoring, setRestoring] = useState(true)
  const shift = useRef(offset)
  shift.current = offset

  useEffect(() => { render() }, [offset.x, offset.y]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let alive = true
    Promise.all([loadImage(after), loadImage(before)]).then(([a, b]) => {
      if (!alive || !view.current) return
      const { naturalWidth: w, naturalHeight: h } = a
      const canvas = (cw = w, ch = h) => Object.assign(document.createElement('canvas'), { width: cw, height: ch })
      // прежний кадр — в размере результата (бэкенд отдаёт результат в размере исходника, подогнанным по краям)
      const prev = canvas()
      prev.getContext('2d')!.drawImage(b, 0, 0, w, h)
      layers.current = { after: a, before: prev, mask: canvas(), tmp: canvas() }
      view.current.width = w
      view.current.height = h
      history.current = []
      onHistory?.(0)
      render()
    })
    return () => { alive = false; cancelAnimationFrame(frame.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [before, after])

  function render() {
    const l = layers.current
    const ctx = view.current?.getContext('2d')
    if (!l || !ctx) return
    // прежний кадр, вырезанный маской, поверх результата
    const t = l.tmp.getContext('2d')!
    t.globalCompositeOperation = 'copy'
    t.drawImage(l.before, shift.current.x, shift.current.y)
    t.globalCompositeOperation = 'destination-in'
    t.drawImage(l.mask, 0, 0)
    ctx.drawImage(l.after, 0, 0)
    ctx.drawImage(l.tmp, 0, 0)
  }

  function schedule() {
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(render)
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

  function dab(m: CanvasRenderingContext2D, x: number, y: number, r: number) {
    const g = m.createRadialGradient(x, y, r * HARDNESS, x, y, r)
    g.addColorStop(0, 'rgba(0,0,0,1)')
    g.addColorStop(1, 'rgba(0,0,0,0)')
    m.fillStyle = g
    m.beginPath()
    m.arc(x, y, r, 0, Math.PI * 2)
    m.fill()
  }

  function stroke(e: React.PointerEvent) {
    const l = layers.current
    if (!l) return
    const { x, y, k } = toImage(e)
    const r = size / 2 / k
    const m = l.mask.getContext('2d')!
    m.globalCompositeOperation = e.altKey ? 'destination-out' : 'source-over'
    const from = last.current ?? { x, y }
    // мазок — дорожка отпечатков через четверть радиуса: при быстром движении без пропусков
    const steps = Math.max(1, Math.ceil(Math.hypot(x - from.x, y - from.y) / (r / 4)))
    for (let i = 1; i <= steps; i++) dab(m, from.x + ((x - from.x) * i) / steps, from.y + ((y - from.y) * i) / steps, r)
    last.current = { x, y }
    schedule()
  }

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
    toBlob() {
      render()
      return new Promise<Blob>((resolve, reject) =>
        view.current!.toBlob((b) => (b ? resolve(b) : reject(new Error('Не удалось собрать кадр'))), 'image/jpeg', 0.95),
      )
    },
  }))

  return (
    <>
      <canvas
        ref={view}
        className="absolute inset-0 size-full cursor-none object-contain"
        style={imageStyle}
        onPointerDown={(e) => {
          if (e.button !== 0 || !layers.current) return
          e.currentTarget.setPointerCapture(e.pointerId)
          const m = layers.current.mask
          history.current.push(m.getContext('2d')!.getImageData(0, 0, m.width, m.height))
          if (history.current.length > HISTORY) history.current.shift()
          onHistory?.(history.current.length)
          last.current = null
          stroke(e)
        }}
        onPointerMove={(e) => {
          setCursor({ x: e.clientX, y: e.clientY })
          setRestoring(!e.altKey)
          if (e.buttons & 1 && last.current) stroke(e)
        }}
        onPointerUp={() => { last.current = null }}
        onPointerLeave={() => setCursor(null)}
      />
      {/* в body: предки с transform (анимации motion) сбили бы position: fixed */}
      {cursor && createPortal(
        <div
          aria-hidden
          className={`pointer-events-none fixed z-50 rounded-full border-2 shadow-[0_0_0_1px_rgba(0,0,0,0.45)] ${restoring ? 'border-white' : 'border-[#f59e0b]'}`}
          style={{ left: cursor.x - size / 2, top: cursor.y - size / 2, width: size, height: size }}
        />,
        document.body,
      )}
    </>
  )
})
