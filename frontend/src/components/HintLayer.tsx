import { useEffect, useRef, type CSSProperties } from 'react'

export interface HintStroke {
  /** образец (строка «Материалов»), к которому относится штрих */
  row: number
  color: string
  /** толщина в пикселях кадра */
  width: number
  points: [number, number][]
}

interface Props {
  /** размер кадра в пикселях — штрихи хранятся в его координатах */
  width: number
  height: number
  strokes: HintStroke[]
  /** масштаб и сдвиг ZoomView */
  imageStyle: CSSProperties
  /** рисовать за этот образец; null — только показывать */
  active: { row: number; color: string } | null
  onStroke: (s: HintStroke) => void
}

const PEN = 5 // толщина штриха в пикселях экрана: галочка, а не заливка

export function drawStrokes(ctx: CanvasRenderingContext2D, strokes: HintStroke[]) {
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const s of strokes) {
    ctx.strokeStyle = s.color
    ctx.lineWidth = s.width
    ctx.beginPath()
    s.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    if (s.points.length === 1) ctx.lineTo(s.points[0][0] + 0.1, s.points[0][1])
    ctx.stroke()
  }
}

/** Штрихи этого образца — прозрачный PNG в размере кадра (подсказка для DeepSeek); нет штрихов — null. */
export function strokesPng(strokes: HintStroke[], row: number, width: number, height: number): Promise<Blob | null> {
  const own = strokes.filter((s) => s.row === row)
  if (!own.length) return Promise.resolve(null)
  const c = Object.assign(document.createElement('canvas'), { width, height })
  drawStrokes(c.getContext('2d')!, own)
  return new Promise((resolve) => c.toBlob(resolve, 'image/png'))
}

/** Галочки и прочерки цветом образца поверх кадра: подсказка, какие поверхности имеются в виду, — не маска. */
export function HintLayer({ width, height, strokes, imageStyle, active, onStroke }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const current = useRef<HintStroke | null>(null)

  function redraw() {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, width, height)
    drawStrokes(ctx, current.current ? [...strokes, current.current] : strokes)
  }

  useEffect(redraw) // eslint-disable-line react-hooks/exhaustive-deps

  /** точка экрана → пиксели кадра (холст object-contain внутри своего прямоугольника после transform) */
  function toImage(e: React.PointerEvent): [number, number, number] {
    const r = ref.current!.getBoundingClientRect()
    const k = Math.min(r.width / width, r.height / height)
    return [(e.clientX - r.left - (r.width - width * k) / 2) / k, (e.clientY - r.top - (r.height - height * k) / 2) / k, k]
  }

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      className={`absolute inset-0 size-full object-contain ${active ? 'cursor-crosshair' : 'pointer-events-none'}`}
      style={imageStyle}
      onPointerDown={(e) => {
        if (!active || e.button !== 0) return
        e.currentTarget.setPointerCapture(e.pointerId)
        const [x, y, k] = toImage(e)
        current.current = { row: active.row, color: active.color, width: PEN / k, points: [[x, y]] }
        redraw()
      }}
      onPointerMove={(e) => {
        if (!current.current) return
        const [x, y] = toImage(e)
        current.current.points.push([x, y])
        redraw()
      }}
      onPointerUp={() => {
        if (current.current) onStroke(current.current)
        current.current = null
      }}
    />
  )
}
