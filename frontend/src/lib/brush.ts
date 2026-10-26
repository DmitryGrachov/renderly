/*
 * Общее для кистей Inpainting и «Материалов»: прямые с Shift, прямоугольник, многоугольник, мягкий край.
 * Координаты — в пикселях холста, на котором рисуем (маска, мазки отката).
 */

export interface Pt {
  x: number
  y: number
}

export type Shape = 'circle' | 'square'

/** Инструменты рисования: кисть, прямоугольник (протянуть), многоугольник (точки кликами, стороны прямые) */
export type DrawTool = 'brush' | 'rect' | 'poly'

/** Жёсткость края по умолчанию: доля радиуса без спада (кисти отката; маска правки — всегда жёсткая) */
export const HARDNESS_DEFAULT = 0.55

/** Shift в мазке — строго по горизонтали или вертикали: ось — по первому заметному движению от начала мазка. */
export function lockAxis(anchor: Pt, p: Pt, axis: 'x' | 'y' | null, minMove: number): { p: Pt; axis: 'x' | 'y' | null } {
  let a = axis
  if (!a) {
    const dx = Math.abs(p.x - anchor.x)
    const dy = Math.abs(p.y - anchor.y)
    if (Math.max(dx, dy) < minMove) return { p: anchor, axis: null }
    a = dx >= dy ? 'x' : 'y'
  }
  return { p: a === 'x' ? { x: p.x, y: anchor.y } : { x: anchor.x, y: p.y }, axis: a }
}

/** Отпечаток кисти: круг или квадрат радиуса r; hardness < 1 — мягкий спад к краю (только круг). */
export function dab(c: CanvasRenderingContext2D, x: number, y: number, r: number, shape: Shape, color: string, hardness = 1) {
  if (shape === 'square') {
    c.fillStyle = color
    c.fillRect(x - r, y - r, r * 2, r * 2)
    return
  }
  if (hardness >= 0.99) {
    c.fillStyle = color
  } else {
    const g = c.createRadialGradient(x, y, r * hardness, x, y, r)
    g.addColorStop(0, color)
    g.addColorStop(1, transparent(color))
    c.fillStyle = g
  }
  c.beginPath()
  c.arc(x, y, r, 0, Math.PI * 2)
  c.fill()
}

/** Мазок от a до b — дорожка отпечатков через четверть радиуса: при быстром движении без пропусков. */
export function segment(
  c: CanvasRenderingContext2D, a: Pt, b: Pt, r: number, shape: Shape, color: string, hardness = 1,
) {
  const n = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / Math.max(0.5, r / 4)))
  for (let i = 0; i <= n; i++) dab(c, a.x + ((b.x - a.x) * i) / n, a.y + ((b.y - a.y) * i) / n, r, shape, color, hardness)
}

/** Залить многоугольник (или прямоугольник по двум углам); feather — размытие края в пикселях холста. */
export function fillPolygon(c: CanvasRenderingContext2D, pts: Pt[], color: string, feather = 0) {
  if (pts.length < 3) return
  c.save()
  if (feather > 0.5) c.filter = `blur(${feather}px)`
  c.fillStyle = color
  c.beginPath()
  pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)))
  c.closePath()
  c.fill()
  c.restore()
}

export const rectPoints = (a: Pt, b: Pt): Pt[] => [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }]

/** SVG-путь для превью контура (лассо, прямоугольник, многоугольник с «резинкой» до курсора) */
export const pathOf = (pts: Pt[], close = false) =>
  pts.length ? `M${pts.map((p) => `${p.x},${p.y}`).join('L')}${close ? 'Z' : ''}` : ''

function transparent(color: string) {
  // #rrggbb / #rgb / rgb() → тот же цвет с нулевой альфой (градиент без серого ореола)
  if (color.startsWith('#')) {
    const h = color.length === 4 ? color.slice(1).split('').map((ch) => ch + ch).join('') : color.slice(1, 7)
    return `#${h}00`
  }
  return color.replace(/rgba?\(([^,]+),([^,]+),([^,)]+).*\)/, 'rgba($1,$2,$3,0)')
}
