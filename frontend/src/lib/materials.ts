/** Render pass: /api/materials — маски по схеме Qwen-Image-Edit-2511: область схемы — маска, линии — разделитель */

export interface MaterialGroup {
  id: number
  label: string
  /** доля кадра 0…1 */
  area: number
  /** средний цвет области на схеме (#rrggbb) */
  color: string
}

/** id разделителя — линий схемы между областями */
export const DIVIDER_ID = 1

export interface MaterialMap {
  /** размер карты (кадр уменьшен до 2048 по длинной стороне) */
  width: number
  height: number
  /** id маски на пиксель: 1 — разделитель, 2… — области */
  ids: Uint16Array
  /** плоская иллюстрация 2511 на месте кадра (PNG data URL), по ней режутся маски */
  schema: string
  groups: MaterialGroup[]
}

/** Карта — RGB PNG: id = R + 256·G (областей бывает больше 255) */
async function decode(url: string, width: number, height: number): Promise<Uint16Array> {
  const img = new Image()
  img.src = url
  await img.decode()
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(img, 0, 0)
  const px = ctx.getImageData(0, 0, width, height).data
  const out = new Uint16Array(width * height)
  for (let i = 0; i < out.length; i++) out[i] = px[i * 4] | (px[i * 4 + 1] << 8)
  return out
}

export async function buildMaterials(file: File, signal?: AbortSignal): Promise<MaterialMap> {
  const body = new FormData()
  body.append('image', file)
  const res = await fetch('/api/materials', { method: 'POST', body, signal })
  if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
  if (res.status === 413) throw new Error('Файл слишком большой')
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail as string).catch(() => '')
    throw new Error(detail || `Сервер ответил ${res.status}`)
  }
  const j = await res.json()
  return { width: j.width, height: j.height, ids: await decode(j.map, j.width, j.height), schema: j.schema, groups: j.groups }
}

/** Маска выбранных областей (1 — выбрано, в размере карты) для Inpainting: области плюс линии между ними —
 * иначе у двух выбранных соседей посередине оставалась бы щель по разделителю. Пиксель линии берётся, если по
 * горизонтали или вертикали в пределах r px по обе его стороны (сквозь линию) лежит выбранное; наружный контур
 * выбранного и линия у края кадра не берутся */
export function pickMask(map: MaterialMap, ids: Set<number>, r = 3): Uint8Array {
  const { width: w, height: h, ids: px } = map
  const sel = new Uint8Array(w * h)
  for (let i = 0; i < px.length; i++) if (ids.has(px[i])) sel[i] = 1
  if (!ids.size) return sel
  // первое за линией в направлении шага: выбрано ли (не дальше r px, не за краем кадра)
  const side = (i: number, x: number, y: number, dx: number, dy: number) => {
    for (let k = 1; k <= r; k++) {
      const nx = x + dx * k
      const ny = y + dy * k
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) return false
      const j = i + (dy * w + dx) * k
      if (px[j] !== DIVIDER_ID) return sel[j] === 1
    }
    return false
  }
  const out = sel.slice()
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (px[i] !== DIVIDER_ID) continue
      if ((side(i, x, y, -1, 0) && side(i, x, y, 1, 0)) || (side(i, x, y, 0, -1) && side(i, x, y, 0, 1))) out[i] = 1
    }
  }
  return out
}

/** Маска выбранных областей в размер исходного кадра: PNG, белое — выбранное */
export async function maskPng(map: MaterialMap, ids: Set<number>, width: number, height: number): Promise<Blob> {
  const c = document.createElement('canvas')
  c.width = map.width
  c.height = map.height
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(map.width, map.height)
  for (let i = 0; i < map.ids.length; i++) {
    const v = ids.has(map.ids[i]) ? 255 : 0
    img.data.set([v, v, v, 255], i * 4)
  }
  ctx.putImageData(img, 0, 0)
  const out = document.createElement('canvas')
  out.width = width
  out.height = height
  const octx = out.getContext('2d')!
  octx.imageSmoothingEnabled = false
  octx.drawImage(c, 0, 0, width, height)
  return new Promise((ok, fail) => out.toBlob((b) => (b ? ok(b) : fail(new Error('Маска не собралась'))), 'image/png'))
}

/** Палитра областей: каждый следующий цвет — самый далёкий от уже взятых (оттенок × насыщенность × светлота).
 * Раньше цвет шёл по кругу оттенков через золотое сечение, и у областей с разницей номеров в число Фибоначчи
 * (4 и 25) оттенки почти совпадали — соседние элементы выглядели одним */
const PALETTE: [number, number, number][] = (() => {
  const cands: [number, number, number][] = []
  for (let h = 0; h < 36; h++) for (const s of [0.65, 0.9]) for (const l of [0.42, 0.56, 0.7]) cands.push(hsl(h / 36, s, l))
  // «redmean» — дешёвое приближение воспринимаемой разницы цветов
  const dist = (a: number[], b: number[]) => {
    const r = (a[0] + b[0]) / 2
    const dr = a[0] - b[0], dg = a[1] - b[1], db = a[2] - b[2]
    return (2 + r / 256) * dr * dr + 4 * dg * dg + (2 + (255 - r) / 256) * db * db
  }
  const out = [cands[0]]
  const near = cands.map((c) => dist(c, cands[0]))
  while (out.length < cands.length) {
    let best = 0
    for (let i = 1; i < cands.length; i++) if (near[i] > near[best]) best = i
    out.push(cands[best])
    for (let i = 0; i < cands.length; i++) near[i] = Math.min(near[i], dist(cands[i], cands[best]))
  }
  return out
})()

/** Контрастный цвет области (номера — по убыванию площади: у крупных самые различимые цвета); разделитель — чёрный */
export function groupColor(id: number): [number, number, number] {
  if (id === DIVIDER_ID) return [12, 12, 14]
  return PALETTE[(id - 2 + PALETTE.length) % PALETTE.length]
}

function hsl(h: number, s: number, l: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 12) % 12
    const a = s * Math.min(l, 1 - l)
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
  }
  return [f(0), f(8), f(4)]
}
