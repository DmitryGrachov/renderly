/** Кадр по материалам (Render pass): /api/materials */

export interface MaterialGroup {
  id: number
  label: string
  /** доля кадра 0…1 */
  area: number
  /** средний цвет материала на кадре (#rrggbb) */
  color: string
}

export interface MaterialMap {
  /** размер карты (кадр уменьшен до 2048 по длинной стороне) */
  width: number
  height: number
  /** id группы на пиксель, 0 — без материала */
  ids: Uint8Array
  /** плоская схема материалов (Qwen-Image-2.1, без света), по ней искались группы */
  schema: string | null
  groups: MaterialGroup[]
}

async function decode(url: string, width: number, height: number): Promise<Uint8Array> {
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
  const out = new Uint8Array(width * height)
  for (let i = 0; i < out.length; i++) out[i] = px[i * 4]
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
  return { width: j.width, height: j.height, ids: await decode(j.map, j.width, j.height), schema: j.schema ?? null, groups: j.groups }
}

/** Маска выбранных групп в размер исходного кадра: PNG, белое — группа */
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

/** Палитра групп: каждый следующий цвет — самый далёкий от уже взятых (оттенок × насыщенность × светлота).
 * Раньше цвет шёл по кругу оттенков через золотое сечение, и у групп с разницей номеров в число Фибоначчи (4 и 25)
 * оттенки почти совпадали — штукатурка и ступени выглядели одной группой */
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

/** Контрастный цвет группы (номера — по убыванию площади: у крупных групп самые различимые цвета) */
export function groupColor(id: number): [number, number, number] {
  return PALETTE[(id - 1 + PALETTE.length) % PALETTE.length]
}

function hsl(h: number, s: number, l: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 12) % 12
    const a = s * Math.min(l, 1 - l)
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
  }
  return [f(0), f(8), f(4)]
}
