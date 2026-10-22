/** Карта зон кадра (Render pass): /api/zones */

export type ZoneKind = 'scene' | 'veg' | 'part' | 'finish'

export interface Zone {
  id: number
  label: string
  /** scene — сцена, veg — растительность, part — часть здания, finish — отделка стен */
  kind: ZoneKind
  /** буква отделки: одна буква — один материал на всех гранях */
  group: string | null
}

export interface ZoneMap {
  /** размер карты (кадр уменьшен до 2048 по длинной стороне) */
  width: number
  height: number
  /** id зоны на пиксель, 0 — без зоны */
  ids: Uint8Array
  /** мягкая альфа растительности 0…255 (ветки на фоне неба) */
  alpha: Uint8Array
  zones: Zone[]
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

export async function buildZones(file: File, signal?: AbortSignal): Promise<ZoneMap> {
  const body = new FormData()
  body.append('image', file)
  const res = await fetch('/api/zones', { method: 'POST', body, signal })
  if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
  if (res.status === 413) throw new Error('Файл слишком большой')
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail as string).catch(() => '')
    throw new Error(detail || `Сервер ответил ${res.status}`)
  }
  const j = await res.json()
  const [ids, alpha] = await Promise.all([decode(j.map, j.width, j.height), decode(j.alpha, j.width, j.height)])
  return { width: j.width, height: j.height, ids, alpha, zones: j.zones }
}

/** Цвет зоны: равномерно по кругу оттенков (золотое сечение), отделка одной группы — одним оттенком */
export function zoneColor(z: Zone): [number, number, number] {
  const key = z.kind === 'finish' && z.group ? 1000 + z.group.charCodeAt(0) : z.id
  const h = (key * 0.618033988749895) % 1
  return hsl(h, z.kind === 'finish' ? 0.55 : 0.7, z.kind === 'finish' ? 0.6 : 0.55)
}

function hsl(h: number, s: number, l: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 12) % 12
    const a = s * Math.min(l, 1 - l)
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
  }
  return [f(0), f(8), f(4)]
}
