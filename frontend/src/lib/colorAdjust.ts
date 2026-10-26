/*
 * Цвет после генерации: яркость, контраст, насыщенность — только там, где картинка изменилась.
 * Область — разница результата с кадром «до» (мягкая маска), сам цвет — фильтры canvas по этой маске.
 */

export interface ColorValues {
  /** −100…100, 0 — как есть */
  brightness: number
  contrast: number
  saturation: number
}

export const NEUTRAL: ColorValues = { brightness: 0, contrast: 0, saturation: 0 }
export const isNeutral = (v: ColorValues) => !v.brightness && !v.contrast && !v.saturation

const SIDE = 768 // разница считается на кадре не больше этого — быстро и без шума отдельных пикселей
// пиксель изменился: разница цвета (сумма по каналам, 0…765) после лёгкого размытия — двойной порог, как у Canny.
// Больше STRONG — точно правка (модели перерисовывают весь кадр, свет вне правки плывёт на единицы-десятки, новый
// материал — на сотни); больше WEAK — правка, если касается сильной: тёмно-серая стена в тени, ставшая тёмным
// кирпичом, меняется всего на 36–60 и одним порогом выпадала. При WEAK 20 область уже стекала на дорожку
const STRONG = 60
const WEAK = 30
const CLOSE = 0.012 // залить дырки и убрать крошки: размытие на эту долю стороны кадра и порог
const FEATHER = 0.004 // мягкий край области — доля длинной стороны итогового кадра

export function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

const canvas = (w: number, h: number) => Object.assign(document.createElement('canvas'), { width: w, height: h })

/** Где результат отличается от кадра «до»: маска (альфа 0…255) в размере результата, с мягким краем. */
export async function changedRegion(beforeUrl: string, afterUrl: string): Promise<HTMLCanvasElement> {
  const [before, after] = await Promise.all([loadImage(beforeUrl), loadImage(afterUrl)])
  const W = after.naturalWidth
  const H = after.naturalHeight
  const k = Math.min(1, SIDE / Math.max(W, H))
  const w = Math.max(1, Math.round(W * k))
  const h = Math.max(1, Math.round(H * k))
  const read = (img: HTMLImageElement) => {
    const c = canvas(w, h).getContext('2d', { willReadFrequently: true })!
    c.filter = 'blur(1px)'
    c.drawImage(img, 0, 0, w, h)
    return c.getImageData(0, 0, w, h).data
  }
  const a = read(before)
  const b = read(after)
  const diff = canvas(w, h)
  const dc = diff.getContext('2d', { willReadFrequently: true })!
  const d = new Uint16Array(w * h)
  for (let i = 0, p = 0; i < a.length; i += 4, p++) d[p] = Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])
  // сильные — затравки, слабые доливаются от них по соседям
  const m = new Uint8Array(w * h)
  const stack: number[] = []
  for (let p = 0; p < d.length; p++) if (d[p] > STRONG) { m[p] = 1; stack.push(p) }
  while (stack.length) {
    const p = stack.pop()!
    const x = p % w
    for (const n of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, p - w, p + w]) {
      if (n >= 0 && n < m.length && !m[n] && d[n] > WEAK) { m[n] = 1; stack.push(n) }
    }
  }
  const out = dc.createImageData(w, h)
  for (let p = 0; p < m.length; p++) out.data[p * 4 + 3] = m[p] * 255
  dc.putImageData(out, 0, 0)
  // закрытие: размытие и порог посередине — дырки внутри области заливаются, одиночные крошки пропадают
  const closed = canvas(w, h)
  const cc = closed.getContext('2d', { willReadFrequently: true })!
  cc.filter = `blur(${Math.max(1, CLOSE * Math.max(w, h))}px)`
  cc.drawImage(diff, 0, 0)
  const px = cc.getImageData(0, 0, w, h)
  for (let i = 3; i < px.data.length; i += 4) px.data[i] = px.data[i] > 110 ? 255 : 0
  cc.filter = 'none'
  cc.putImageData(px, 0, 0)
  // в размер результата с мягким краем
  const region = canvas(W, H)
  const rc = region.getContext('2d')!
  rc.filter = `blur(${Math.max(1, FEATHER * Math.max(W, H))}px)`
  rc.imageSmoothingQuality = 'high'
  rc.drawImage(closed, 0, 0, W, H)
  return region
}

/** CSS-фильтр canvas для значений ползунков */
export const filterOf = (v: ColorValues) =>
  `brightness(${1 + v.brightness / 200}) contrast(${1 + v.contrast / 200}) saturate(${1 + v.saturation / 100})`

/** Картинка с цветом, поправленным только в области → object URL (JPEG). */
export async function colorize(baseUrl: string, region: HTMLCanvasElement, v: ColorValues): Promise<string> {
  const base = await loadImage(baseUrl)
  const W = base.naturalWidth
  const H = base.naturalHeight
  const out = canvas(W, H)
  const o = out.getContext('2d')!
  o.drawImage(base, 0, 0)
  if (!isNeutral(v)) {
    const layer = canvas(W, H)
    const l = layer.getContext('2d')!
    l.filter = filterOf(v)
    l.drawImage(base, 0, 0)
    l.filter = 'none'
    l.globalCompositeOperation = 'destination-in'
    l.drawImage(region, 0, 0, W, H)
    o.drawImage(layer, 0, 0)
  }
  const blob = await new Promise<Blob>((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/jpeg', 0.95))
  return URL.createObjectURL(blob)
}

/** Подсветка области для «Показать область»: полупрозрачная заливка → object URL (PNG). */
export async function regionOverlay(region: HTMLCanvasElement, color = '#ff2bd6'): Promise<string> {
  const c = canvas(region.width, region.height)
  const x = c.getContext('2d')!
  x.fillStyle = color
  x.globalAlpha = 0.45
  x.fillRect(0, 0, c.width, c.height)
  x.globalAlpha = 1
  x.globalCompositeOperation = 'destination-in'
  x.drawImage(region, 0, 0)
  const blob = await new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/png'))
  return URL.createObjectURL(blob)
}
