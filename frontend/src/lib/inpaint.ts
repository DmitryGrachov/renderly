import { loadImage } from './grade'

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'

export type InpaintMode = 'edit' | 'remove'

export interface InpaintRequest {
  imageUrl: string
  /** Маска: любой непрозрачный пиксель = область для перерисовки */
  mask: HTMLCanvasElement
  mode: InpaintMode
  prompt?: string
}

export type ProgressFn = (progress: number, stage: string) => void

export const INPAINT_STAGES: Record<InpaintMode, string[]> = {
  remove: ['Анализ области', 'Восстановление фона', 'Сведение'],
  edit: ['Анализ области', 'Генерация', 'Сведение'],
}

/** Возвращает object URL нового изображения. */
export function inpaint(req: InpaintRequest, onProgress: ProgressFn, signal?: AbortSignal): Promise<string> {
  return USE_MOCK ? mockInpaint(req, onProgress, signal) : realInpaint(req, onProgress, signal)
}

/** Ч/б маска в PNG (белое — перерисовать), в размере исходника. */
export function exportMask(mask: HTMLCanvasElement, width: number, height: number): Promise<Blob> {
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const ctx = c.getContext('2d')!
  ctx.drawImage(mask, 0, 0, width, height)
  ctx.globalCompositeOperation = 'source-in'
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, width, height)
  ctx.globalCompositeOperation = 'destination-over'
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, width, height)
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('mask export failed'))), 'image/png'))
}

async function realInpaint(req: InpaintRequest, onProgress: ProgressFn, signal?: AbortSignal) {
  const stages = INPAINT_STAGES[req.mode]
  const img = await loadImage(req.imageUrl)
  const body = new FormData()
  body.append('image', await fetch(req.imageUrl).then((r) => r.blob()), 'image.png')
  body.append('mask', await exportMask(req.mask, img.naturalWidth, img.naturalHeight), 'mask.png')
  body.append('mode', req.mode)
  if (req.prompt) body.append('prompt', req.prompt)

  onProgress(0.05, stages[0])
  const res = await fetch('/api/inpaint', { method: 'POST', body, signal })
  if (!res.ok) throw new Error(`Сервер ответил ${res.status}`)
  onProgress(1, stages[stages.length - 1])
  return URL.createObjectURL(await res.blob())
}

/* ------------------------------------------------------------------ */
/* Мок: «заливаем» область окружающими цветами, чтобы демо выглядело живо */

async function mockInpaint(req: InpaintRequest, onProgress: ProgressFn, signal?: AbortSignal) {
  const stages = INPAINT_STAGES[req.mode]
  const duration = 3200
  const start = performance.now()
  await new Promise<void>((resolve, reject) => {
    const tick = () => {
      if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'))
      const t = Math.min(1, (performance.now() - start) / duration)
      onProgress(1 - Math.pow(1 - t, 2), stages[Math.min(stages.length - 1, Math.floor(t * stages.length))])
      t < 1 ? setTimeout(tick, 40) : resolve()
    }
    tick()
  })

  const img = await loadImage(req.imageUrl)
  const w = img.naturalWidth
  const h = img.naturalHeight
  const canvas = (cw = w, ch = h) => {
    const c = document.createElement('canvas')
    c.width = cw
    c.height = ch
    return [c, c.getContext('2d')!] as const
  }

  // мягкая маска
  const [feather, fctx] = canvas()
  fctx.filter = `blur(${Math.max(2, w / 400)}px)`
  fctx.drawImage(req.mask, 0, 0, w, h)
  fctx.drawImage(req.mask, 0, 0, w, h)

  // изображение с «дыркой»
  const [holed, hctx] = canvas()
  hctx.drawImage(img, 0, 0)
  hctx.globalCompositeOperation = 'destination-out'
  hctx.drawImage(req.mask, 0, 0, w, h)

  // заливка дырки окружением: от крупного размытия к мелкому
  const [fill, xctx] = canvas()
  for (const div of [12, 24, 48, 96]) {
    xctx.filter = `blur(${w / div}px)`
    for (let i = 0; i < 4; i++) xctx.drawImage(holed, 0, 0)
  }
  xctx.filter = 'none'

  if (req.mode === 'edit') {
    // лёгкий «новый» оттенок, зависящий от промпта
    const hue = [...(req.prompt ?? '')].reduce((a, ch) => a + ch.charCodeAt(0), 0) % 360
    xctx.globalCompositeOperation = 'soft-light'
    xctx.fillStyle = `hsl(${hue} 60% 50%)`
    xctx.fillRect(0, 0, w, h)
  }

  // зерно, чтобы заливка не выглядела пластиковой
  xctx.globalCompositeOperation = 'overlay'
  xctx.globalAlpha = 0.08
  xctx.drawImage(noise(256), 0, 0, w, h)
  xctx.globalAlpha = 1

  xctx.globalCompositeOperation = 'destination-in'
  xctx.drawImage(feather, 0, 0)

  const [out, octx] = canvas()
  octx.drawImage(img, 0, 0)
  octx.drawImage(fill, 0, 0)

  const blob = await new Promise<Blob>((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/jpeg', 0.94))
  return URL.createObjectURL(blob)
}

function noise(size: number) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d')!
  const data = ctx.createImageData(size, size)
  for (let i = 0; i < data.data.length; i += 4) {
    const v = Math.random() * 255
    data.data[i] = data.data[i + 1] = data.data[i + 2] = v
    data.data[i + 3] = 255
  }
  ctx.putImageData(data, 0, 0)
  return c
}
