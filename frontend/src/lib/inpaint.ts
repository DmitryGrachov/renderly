import { loadImage } from './grade'
import type { Scene } from './scene'

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'

export type InpaintMode = 'edit' | 'material' | 'remove'

/** Тип правки по первому слову запроса (как на бэкенде, backend/app/inpaint.py: parse_kind); null — решает DeepSeek */
export type EditKind = 'add' | 'change'
const ADD_WORDS = ['добавь', 'добавить', 'поставь', 'поставить', 'посади', 'посадить', 'размести', 'разместить', 'нарисуй', 'нарисовать', 'установи', 'установить', 'add']
const CHANGE_WORDS = ['измени', 'изменить', 'замени', 'заменить', 'поменяй', 'поменять', 'перекрась', 'перекрасить', 'сделай', 'сделать', 'change', 'replace']

export function editKind(prompt: string): EditKind | null {
  const first = (prompt.trim().toLowerCase().replace(/ё/g, 'е').split(/\s+/)[0] ?? '').replace(/[,.:;!]+$/, '')
  if (ADD_WORDS.includes(first)) return 'add'
  if (CHANGE_WORDS.includes(first)) return 'change'
  return null
}

export interface InpaintRequest {
  imageUrl: string
  scene: Scene
  /** Маска: любой непрозрачный пиксель = область для перерисовки */
  mask: HTMLCanvasElement
  mode: InpaintMode
  prompt?: string
  /** Модель на бэкенде: klein-4b | klein-9b */
  model?: string
  /** Модель DeepSeek: deepseek-flash | deepseek-v4-pro */
  llm?: string
  /** Рамка объекта [x1, y1, x2, y2] в долях кадра, поправленная на канвасе (добавить / заменить) */
  box?: [number, number, number, number]
  /** Образец материала (вкладка «Материалы»): текстура, которую перетащили в окно */
  texture?: Blob
  /** Сколько сантиметров материала на образце поперёк; пусто — оценит бэкенд */
  textureWidthCm?: number
  /** Маска собрана по схеме («Выбор маски»): бэкенд доводит её край по фото; материал — ровно по ней, без запаса и SAM */
  exactMask?: boolean
}

export type ProgressFn = (progress: number, stage: string) => void

export const INPAINT_STAGES: Record<InpaintMode, string[]> = {
  remove: ['Анализ области', 'Восстановление фона', 'Сведение'],
  edit: ['Анализ области', 'Генерация', 'Сведение'],
  material: ['Анализ поверхности', 'Новый материал', 'Свет и сведение'],
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
  body.append('scene', req.scene)
  body.append('mode', req.mode)
  if (req.prompt) body.append('prompt', req.prompt)
  if (req.model) body.append('model', req.model)
  if (req.llm) body.append('llm', req.llm)
  if (req.exactMask) body.append('exact', 'true')
  if (req.box) body.append('box', req.box.map((v) => v.toFixed(4)).join(','))
  if (req.texture && req.mode === 'material') {
    body.append('texture', req.texture, 'texture')
    if (req.textureWidthCm) body.append('texture_width', String(req.textureWidthCm))
  }

  // Бэкенд не стримит прогресс: плавно ведём шкалу к 95% за ~типичное время удаления
  const expected = req.model === 'qwen-edit-2511' ? 60000 : req.model === 'qwen-image-2.1' ? 100000 : 12000
  const start = performance.now()
  const timer = setInterval(() => {
    const t = (performance.now() - start) / expected
    const p = 0.95 * (1 - Math.exp(-2.2 * t))
    onProgress(p, stages[Math.min(stages.length - 1, Math.floor((p / 0.95) * stages.length))])
  }, 200)
  try {
    const res = await fetch('/api/inpaint', { method: 'POST', body, signal })
    if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
    if (res.status === 413) throw new Error('Файлы слишком большие для отправки — уменьшите образец (например, 2K вместо 4K)')
    if (!res.ok) {
      const detail = await res.json().then((j) => j.detail as string).catch(() => '')
      throw new Error(detail || `Сервер ответил ${res.status}`)
    }
    const blob = await res.blob()
    onProgress(1, stages[stages.length - 1])
    return URL.createObjectURL(blob)
  } finally {
    clearInterval(timer)
  }
}

/** План правки до генерации: тип и рамка объекта (доли кадра) — показывается на канвасе, чтобы поправить.
 *  В моке и для «изменить» рамки нет — генерация идёт сразу. */
export async function planInpaint(req: InpaintRequest, signal?: AbortSignal): Promise<{ kind: string; box: [number, number, number, number] | null }> {
  if (USE_MOCK || req.mode !== 'edit' || !req.prompt) return { kind: 'change', box: null }
  const img = await loadImage(req.imageUrl)
  const body = new FormData()
  body.append('image', await fetch(req.imageUrl).then((r) => r.blob()), 'image.png')
  body.append('mask', await exportMask(req.mask, img.naturalWidth, img.naturalHeight), 'mask.png')
  body.append('prompt', req.prompt)
  if (req.llm) body.append('llm', req.llm)
  const res = await fetch('/api/inpaint/plan', { method: 'POST', body, signal })
  if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail as string).catch(() => '')
    throw new Error(detail || `Сервер ответил ${res.status}`)
  }
  return res.json()
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

  if (req.mode !== 'remove') {
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
