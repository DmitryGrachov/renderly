import { gradeImage } from './grade'
import type { PresetId } from './presets'
import type { Scene } from './scene'

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'

export interface Img2ImgRequest {
  image: File
  imageUrl: string
  scene: Scene
  preset: PresetId
  strength: number
  /** Материалы от архитектора; пустая строка — не применяются */
  materials?: string
  /** true — небо с исходника, false — по референсу */
  keepSky?: boolean
  /** true — фон (всё позади здания) с исходника, false — по референсу */
  keepBg?: boolean
  /** true — окружение (земля, озеленение, люди перед зданием) с исходника, false — по референсу */
  keepSur?: boolean
  /** Модель на бэкенде: klein-4b | klein-9b */
  model?: string
  /** Модель DeepSeek: deepseek-flash | deepseek-v4-pro */
  llm?: string
  /** Сид генерации; не задан — случайный */
  seed?: number | null
  reference?: File | null
  referenceUrl?: string
}

export type ProgressFn = (progress: number, stage: string) => void

export const STAGES = ['Анализ геометрии', 'Построение карты глубины', 'Расчёт освещения', 'Финальный рендер'] as const

export interface Img2ImgResult {
  /** object URL готового изображения */
  url: string
  /** сид, с которым сделан рендер (для повтора и «добивания» той же картинки) */
  seed: number
}

export async function img2img(req: Img2ImgRequest, onProgress: ProgressFn, signal?: AbortSignal): Promise<Img2ImgResult> {
  return USE_MOCK ? mockImg2Img(req, onProgress, signal) : realImg2Img(req, onProgress, signal)
}

async function mockImg2Img(req: Img2ImgRequest, onProgress: ProgressFn, signal?: AbortSignal) {
  const duration = 4200
  const start = performance.now()
  await new Promise<void>((resolve, reject) => {
    const tick = () => {
      if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'))
      const t = Math.min(1, (performance.now() - start) / duration)
      const eased = 1 - Math.pow(1 - t, 2.2)
      onProgress(eased, STAGES[Math.min(STAGES.length - 1, Math.floor(t * STAGES.length))])
      t < 1 ? setTimeout(tick, 40) : resolve()
    }
    tick()
  })
  const blob = await gradeImage(req.imageUrl, req.scene, req.preset, req.strength, req.referenceUrl)
  return { url: URL.createObjectURL(blob), seed: req.seed ?? Math.floor(Math.random() * 1e9) }
}

async function realImg2Img(req: Img2ImgRequest, onProgress: ProgressFn, signal?: AbortSignal) {
  const body = new FormData()
  body.append('image', req.image)
  body.append('scene', req.scene)
  body.append('preset', req.preset)
  body.append('strength', String(req.strength))
  if (req.materials?.trim()) body.append('materials', req.materials.trim())
  body.append('keep_sky', String(!!req.keepSky))
  body.append('keep_bg', String(!!req.keepBg))
  body.append('keep_sur', String(!!req.keepSur))
  if (req.model) body.append('model', req.model)
  if (req.llm) body.append('llm', req.llm)
  if (req.seed != null) body.append('seed', String(req.seed))
  if (req.reference) body.append('reference', req.reference)

  // Бэкенд не стримит прогресс: плавно ведём шкалу к 95% за ~типичное время рендера
  // Qwen-Image-2.1: рендер Turbo + доводка базовой моделью — ~2 мин (с зонами ~2.7 мин); Turbo — ~55 с
  const expected = req.model === 'qwen-image-2.1' ? 125000 : 30000
  const start = performance.now()
  const timer = setInterval(() => {
    const t = (performance.now() - start) / expected
    const p = 0.95 * (1 - Math.exp(-2.2 * t))
    onProgress(p, STAGES[Math.min(STAGES.length - 1, Math.floor((p / 0.95) * STAGES.length))])
  }, 200)
  try {
    const res = await fetch('/api/img2img', { method: 'POST', body, signal })
    if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
    if (!res.ok) {
      const detail = await res.json().then((j) => j.detail as string).catch(() => '')
      throw new Error(detail || `Сервер ответил ${res.status}`)
    }
    const blob = await res.blob()
    onProgress(1, STAGES[STAGES.length - 1])
    return { url: URL.createObjectURL(blob), seed: Number(res.headers.get('X-Seed')) }
  } finally {
    clearInterval(timer)
  }
}
