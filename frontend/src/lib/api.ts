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
  reference?: File | null
  referenceUrl?: string
}

export type ProgressFn = (progress: number, stage: string) => void

export const STAGES = ['Анализ геометрии', 'Построение карты глубины', 'Расчёт освещения', 'Финальный рендер'] as const

/** Возвращает object URL готового изображения. */
export async function img2img(req: Img2ImgRequest, onProgress: ProgressFn, signal?: AbortSignal): Promise<string> {
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
  return URL.createObjectURL(blob)
}

async function realImg2Img(req: Img2ImgRequest, onProgress: ProgressFn, signal?: AbortSignal) {
  const body = new FormData()
  body.append('image', req.image)
  body.append('scene', req.scene)
  body.append('preset', req.preset)
  body.append('strength', String(req.strength))
  if (req.reference) body.append('reference', req.reference)

  onProgress(0.05, STAGES[0])
  const res = await fetch('/api/img2img', { method: 'POST', body, signal })
  if (res.status === 401) throw new Error('Сессия истекла — обновите страницу и войдите снова')
  if (!res.ok) throw new Error(`Сервер ответил ${res.status}`)
  onProgress(1, STAGES[STAGES.length - 1])
  return URL.createObjectURL(await res.blob())
}
