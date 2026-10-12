import type { PresetId } from './presets'
import type { Scene } from './scene'

/**
 * Клиентская «имитация» рендера: цветокоррекция на canvas.
 * Нужна только для демо, пока не подключён ComfyUI.
 */

const MAX_SIDE = 2560

interface Look {
  filter: string
  overlays: Array<{ mode: GlobalCompositeOperation; alpha: number; paint: (ctx: CanvasRenderingContext2D, w: number, h: number) => void }>
}

const linear = (ctx: CanvasRenderingContext2D, w: number, h: number, stops: [number, string][], angle: 'v' | 'd' = 'v') => {
  const g = angle === 'v' ? ctx.createLinearGradient(0, 0, 0, h) : ctx.createLinearGradient(w, 0, 0, h)
  stops.forEach(([o, c]) => g.addColorStop(o, c))
  ctx.fillStyle = g
  ctx.fillRect(0, 0, w, h)
}

const radial = (ctx: CanvasRenderingContext2D, w: number, h: number, cx: number, cy: number, stops: [number, string][]) => {
  const r = Math.max(w, h)
  const g = ctx.createRadialGradient(cx * w, cy * h, 0, cx * w, cy * h, r)
  stops.forEach(([o, c]) => g.addColorStop(o, c))
  ctx.fillStyle = g
  ctx.fillRect(0, 0, w, h)
}

const LOOKS: Record<Exclude<PresetId, 'custom'>, Look> = {
  natural: {
    filter: 'brightness(1.05) contrast(1.04) saturate(1.04)',
    overlays: [
      { mode: 'soft-light', alpha: 0.35, paint: (c, w, h) => linear(c, w, h, [[0, '#ffffff'], [1, '#e9e4dc']]) },
    ],
  },
  day: {
    filter: 'brightness(1.1) contrast(1.12) saturate(1.2)',
    overlays: [
      { mode: 'soft-light', alpha: 0.4, paint: (c, w, h) => linear(c, w, h, [[0, '#7fb6ff'], [0.6, '#ffffff'], [1, '#fff4dc']]) },
      { mode: 'screen', alpha: 0.12, paint: (c, w, h) => radial(c, w, h, 0.8, 0.1, [[0, '#fffbe8'], [0.5, 'rgba(255,255,255,0)']]) },
    ],
  },
  golden: {
    filter: 'brightness(1.04) contrast(1.08) saturate(1.25) sepia(0.18)',
    overlays: [
      { mode: 'soft-light', alpha: 0.7, paint: (c, w, h) => linear(c, w, h, [[0, '#ff9a4a'], [1, '#ffd08a']], 'd') },
      { mode: 'screen', alpha: 0.28, paint: (c, w, h) => radial(c, w, h, 0.15, 0.75, [[0, '#ffcf7a'], [0.55, 'rgba(255,160,80,0)']]) },
    ],
  },
  evening: {
    filter: 'brightness(0.78) contrast(1.12) saturate(0.92)',
    overlays: [
      { mode: 'multiply', alpha: 0.45, paint: (c, w, h) => linear(c, w, h, [[0, '#2b3a78'], [1, '#6b6fa8']]) },
      { mode: 'screen', alpha: 0.22, paint: (c, w, h) => linear(c, w, h, [[0, 'rgba(0,0,0,0)'], [0.65, 'rgba(0,0,0,0)'], [1, '#ff9a5a']]) },
    ],
  },
}

// Интерьер: вечером свет даёт не небо, а тёплые лампы
const INTERIOR_LOOKS: Partial<typeof LOOKS> = {
  evening: {
    filter: 'brightness(0.82) contrast(1.1) saturate(1.05) sepia(0.12)',
    overlays: [
      { mode: 'multiply', alpha: 0.35, paint: (c, w, h) => linear(c, w, h, [[0, '#3a3560'], [1, '#6b5a70']]) },
      { mode: 'screen', alpha: 0.35, paint: (c, w, h) => radial(c, w, h, 0.25, 0.35, [[0, '#ffb760'], [0.35, 'rgba(255,160,80,0)']]) },
      { mode: 'screen', alpha: 0.3, paint: (c, w, h) => radial(c, w, h, 0.8, 0.45, [[0, '#ffc27a'], [0.3, 'rgba(255,160,80,0)']]) },
    ],
  },
  day: {
    filter: 'brightness(1.08) contrast(1.12) saturate(1.1)',
    overlays: [
      { mode: 'soft-light', alpha: 0.35, paint: (c, w, h) => linear(c, w, h, [[0, '#ffffff'], [1, '#fff1d6']], 'd') },
      { mode: 'screen', alpha: 0.18, paint: (c, w, h) => radial(c, w, h, 0.3, 0.2, [[0, '#fffbe8'], [0.45, 'rgba(255,255,255,0)']]) },
    ],
  },
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

async function averageColor(src: string): Promise<string> {
  const img = await loadImage(src)
  const c = document.createElement('canvas')
  c.width = c.height = 12
  const ctx = c.getContext('2d')!
  ctx.drawImage(img, 0, 0, 12, 12)
  const d = ctx.getImageData(0, 0, 12, 12).data
  let r = 0, g = 0, b = 0
  for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2] }
  const n = d.length / 4
  return `rgb(${Math.round(r / n)}, ${Math.round(g / n)}, ${Math.round(b / n)})`
}

export async function gradeImage(src: string, scene: Scene, preset: PresetId, strength: number, referenceSrc?: string): Promise<Blob> {
  const img = await loadImage(src)
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight))
  const w = Math.round(img.naturalWidth * scale)
  const h = Math.round(img.naturalHeight * scale)

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!

  let look: Look
  if (preset === 'custom') {
    const tint = referenceSrc ? await averageColor(referenceSrc) : '#cccccc'
    look = {
      filter: 'contrast(1.08) saturate(1.1)',
      overlays: [{ mode: 'soft-light', alpha: 0.75, paint: (c) => { c.fillStyle = tint; c.fillRect(0, 0, w, h) } }],
    }
  } else {
    look = (scene === 'interior' && INTERIOR_LOOKS[preset]) || LOOKS[preset]
  }

  // базовый слой
  ctx.drawImage(img, 0, 0, w, h)
  // стилизованный слой, смешиваем по силе
  ctx.globalAlpha = strength
  ctx.filter = look.filter
  ctx.drawImage(img, 0, 0, w, h)
  ctx.filter = 'none'
  for (const o of look.overlays) {
    ctx.globalCompositeOperation = o.mode
    ctx.globalAlpha = o.alpha * strength
    o.paint(ctx, w, h)
  }
  ctx.globalCompositeOperation = 'source-over'
  ctx.globalAlpha = 1

  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', 0.94),
  )
}
