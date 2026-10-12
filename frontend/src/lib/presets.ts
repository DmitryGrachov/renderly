export type PresetId = 'natural' | 'day' | 'golden' | 'evening' | 'custom'

export interface Preset {
  id: PresetId
  title: string
  caption: string
  meta: string
  /** CSS-фон превью: «небо» пресета */
  sky: string
  /** Положение и цвет «солнца» на превью */
  sun?: { x: string; y: string; color: string; size: number }
}

export const PRESETS: Preset[] = [
  {
    id: 'natural',
    title: 'Естественный свет',
    caption: 'Мягкий рассеянный свет, честные материалы',
    meta: '5600K · Overcast',
    sky: 'linear-gradient(180deg, #e9eef3 0%, #dfe5ec 45%, #f4f1ec 100%)',
    sun: { x: '50%', y: '34%', color: 'rgba(255,255,255,0.9)', size: 30 },
  },
  {
    id: 'day',
    title: 'День',
    caption: 'Яркое солнце, чистое небо, чёткие тени',
    meta: '6500K · Clear sky',
    sky: 'linear-gradient(180deg, #3a8dff 0%, #79b8ff 50%, #d8ecff 100%)',
    sun: { x: '72%', y: '24%', color: '#fffbe8', size: 34 },
  },
  {
    id: 'golden',
    title: 'Золотой час',
    caption: 'Тёплый низкий свет и длинные тени',
    meta: '3200K · Sunset',
    sky: 'linear-gradient(180deg, #ff9a5a 0%, #ffbf73 45%, #ffe2a8 100%)',
    sun: { x: '30%', y: '72%', color: '#fff2c4', size: 44 },
  },
  {
    id: 'evening',
    title: 'Вечер',
    caption: 'Синий час и тёплое свечение окон',
    meta: '2700K · Blue hour',
    sky: 'linear-gradient(180deg, #0f1a3d 0%, #2b3a78 50%, #c7739a 100%)',
    sun: { x: '78%', y: '22%', color: '#f4f1ff', size: 14 },
  },
]

export const CUSTOM_PRESET: Preset = {
  id: 'custom',
  title: 'Свой референс',
  caption: 'Загрузите фото — перенесём настроение света',
  meta: 'Style transfer',
  sky: 'conic-gradient(from 210deg at 50% 50%, #ff8a3d, #ff4f7b, #8f5cff, #2f7bff, #5ee7c4, #ff8a3d)',
}

export const ALL_PRESETS = [...PRESETS, CUSTOM_PRESET]
export const presetById = (id: PresetId) => ALL_PRESETS.find((p) => p.id === id)!
