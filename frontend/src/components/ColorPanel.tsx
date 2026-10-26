import { Brush, Eye, EyeOff, Minus, Pentagon, Plus, RotateCcw, SquareDashed, SunMedium, Undo2 } from 'lucide-react'
import type { DrawTool } from '../lib/brush'
import type { RegionMode } from './RegionBrush'
import { NEUTRAL, OFFSET_MAX, isNeutral, type ColorValues } from '../lib/colorAdjust'

const SLIDERS: { key: keyof ColorValues; label: string }[] = [
  { key: 'brightness', label: 'Яркость' },
  { key: 'contrast', label: 'Контраст' },
  { key: 'saturation', label: 'Насыщенность' },
]

/** Цвет сгенерированного куска: яркость, контраст, насыщенность (−100…100) только в области изменений. */
export function ColorPanel({
  values, onChange, showRegion, onShowRegion, onDone, working, regionMode, onRegionMode, brushSize, onBrushSize, undoable, onUndo,
  regionTool, onRegionTool, regionHardness, onRegionHardness, regionOffset, onRegionOffset, compact,
}: {
  values: ColorValues
  onChange: (v: ColorValues) => void
  showRegion: boolean
  onShowRegion: (v: boolean) => void
  onDone: () => void
  /** пересборка идёт — значок у заголовка */
  working?: boolean
  /** кисть области: добавить / убрать; null — не рисуем */
  regionMode: RegionMode | null
  onRegionMode: (m: RegionMode | null) => void
  brushSize: number
  onBrushSize: (n: number) => void
  undoable: number
  onUndo: () => void
  /** чем править область: кисть, прямоугольник, многоугольник (прямые границы) */
  regionTool: DrawTool
  onRegionTool: (t: DrawTool) => void
  /** жёсткость края всей области 0…1 */
  regionHardness: number
  onRegionHardness: (v: number) => void
  /** сдвиг края всей области, px кадра: > 0 — расширить, < 0 — поджать */
  regionOffset: number
  onRegionOffset: (v: number) => void
  /** только ползунки и «Сбросить»: область правится инструментами на кадре (Img to Img, «Материалы») */
  compact?: boolean
}) {
  return (
    <div className="glass w-[min(92vw,340px)] rounded-[22px] p-3.5 shadow-float ring-1 ring-fg/5">
      <div className="mb-2 flex items-center gap-2 text-[13px] font-semibold">
        <SunMedium className={`size-4 ${working ? 'animate-spin text-accent' : 'text-ink-2'}`} />
        Цвет сгенерированного
        <span className="ml-auto text-[11.5px] font-normal text-faint">только где изменилось</span>
      </div>
      <div className="flex flex-col gap-2">
        {SLIDERS.map(({ key, label }) => {
          const v = values[key]
          // заливка от середины: влево — меньше, вправо — больше
          const from = Math.min(50, 50 + v / 2)
          const to = Math.max(50, 50 + v / 2)
          return (
            <label key={key} className="flex items-center gap-3 text-[12.5px]">
              <span className="w-[92px] shrink-0 text-ink-2">{label}</span>
              <input
                type="range" min={-100} max={100} step={1} value={v}
                onChange={(e) => onChange({ ...values, [key]: Number(e.target.value) })}
                onDoubleClick={() => onChange({ ...values, [key]: 0 })}
                aria-label={label}
                className="range-center w-full"
                style={{ '--from': `${from}%`, '--to': `${to}%` } as React.CSSProperties}
              />
              <span className="w-9 shrink-0 text-right font-semibold tabular-nums">{v > 0 ? `+${v}` : v}</span>
            </label>
          )
        })}
      </div>
      {compact && (
        <div className="mt-2.5 flex justify-end">
          <button
            onClick={() => onChange(NEUTRAL)}
            disabled={isNeutral(values)}
            className="flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[12px] font-medium text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink disabled:opacity-40"
          >
            <RotateCcw className="size-3.5" /> Сбросить
          </button>
        </div>
      )}
      {/* область: показать и поправить кистью */}
      {!compact && <>
      <div className="mt-3 border-t border-line pt-2.5">
        <div className="flex items-center gap-1.5">
          <span className="mr-auto text-[12.5px] text-ink-2">Область</span>
          {(['add', 'remove'] as const).map((m) => (
            <button
              key={m}
              onClick={() => onRegionMode(regionMode === m ? null : m)}
              title={m === 'add' ? 'Кисть: добавить в область' : 'Кисть: убрать из области'}
              className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors ${
                regionMode === m ? 'bg-ink text-on-ink' : 'text-ink-2 ring-1 ring-fg/[0.08] hover:bg-fg/[0.05] hover:text-ink'
              }`}
            >
              {m === 'add' ? <Plus className="size-3.5" /> : <Minus className="size-3.5" />}
              {m === 'add' ? 'Добавить' : 'Убрать'}
            </button>
          ))}
          <button
            onClick={onUndo}
            disabled={!undoable}
            title="Отменить мазок области"
            className="grid size-7 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink disabled:opacity-35"
          >
            <Undo2 className="size-3.5" />
          </button>
        </div>
        {regionMode && (
          <div className="mt-2 flex items-center gap-1">
            <span className="w-[92px] shrink-0 text-[12px] text-mute">Инструмент</span>
            {([
              ['brush', Brush, 'Кисть — Shift+клик: прямая от конца мазка'],
              ['rect', SquareDashed, 'Прямоугольник — протяните; Shift — квадрат'],
              ['poly', Pentagon, 'Многоугольник — точки кликами; Enter или клик в первую точку'],
            ] as const).map(([t, Icon, title]) => (
              <button
                key={t}
                onClick={() => onRegionTool(t)}
                title={title}
                className={`grid size-8 place-items-center rounded-full transition-colors ${regionTool === t ? 'bg-ink text-on-ink' : 'text-ink-2 hover:bg-fg/[0.06] hover:text-ink'}`}
              >
                <Icon className="size-4" strokeWidth={1.75} />
              </button>
            ))}
          </div>
        )}
        <label className="mt-2 flex items-center gap-3 text-[12px] text-mute">
          <span className="w-[92px] shrink-0" title="Граница области: жёсткая — прямые края без перехода, мягкая — плавный переход">Край</span>
          <input
            type="range" min={0} max={1} step={0.05} value={regionHardness}
            onChange={(e) => onRegionHardness(Number(e.target.value))}
            className="range w-full"
            style={{ '--fill': `${regionHardness * 100}%` } as React.CSSProperties}
            aria-label="Жёсткость края области"
          />
          <span className="w-14 shrink-0 text-right text-ink-2">{regionHardness > 0.8 ? 'жёсткий' : regionHardness < 0.3 ? 'мягкий' : 'средний'}</span>
        </label>
        <label className="mt-2 flex items-center gap-3 text-[12px] text-mute">
          <span className="w-[92px] shrink-0" title="Поджать (влево) или расширить (вправо) всю область на столько пикселей">Сдвиг края</span>
          <input
            type="range" min={-OFFSET_MAX} max={OFFSET_MAX} step={1} value={regionOffset}
            onChange={(e) => onRegionOffset(Number(e.target.value))}
            onDoubleClick={() => onRegionOffset(0)}
            className="range-center w-full"
            style={{ '--from': `${Math.min(50, 50 + (regionOffset / OFFSET_MAX) * 50)}%`, '--to': `${Math.max(50, 50 + (regionOffset / OFFSET_MAX) * 50)}%` } as React.CSSProperties}
            aria-label="Сдвиг края области"
          />
          <span className="w-14 shrink-0 text-right tabular-nums text-ink-2">{regionOffset > 0 ? `+${regionOffset}` : regionOffset} px</span>
        </label>
        {regionMode && regionTool === 'brush' && (
          <label className="mt-2 flex items-center gap-3 text-[12px] text-mute">
            <span className="w-[92px] shrink-0">Кисть</span>
            <input
              type="range" min={10} max={300} step={2} value={brushSize}
              onChange={(e) => onBrushSize(Number(e.target.value))}
              className="range w-full"
              style={{ '--fill': `${((brushSize - 10) / 290) * 100}%` } as React.CSSProperties}
              aria-label="Размер кисти области"
            />
            <span className="w-9 shrink-0 text-right tabular-nums text-ink-2">{brushSize}</span>
          </label>
        )}
      </div>
      <div className="mt-2.5 flex items-center gap-1.5">
        <button
          onClick={() => onShowRegion(!showRegion)}
          className={`flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[12px] font-medium transition-colors ${showRegion ? 'bg-fg/[0.08] text-ink' : 'text-mute hover:bg-fg/[0.05] hover:text-ink'}`}
          title="Подсветить, где действуют ползунки"
        >
          {showRegion ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />} Показать
        </button>
        <button
          onClick={() => onChange(NEUTRAL)}
          disabled={isNeutral(values)}
          className="flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[12px] font-medium text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink disabled:opacity-40"
        >
          <RotateCcw className="size-3.5" /> Сбросить
        </button>
        <button
          onClick={onDone}
          className="ml-auto rounded-full bg-ink px-3.5 py-1.5 text-[12.5px] font-medium text-on-ink transition-colors hover:bg-ink-hover"
        >
          Готово
        </button>
      </div>
      </>}
    </div>
  )
}
