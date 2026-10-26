import { Eye, EyeOff, RotateCcw, SunMedium } from 'lucide-react'
import { NEUTRAL, isNeutral, type ColorValues } from '../lib/colorAdjust'

const SLIDERS: { key: keyof ColorValues; label: string }[] = [
  { key: 'brightness', label: 'Яркость' },
  { key: 'contrast', label: 'Контраст' },
  { key: 'saturation', label: 'Насыщенность' },
]

/** Цвет сгенерированного куска: яркость, контраст, насыщенность (−100…100) только в области изменений. */
export function ColorPanel({ values, onChange, showRegion, onShowRegion, onDone, working }: {
  values: ColorValues
  onChange: (v: ColorValues) => void
  showRegion: boolean
  onShowRegion: (v: boolean) => void
  onDone: () => void
  /** пересборка идёт — значок у заголовка */
  working?: boolean
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
      <div className="mt-3 flex items-center gap-1.5">
        <button
          onClick={() => onShowRegion(!showRegion)}
          className={`flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[12px] font-medium transition-colors ${showRegion ? 'bg-fg/[0.08] text-ink' : 'text-mute hover:bg-fg/[0.05] hover:text-ink'}`}
          title="Подсветить, где действуют ползунки"
        >
          {showRegion ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />} Область
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
    </div>
  )
}
