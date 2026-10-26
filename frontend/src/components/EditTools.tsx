import { motion } from 'motion/react'
import {
  Brush, Check, Eraser, Eye, EyeOff, ImageIcon, Minus, Pentagon, Plus, SquareDashed, SunMedium, Trash2, Undo2, X,
} from 'lucide-react'
import { ToolButton } from './ToolButton'
import { BrushOptions } from './inpaint/Toolbar'
import { OFFSET_MAX } from '../lib/colorAdjust'
import type { DrawTool, Shape } from '../lib/brush'
import type { RegionMode } from './RegionBrush'

/* Правка результата на кадре, как в Photoshop: слева — режимы и инструменты режима, сверху — параметры режима. */

export type EditMode = 'view' | 'restore' | 'color'

const SHAPES: { id: DrawTool; icon: typeof Brush; label: string; key: string }[] = [
  { id: 'brush', icon: Brush, label: 'Кисть — Shift+клик: прямая от конца мазка, Shift в мазке: по горизонтали / вертикали', key: 'B' },
  { id: 'rect', icon: SquareDashed, label: 'Прямоугольник — протяните; Shift — квадрат', key: 'U' },
  { id: 'poly', icon: Pentagon, label: 'Многоугольник — точки кликами; Enter, двойной клик или клик в первую точку', key: 'P' },
]

const Active = ({ id }: { id: string }) => (
  <motion.span layoutId={id} className="absolute inset-0 -z-10 rounded-xl bg-ink" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />
)

/** Слева: режимы (результат / откат / цвет), под чертой — инструменты выбранного режима. */
export function ModeRail({
  mode, onMode, tool, onTool, undoable, onUndo, onClear, peek, onPeek, restoreEdited, colorEdited,
}: {
  mode: EditMode
  onMode: (m: EditMode) => void
  tool: DrawTool
  onTool: (t: DrawTool) => void
  undoable: number
  onUndo: () => void
  /** сбросить мазки отката / правки области */
  onClear?: () => void
  peek?: boolean
  onPeek?: (v: boolean) => void
  /** откат / цвет уже применены — точка у значка */
  restoreEdited?: boolean
  colorEdited?: boolean
}) {
  const modes: { id: EditMode; icon: typeof Brush; label: string; key: string; dot?: boolean }[] = [
    { id: 'view', icon: ImageIcon, label: 'Результат — просмотр и сравнение', key: 'Esc' },
    { id: 'restore', icon: Eraser, label: 'Откат — вернуть кадр до материала там, где закрасите', key: 'R', dot: restoreEdited },
    { id: 'color', icon: SunMedium, label: 'Цвет — яркость, контраст, насыщенность там, где изменилось', key: 'C', dot: colorEdited },
  ]
  return (
    <div className="glass flex flex-col gap-1 rounded-[18px] p-1.5 shadow-float ring-1 ring-fg/5">
      {modes.map(({ id, icon: Icon, label, key, dot }) => (
        <ToolButton key={id} label={label} shortcut={key} onClick={() => onMode(mode === id && id !== 'view' ? 'view' : id)} active={mode === id}>
          {mode === id && <Active id="edit-mode" />}
          <Icon className="size-[18px]" strokeWidth={1.75} />
          {dot && mode !== id && <span className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-accent" />}
        </ToolButton>
      ))}
      {mode !== 'view' && (
        <>
          <div className="mx-2 my-1 h-px bg-fg/10" />
          {SHAPES.map(({ id, icon: Icon, label, key }) => (
            <ToolButton key={id} label={label} shortcut={key} onClick={() => onTool(id)} active={tool === id}>
              {tool === id && <Active id="edit-tool" />}
              <Icon className="size-[18px]" strokeWidth={1.75} />
            </ToolButton>
          ))}
          {mode === 'restore' && onPeek && (
            <ToolButton label={peek ? 'Показать результат' : 'Показать кадр до материала целиком — вести границы по нему'} shortcut="V" onClick={() => onPeek(!peek)} active={peek}>
              {peek && <span className="absolute inset-0 -z-10 rounded-xl bg-[#f59e0b]" />}
              {peek ? <EyeOff className="size-[18px]" strokeWidth={1.75} /> : <Eye className="size-[18px]" strokeWidth={1.75} />}
            </ToolButton>
          )}
          <div className="mx-2 my-1 h-px bg-fg/10" />
          <ToolButton label="Отменить" shortcut="Ctrl Z" onClick={onUndo} disabled={undoable === 0}>
            <Undo2 className="size-[18px]" strokeWidth={1.75} />
          </ToolButton>
          {onClear && (
            <ToolButton label={mode === 'restore' ? 'Сбросить все мазки отката' : 'Сбросить правки области'} shortcut="" onClick={onClear}>
              <Trash2 className="size-[18px]" strokeWidth={1.75} />
            </ToolButton>
          )}
        </>
      )}
    </div>
  )
}

const Bar = ({ children }: { children: React.ReactNode }) => (
  <div className="glass flex max-w-full flex-wrap items-center justify-center gap-x-3 gap-y-1.5 rounded-[20px] py-1.5 pl-2 pr-1.5 shadow-float ring-1 ring-fg/5">
    {children}
  </div>
)

const Done = ({ onCancel, onDone, cancelLabel = 'Отмена' }: { onCancel?: () => void; onDone: () => void; cancelLabel?: string }) => (
  <div className="flex items-center gap-1">
    {onCancel && (
      <button onClick={onCancel} className="flex items-center gap-1 rounded-full px-3 py-1.5 text-[12.5px] font-medium text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink">
        <X className="size-3.5" /> {cancelLabel}
      </button>
    )}
    <button onClick={onDone} className="flex items-center gap-1.5 rounded-full bg-ink px-3.5 py-1.5 text-[12.5px] font-medium text-on-ink transition-colors hover:bg-ink-hover">
      <Check className="size-3.5" /> Готово
    </button>
  </div>
)

/** Сверху в режиме «Откат»: форма, размер, край, сдвиг прежнего кадра; отмена / готово. */
export function RestoreBar({
  tool, shape, onShape, size, onSize, hardness, onHardness, nudge, onResetNudge, onCancel, onDone,
}: {
  tool: DrawTool
  shape: Shape
  onShape: (s: Shape) => void
  size: number
  onSize: (n: number) => void
  hardness: number
  onHardness: (v: number) => void
  nudge: { x: number; y: number }
  onResetNudge: () => void
  onCancel: () => void
  onDone: () => void
}) {
  const shifted = nudge.x !== 0 || nudge.y !== 0
  return (
    <Bar>
      <span className="flex items-center gap-1.5 pl-1.5 text-[12.5px] font-medium text-ink-2"><Eraser className="size-3.5" /> Откат</span>
      {tool === 'brush' ? (
        <BrushOptions shape={shape} onShape={onShape} size={size} onSize={onSize} min={10} max={300} hardness={hardness} onHardness={onHardness} bare />
      ) : (
        <EdgeSlider label="Край" value={hardness} onChange={onHardness} />
      )}
      <span className="hidden text-[12px] text-faint lg:inline">Alt — вернуть результат · стрелки — сдвиг</span>
      {shifted && (
        <button onClick={onResetNudge} className="text-[12px] font-medium tabular-nums text-ink hover:text-accent" title="Сбросить сдвиг прежнего кадра">
          {nudge.x > 0 ? '→' : '←'}{Math.abs(nudge.x)} {nudge.y > 0 ? '↓' : '↑'}{Math.abs(nudge.y)} px
        </button>
      )}
      <Done onCancel={onCancel} onDone={onDone} />
    </Bar>
  )
}

/** Сверху в режиме «Цвет»: правка области — добавить / убрать, размер кисти, край, сдвиг края, показать; готово. */
export function RegionBar({
  tool, mode, onMode, size, onSize, hardness, onHardness, offset, onOffset, showRegion, onShowRegion, onDone,
}: {
  tool: DrawTool
  mode: RegionMode | null
  onMode: (m: RegionMode | null) => void
  size: number
  onSize: (n: number) => void
  hardness: number
  onHardness: (v: number) => void
  offset: number
  onOffset: (v: number) => void
  showRegion: boolean
  onShowRegion: (v: boolean) => void
  onDone: () => void
}) {
  return (
    <Bar>
      <span className="flex items-center gap-1.5 pl-1.5 text-[12.5px] font-medium text-ink-2"><SunMedium className="size-3.5" /> Область</span>
      <div className="flex rounded-full bg-fg/[0.05] p-0.5">
        {(['add', 'remove'] as const).map((m) => (
          <button
            key={m}
            onClick={() => onMode(mode === m ? null : m)}
            title={m === 'add' ? 'Рисовать — добавить в область' : 'Рисовать — убрать из области'}
            className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors ${mode === m ? 'bg-ink text-on-ink' : 'text-ink-2 hover:text-ink'}`}
          >
            {m === 'add' ? <Plus className="size-3.5" /> : <Minus className="size-3.5" />}
            {m === 'add' ? 'Добавить' : 'Убрать'}
          </button>
        ))}
      </div>
      {mode && tool === 'brush' && (
        <label className="flex items-center gap-2 text-[12px] text-mute">
          Кисть
          <input
            type="range" min={10} max={300} step={2} value={size} onChange={(e) => onSize(Number(e.target.value))}
            className="range w-20" style={{ '--fill': `${((size - 10) / 290) * 100}%` } as React.CSSProperties} aria-label="Размер кисти области"
          />
        </label>
      )}
      <EdgeSlider label="Край" value={hardness} onChange={onHardness} />
      <label className="flex items-center gap-2 text-[12px] text-mute" title="Поджать (влево) или расширить (вправо) всю область">
        Сдвиг
        <input
          type="range" min={-OFFSET_MAX} max={OFFSET_MAX} step={1} value={offset}
          onChange={(e) => onOffset(Number(e.target.value))} onDoubleClick={() => onOffset(0)}
          className="range-center w-20"
          style={{ '--from': `${Math.min(50, 50 + (offset / OFFSET_MAX) * 50)}%`, '--to': `${Math.max(50, 50 + (offset / OFFSET_MAX) * 50)}%` } as React.CSSProperties}
          aria-label="Сдвиг края области"
        />
        <span className="w-11 tabular-nums text-ink-2">{offset > 0 ? `+${offset}` : offset} px</span>
      </label>
      <button
        onClick={() => onShowRegion(!showRegion)}
        className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors ${showRegion ? 'bg-fg/[0.08] text-ink' : 'text-mute hover:bg-fg/[0.05] hover:text-ink'}`}
        title="Подсветить, где действует цвет"
      >
        {showRegion ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />} Показать
      </button>
      <Done onDone={onDone} />
    </Bar>
  )
}

function EdgeSlider({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <label className="flex items-center gap-2 text-[12px] text-mute" title="Жёсткость края: мягкий — плавный переход">
      {label}
      <input
        type="range" min={0} max={1} step={0.05} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="range w-20" style={{ '--fill': `${value * 100}%` } as React.CSSProperties} aria-label={`${label} — жёсткость`}
      />
    </label>
  )
}
