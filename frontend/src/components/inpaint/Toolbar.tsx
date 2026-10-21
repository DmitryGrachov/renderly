import { motion } from 'motion/react'
import { Brush, Eraser, Hand, History, Lasso, Redo2, Trash2, Undo2 } from 'lucide-react'
import { ToolButton } from '../ToolButton'
import type { InpaintMode } from '../../lib/inpaint'

export type Tool = 'lasso' | 'brush' | 'eraser' | 'restore' | 'hand'
export type Shape = 'circle' | 'square'

export const MODE_COLOR: Record<InpaintMode, string> = {
  edit: '#2f7bff',
  material: '#f59e0b',
  remove: '#ff3b5c',
}

const TOOLS: { id: Tool; icon: typeof Brush; label: string; key: string }[] = [
  { id: 'lasso', icon: Lasso, label: 'Лассо', key: 'L' },
  { id: 'brush', icon: Brush, label: 'Кисть', key: 'B' },
  { id: 'eraser', icon: Eraser, label: 'Ластик', key: 'E' },
  { id: 'restore', icon: History, label: 'Кисть отката — вернуть предыдущую версию там, где закрасите', key: 'R' },
  { id: 'hand', icon: Hand, label: 'Рука — перемещение', key: 'H / пробел' },
]

interface ToolbarProps {
  tool: Tool
  onTool: (t: Tool) => void
  canUndo: boolean
  canRedo: boolean
  hasMask: boolean
  onUndo: () => void
  onRedo: () => void
  onClear: () => void
  /** Есть предыдущая версия — кисть отката доступна */
  canRestore: boolean
  disabled?: boolean
}

export function Toolbar({ tool, onTool, canUndo, canRedo, hasMask, onUndo, onRedo, onClear, canRestore, disabled }: ToolbarProps) {
  return (
    <div className={`glass flex flex-col gap-1 rounded-[18px] p-1.5 shadow-float ring-1 ring-fg/5 ${disabled ? 'pointer-events-none opacity-60' : ''}`}>
      {TOOLS.map(({ id, icon: Icon, label, key }) => {
        const active = tool === id
        return (
          <ToolButton
            key={id}
            label={label}
            shortcut={key}
            onClick={() => onTool(id)}
            active={active}
            disabled={id === 'restore' && !canRestore}
          >
            {active && (
              <motion.span layoutId="inpaint-tool" className="absolute inset-0 -z-10 rounded-xl bg-ink" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />
            )}
            <Icon className="size-[18px]" strokeWidth={1.75} />
          </ToolButton>
        )
      })}
      <div className="mx-2 my-1 h-px bg-fg/10" />
      <ToolButton label="Отменить" shortcut="Ctrl Z" onClick={onUndo} disabled={!canUndo}>
        <Undo2 className="size-[18px]" strokeWidth={1.75} />
      </ToolButton>
      <ToolButton label="Повторить" shortcut="Ctrl ⇧ Z" onClick={onRedo} disabled={!canRedo}>
        <Redo2 className="size-[18px]" strokeWidth={1.75} />
      </ToolButton>
      <ToolButton label={tool === 'restore' ? 'Сбросить откат' : 'Очистить маску'} shortcut="Esc" onClick={onClear} disabled={!hasMask}>
        <Trash2 className="size-[18px]" strokeWidth={1.75} />
      </ToolButton>
    </div>
  )
}

interface ModeSwitchProps {
  mode: InpaintMode
  onMode: (m: InpaintMode) => void
  disabled?: boolean
}

export function ModeSwitch({ mode, onMode, disabled }: ModeSwitchProps) {
  const items: { id: InpaintMode; label: string }[] = [
    { id: 'edit', label: 'Добавить / изменить' },
    { id: 'material', label: 'Материалы' },
    { id: 'remove', label: 'Удалить' },
  ]
  return (
    <div className={`glass flex rounded-full p-1 shadow-float ring-1 ring-fg/5 ${disabled ? 'pointer-events-none opacity-60' : ''}`}>
      {items.map((it) => {
        const active = mode === it.id
        return (
          <button
            key={it.id}
            onClick={() => onMode(it.id)}
            className={`relative isolate flex items-center gap-2 rounded-full px-4 py-1.5 text-[13px] font-medium transition-colors duration-300 ${
              active ? 'text-on-ink' : 'text-ink-2 hover:text-ink'
            }`}
          >
            {active && (
              <motion.span layoutId="inpaint-mode" className="absolute inset-0 -z-10 rounded-full bg-ink" transition={{ type: 'spring', stiffness: 460, damping: 36 }} />
            )}
            <span className="size-2 rounded-full" style={{ background: MODE_COLOR[it.id], boxShadow: active ? `0 0 8px ${MODE_COLOR[it.id]}` : undefined }} />
            {it.label}
          </button>
        )
      })}
    </div>
  )
}

interface BrushOptionsProps {
  shape: Shape
  onShape: (s: Shape) => void
  size: number
  onSize: (n: number) => void
  min: number
  max: number
}

export function BrushOptions({ shape, onShape, size, onSize, min, max }: BrushOptionsProps) {
  return (
    <div className="glass flex items-center gap-3 rounded-full py-1.5 pl-1.5 pr-4 shadow-float ring-1 ring-fg/5">
      <div className="flex rounded-full bg-fg/[0.05] p-0.5">
        {(['circle', 'square'] as const).map((s) => (
          <button
            key={s}
            onClick={() => onShape(s)}
            aria-label={s === 'circle' ? 'Круглая кисть' : 'Квадратная кисть'}
            className={`grid size-8 place-items-center rounded-full transition-all duration-200 ${shape === s ? 'bg-pill shadow-sm' : 'opacity-50 hover:opacity-100'}`}
          >
            <span className={`size-3 border-[1.5px] border-ink ${s === 'circle' ? 'rounded-full' : 'rounded-[2px]'}`} />
          </button>
        ))}
      </div>
      <span className="text-[12px] font-medium text-mute">Размер</span>
      <input
        type="range" min={min} max={max} value={size}
        onChange={(e) => onSize(Number(e.target.value))}
        className="range w-28 sm:w-36"
        style={{ '--fill': `${((size - min) / (max - min)) * 100}%` } as React.CSSProperties}
        aria-label="Размер кисти"
      />
      <span className="w-9 text-right text-[13px] font-semibold tabular-nums">{size}</span>
    </div>
  )
}
