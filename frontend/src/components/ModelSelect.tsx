import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, ChevronDown, Sparkles, Zap } from 'lucide-react'
import { MODELS, fmt, modelById, type ModelId } from '../lib/account'

interface Props {
  value: ModelId
  onChange: (m: ModelId) => void
  disabled?: boolean
}

function ModelIcon({ id, size = 36 }: { id: ModelId; size?: number }) {
  return (
    <span
      className="grid shrink-0 place-items-center rounded-[11px] text-white shadow-soft"
      style={{
        width: size,
        height: size,
        background: `linear-gradient(140deg, var(--color-m-${id}), color-mix(in oklab, var(--color-m-${id}) 55%, #000))`,
      }}
    >
      <Sparkles className="size-4" strokeWidth={1.9} />
    </span>
  )
}

/** Шкала из пяти делений — скорость/качество. */
function Pips({ value }: { value: number }) {
  return (
    <span className="flex gap-[3px]">
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`h-[3px] w-2.5 rounded-full ${i <= value ? 'bg-ink' : 'bg-fg/[0.1]'}`} />
      ))}
    </span>
  )
}

/** Выпадающий выбор модели генерации. */
export function ModelSelect({ value, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false)
  const [focus, setFocus] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const items = useRef<(HTMLButtonElement | null)[]>([])
  const current = modelById(value)

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => !ref.current?.contains(e.target as Node) && setOpen(false)
    window.addEventListener('pointerdown', onDown)
    return () => window.removeEventListener('pointerdown', onDown)
  }, [open])

  useEffect(() => {
    if (disabled) setOpen(false)
  }, [disabled])

  useEffect(() => {
    if (open) items.current[focus]?.focus()
  }, [open, focus])

  const toggle = () => {
    setFocus(Math.max(0, MODELS.findIndex((m) => m.id === value)))
    setOpen((o) => !o)
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') { setOpen(false); return }
    if (!open) return
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      setFocus((f) => (f + (e.key === 'ArrowDown' ? 1 : MODELS.length - 1)) % MODELS.length)
    }
  }

  return (
    <div ref={ref} className="relative mt-4" onKeyDown={onKey}>
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={`flex w-full items-center gap-3 rounded-2xl bg-canvas p-2.5 pr-3.5 text-left ring-1 transition-all duration-300 ease-apple disabled:opacity-60 ${
          open ? 'ring-fg/[0.14]' : 'ring-transparent hover:ring-fg/[0.08]'
        }`}
      >
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.span
            key={current.id}
            initial={{ opacity: 0, scale: 0.6, rotate: -20 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
          >
            <ModelIcon id={current.id} size={40} />
          </motion.span>
        </AnimatePresence>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-[15px] font-semibold tracking-[-0.01em]">{current.name}</span>
            <span className="shrink-0 text-[12px] text-faint">{current.tag}</span>
          </span>
          <span className="mt-0.5 flex items-center gap-1 text-[12.5px] text-mute">
            <Zap className="size-3" strokeWidth={2.2} />
            {current.time} · {fmt(current.cost)} ток. за рендер
          </span>
        </span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ type: 'spring', stiffness: 400, damping: 30 }} className="text-mute">
          <ChevronDown className="size-[18px]" />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            role="listbox"
            aria-label="Модель"
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98, transition: { duration: 0.12 } }}
            transition={{ type: 'spring', stiffness: 460, damping: 32 }}
            style={{ transformOrigin: 'top center' }}
            className="absolute inset-x-0 top-full z-40 mt-2 rounded-[20px] bg-surface/[0.94] p-1.5 shadow-float ring-1 ring-fg/[0.06] backdrop-blur-2xl"
          >
            {MODELS.map((m, i) => {
              const active = m.id === value
              return (
                <motion.button
                  key={m.id}
                  ref={(el) => { items.current[i] = el }}
                  type="button"
                  role="option"
                  aria-selected={active}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.03 * i, duration: 0.2 }}
                  onClick={() => { onChange(m.id); setOpen(false) }}
                  onMouseEnter={() => setFocus(i)}
                  className={`flex w-full items-start gap-3 rounded-[14px] p-2.5 text-left outline-none transition-colors duration-150 ${
                    focus === i ? 'bg-fg/[0.05]' : ''
                  }`}
                >
                  <ModelIcon id={m.id} />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className="truncate text-[14px] font-semibold">{m.name}</span>
                      {m.badge && <span className="shrink-0 rounded-full bg-fg/[0.06] px-1.5 py-px text-[10.5px] font-semibold text-ink-2">{m.badge}</span>}
                    </span>
                    <span className="mt-0.5 block text-[12.5px] leading-snug text-mute">{m.description}</span>
                    <span className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-faint">
                      <span className="flex items-center gap-1.5">Скорость <Pips value={m.speed} /></span>
                      <span className="flex items-center gap-1.5">Качество <Pips value={m.quality} /></span>
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1.5">
                    <span className="text-[13px] font-semibold tabular-nums">{fmt(m.cost)} <span className="font-normal text-faint">ток.</span></span>
                    <span className={`grid size-5 place-items-center rounded-full bg-ink text-on-ink transition-transform duration-300 ease-apple ${active ? 'scale-100' : 'scale-0'}`}>
                      <Check className="size-3" strokeWidth={3} />
                    </span>
                  </span>
                </motion.button>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
