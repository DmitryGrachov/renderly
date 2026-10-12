import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, Monitor, Moon, Sun } from 'lucide-react'
import { useTheme, type ThemePref } from '../lib/theme'

const OPTIONS: { id: ThemePref; label: string; icon: typeof Sun }[] = [
  { id: 'light', label: 'Светлая', icon: Sun },
  { id: 'dark', label: 'Тёмная', icon: Moon },
  { id: 'system', label: 'Как в системе', icon: Monitor },
]

/** Круглая кнопка с текущей темой и выпадающим списком. */
export function ThemeMenu({ className = '' }: { className?: string }) {
  const { pref, theme, setPref } = useTheme()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => !ref.current?.contains(e.target as Node) && setOpen(false)
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('pointerdown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const Icon = theme === 'dark' ? Moon : Sun

  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Оформление"
        aria-expanded={open}
        title="Оформление"
        className="grid size-8 place-items-center rounded-full text-ink-2 transition-colors hover:bg-fg/[0.06] hover:text-ink"
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={theme}
            initial={{ opacity: 0, rotate: -60, scale: 0.6 }}
            animate={{ opacity: 1, rotate: 0, scale: 1 }}
            exit={{ opacity: 0, rotate: 60, scale: 0.6 }}
            transition={{ duration: 0.2 }}
            className="grid place-items-center"
          >
            <Icon className="size-[17px]" strokeWidth={1.9} />
          </motion.span>
        </AnimatePresence>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98, transition: { duration: 0.12 } }}
            transition={{ type: 'spring', stiffness: 460, damping: 32 }}
            style={{ transformOrigin: 'top right' }}
            className="absolute right-0 top-full z-50 mt-2.5 w-52 rounded-[18px] bg-surface/[0.92] p-1.5 shadow-float ring-1 ring-fg/[0.06] backdrop-blur-2xl"
            role="menu"
          >
            <div className="px-3 pb-1.5 pt-2 text-[12px] font-medium text-faint">Оформление</div>
            {OPTIONS.map(({ id, label, icon: OptIcon }) => (
              <button
                key={id}
                role="menuitemradio"
                aria-checked={pref === id}
                onClick={() => { setPref(id); setOpen(false) }}
                className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-[14px] text-ink-2 transition-colors hover:bg-fg/[0.05] hover:text-ink"
              >
                <OptIcon className="size-4" strokeWidth={1.9} />
                <span className="flex-1">{label}</span>
                {pref === id && <Check className="size-4 text-accent" strokeWidth={2.5} />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
