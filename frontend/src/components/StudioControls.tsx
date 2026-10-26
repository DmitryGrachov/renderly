import { motion } from 'motion/react'

export function Step({ n, title, hint }: { n: number; title: string; hint: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-ink text-[12px] font-semibold text-on-ink">{n}</span>
      <div>
        <div className="text-[17px] font-semibold tracking-[-0.015em]">{title}</div>
        <div className="text-[13px] text-mute">{hint}</div>
      </div>
    </div>
  )
}

export function IconButton({ children, label, onClick, plain }: { children: React.ReactNode; label: string; onClick: () => void; plain?: boolean }) {
  return (
    <button
      aria-label={label}
      title={label}
      onClick={onClick}
      className={`grid size-9 place-items-center rounded-full text-ink-2 transition-all duration-300 ease-apple hover:text-ink active:scale-95 ${
        plain ? 'hover:bg-fg/[0.06]' : 'glass shadow-soft ring-1 ring-fg/5 hover:bg-surface'
      }`}
    >
      {children}
    </button>
  )
}

export function SecondaryButton({ children, onClick, disabled }: { children: React.ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <motion.button
      whileTap={disabled ? undefined : { scale: 0.97 }}
      onClick={onClick}
      disabled={disabled}
      className="flex flex-1 items-center justify-center gap-2 rounded-full bg-fg/[0.05] px-5 py-2.5 text-[14px] font-medium text-ink transition-colors hover:bg-fg/[0.08] disabled:opacity-40 sm:flex-none"
    >
      {children}
    </motion.button>
  )
}
