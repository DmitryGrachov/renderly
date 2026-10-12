import { motion } from 'motion/react'

export type Tab = 'img2img' | 'inpainting'

const TABS: { id: Tab; label: string }[] = [
  { id: 'img2img', label: 'Img to Img' },
  { id: 'inpainting', label: 'Inpainting' },
]

interface Props {
  tab: Tab
  onChange: (tab: Tab) => void
  inpaintBadge?: boolean
}

export function Nav({ tab, onChange, inpaintBadge }: Props) {
  return (
    <header className="glass-nav sticky top-0 z-50 border-b border-black/[0.06]">
      <div className="mx-auto flex h-14 max-w-[1280px] items-center justify-between px-5 sm:px-8">
        <a href="#/img2img" className="group flex items-center gap-2.5" aria-label="INEGO.dev">
          <Logo />
          <span className="text-[17px] font-semibold tracking-[-0.02em]">
            INEGO<span className="text-faint">.dev</span>
          </span>
        </a>

        <nav className="relative flex rounded-full bg-black/[0.05] p-1" role="tablist">
          {TABS.map((t) => {
            const active = t.id === tab
            return (
              <button
                key={t.id}
                role="tab"
                aria-selected={active}
                onClick={() => onChange(t.id)}
                className={`relative z-10 rounded-full px-4 py-1.5 text-[13px] font-medium transition-colors duration-300 sm:px-5 ${
                  active ? 'text-ink' : 'text-mute hover:text-ink'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="tab-pill"
                    className="absolute inset-0 -z-10 rounded-full bg-white shadow-[0_1px_2px_rgba(0,0,0,0.06),0_2px_8px_rgba(0,0,0,0.06)]"
                    transition={{ type: 'spring', stiffness: 420, damping: 36 }}
                  />
                )}
                {t.label}
                {t.id === 'inpainting' && inpaintBadge && !active && (
                  <span className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-accent" />
                )}
              </button>
            )
          })}
        </nav>

        <div className="hidden items-center gap-2 text-[12px] text-mute sm:flex">
          <span className="relative flex size-2">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-60" />
            <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
          </span>
          ComfyUI · demo
        </div>
      </div>
    </header>
  )
}

export function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" className="transition-transform duration-500 ease-apple group-hover:rotate-[20deg]">
      <defs>
        <linearGradient id="logo-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffb36b" />
          <stop offset=".5" stopColor="#ff6a88" />
          <stop offset="1" stopColor="#6a5cff" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill="#1d1d1f" />
      <circle cx="32" cy="32" r="14" fill="url(#logo-g)" />
    </svg>
  )
}
