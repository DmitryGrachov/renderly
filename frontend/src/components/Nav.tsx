import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Building2, LogOut, Sofa } from 'lucide-react'
import { useAuth } from '../lib/auth'
import { ThemeMenu } from './ThemeMenu'
import { SCENE_LABEL, useScene, type Scene } from '../lib/scene'

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
    <header className="glass-nav sticky top-0 z-50 border-b border-fg/[0.06]">
      <div className="mx-auto flex h-14 max-w-[1280px] items-center justify-between px-5 sm:px-8">
        <a href="#/img2img" className="group flex items-center gap-2.5" aria-label="INEGO.dev">
          <Logo />
          <span className="hidden text-[17px] font-semibold tracking-[-0.02em] sm:inline">
            INEGO<span className="text-faint">.dev</span>
          </span>
        </a>

        <nav className="relative flex rounded-full bg-fg/[0.05] p-1" role="tablist">
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
                    className="absolute inset-0 -z-10 rounded-full bg-pill shadow-[0_1px_2px_rgba(0,0,0,0.06),0_2px_8px_rgba(0,0,0,0.06)]"
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

        <div className="flex items-center gap-3 sm:gap-4">
          <SceneSwitch />
          <div className="flex items-center gap-2">
            <ThemeMenu />
            <UserMenu />
          </div>
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
      <rect width="64" height="64" rx="16" fill="var(--logo-bg)" />
      <circle cx="32" cy="32" r="14" fill="url(#logo-g)" />
    </svg>
  )
}

const SCENES: { id: Scene; icon: typeof Sofa }[] = [
  { id: 'exterior', icon: Building2 },
  { id: 'interior', icon: Sofa },
]

function SceneSwitch() {
  const { scene, setScene } = useScene()
  return (
    <div className="relative flex rounded-full p-1 ring-1 ring-fg/[0.08]" role="radiogroup" aria-label="Тип сцены">
      {SCENES.map(({ id, icon: Icon }) => {
        const active = scene === id
        return (
          <button
            key={id}
            role="radio"
            aria-checked={active}
            aria-label={SCENE_LABEL[id]}
            title={SCENE_LABEL[id]}
            onClick={() => setScene(id)}
            className={`relative isolate flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[13px] font-medium transition-colors duration-300 md:px-3.5 ${
              active ? 'text-on-ink' : 'text-mute hover:text-ink'
            }`}
          >
            {active && (
              <motion.span
                layoutId="scene-pill"
                className="absolute inset-0 -z-10 rounded-full bg-ink"
                transition={{ type: 'spring', stiffness: 420, damping: 36 }}
              />
            )}
            <Icon className="size-[15px]" strokeWidth={1.9} />
            <span className="hidden md:inline">{SCENE_LABEL[id]}</span>
          </button>
        )
      })}
    </div>
  )
}

function UserMenu() {
  const { user, logout } = useAuth()
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

  if (!user) return null
  const initials = user.name.split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase()

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Меню аккаунта"
        aria-expanded={open}
        className="grid size-8 place-items-center rounded-full text-[12px] font-semibold text-white ring-2 ring-canvas transition-transform duration-200 hover:scale-105 active:scale-95"
        style={{ background: 'linear-gradient(135deg,#ff8a3d,#ff4f7b 50%,#6a5cff)' }}
      >
        {initials}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98, transition: { duration: 0.12 } }}
            transition={{ type: 'spring', stiffness: 460, damping: 32 }}
            style={{ transformOrigin: 'top right' }}
            className="absolute right-0 top-full mt-2.5 w-64 rounded-[18px] bg-surface/[0.92] p-1.5 shadow-float ring-1 ring-fg/[0.06] backdrop-blur-2xl"
          >
            <div className="px-3 pb-2.5 pt-2">
              <div className="truncate text-[14px] font-semibold">{user.name}</div>
              <div className="truncate text-[12.5px] text-mute">{user.email}</div>
            </div>
            <div className="mx-2 h-px bg-fg/[0.07]" />
            <button
              onClick={() => { setOpen(false); logout() }}
              className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-[14px] text-ink-2 transition-colors hover:bg-fg/[0.05] hover:text-ink"
            >
              <LogOut className="size-4" /> Выйти
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
