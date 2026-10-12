import { createContext, useCallback, useContext, useEffect, useState } from 'react'

export type ThemePref = 'light' | 'dark' | 'system'
export type Theme = 'light' | 'dark'

// тот же ключ читает инлайн-скрипт в index.html (чтобы не было вспышки)
const STORAGE_KEY = 'inego.theme'
const media = () => window.matchMedia('(prefers-color-scheme: dark)')

const readPref = (): ThemePref => {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch { /* storage недоступен */ }
  return 'system'
}

const resolve = (pref: ThemePref): Theme => (pref === 'system' ? (media().matches ? 'dark' : 'light') : pref)

function apply(theme: Theme, animate: boolean) {
  const root = document.documentElement
  if (animate) {
    root.classList.add('theme-switching')
    window.setTimeout(() => root.classList.remove('theme-switching'), 400)
  }
  root.dataset.theme = theme
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#000000' : '#f5f5f7')
}

const ThemeContext = createContext<{ pref: ThemePref; theme: Theme; setPref: (p: ThemePref) => void } | null>(null)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [pref, setPrefState] = useState<ThemePref>(readPref)
  const [theme, setTheme] = useState<Theme>(() => resolve(readPref()))

  // первичное применение (инлайн-скрипт уже выставил атрибут, это подстраховка)
  useEffect(() => apply(resolve(readPref()), false), [])

  // следим за системной темой, если выбран «как в системе»
  useEffect(() => {
    if (pref !== 'system') return
    const m = media()
    const onChange = () => {
      const t = m.matches ? 'dark' : 'light'
      setTheme(t)
      apply(t, true)
    }
    m.addEventListener('change', onChange)
    return () => m.removeEventListener('change', onChange)
  }, [pref])

  const setPref = useCallback((p: ThemePref) => {
    try { localStorage.setItem(STORAGE_KEY, p) } catch { /* storage недоступен */ }
    const t = resolve(p)
    setPrefState(p)
    setTheme(t)
    apply(t, true)
  }, [])

  return <ThemeContext.Provider value={{ pref, theme, setPref }}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside ThemeProvider')
  return ctx
}
