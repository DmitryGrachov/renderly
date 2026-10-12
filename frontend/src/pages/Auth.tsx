import { useEffect, useId, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Eye, EyeOff, Loader2 } from 'lucide-react'
import { Logo } from '../components/Nav'
import { presetsFor } from '../lib/presets'
import { AuthError, useAuth } from '../lib/auth'

type Mode = 'login' | 'register'

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

const COPY: Record<Mode, { title: string; subtitle: string; submit: string }> = {
  login: { title: 'С возвращением', subtitle: 'Войдите, чтобы продолжить работу над проектами.', submit: 'Войти' },
  register: { title: 'Создайте аккаунт', subtitle: 'Пара секунд — и можно пересвечивать рендеры.', submit: 'Создать аккаунт' },
}

export function AuthPage() {
  const { login, register } = useAuth()
  const [mode, setMode] = useState<Mode>('login')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<{ text: string; n: number } | null>(null)

  const errors = {
    name: mode === 'register' && !name.trim() ? 'Как к вам обращаться?' : null,
    email: !EMAIL_RE.test(email.trim()) ? 'Введите корректную почту' : null,
    password:
      mode === 'register' && password.length < 8
        ? 'Минимум 8 символов'
        : mode === 'login' && !password
          ? 'Введите пароль'
          : null,
  }
  const valid = !errors.name && !errors.email && !errors.password

  function switchMode(m: Mode) {
    setMode(m)
    setSubmitted(false)
    setError(null)
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitted(true)
    if (!valid || loading) return
    setLoading(true)
    setError(null)
    try {
      if (mode === 'login') await login(email.trim(), password)
      else await register(name.trim(), email.trim(), password)
    } catch (err) {
      const text = err instanceof AuthError ? err.message : 'Что-то пошло не так. Попробуйте ещё раз'
      setError((prev) => ({ text, n: (prev?.n ?? 0) + 1 }))
      setLoading(false)
    }
  }

  const copy = COPY[mode]

  return (
    <div className="min-h-dvh bg-canvas lg:grid lg:grid-cols-[1.1fr_1fr]">
      <Showcase />

      <main className="flex min-h-dvh items-center justify-center px-5 py-12 sm:px-8">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="w-full max-w-[400px]">
          <div className="mb-10 flex items-center gap-2.5 lg:hidden">
            <Logo />
            <span className="text-[17px] font-semibold tracking-[-0.02em]">
              INEGO<span className="text-faint">.dev</span>
            </span>
          </div>

          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={mode} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.22 }}>
              <h1 className="text-[34px] font-semibold leading-[1.1] tracking-[-0.03em] sm:text-[40px]">{copy.title}</h1>
              <p className="mt-2 text-[16px] text-mute">{copy.subtitle}</p>
            </motion.div>
          </AnimatePresence>

          <div className="mt-8 flex rounded-full bg-black/[0.05] p-1" role="tablist">
            {(['login', 'register'] as const).map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                onClick={() => switchMode(m)}
                className={`relative isolate flex-1 rounded-full py-2 text-[14px] font-medium transition-colors duration-300 ${mode === m ? 'text-ink' : 'text-mute hover:text-ink'}`}
              >
                {mode === m && (
                  <motion.span
                    layoutId="auth-mode"
                    className="absolute inset-0 -z-10 rounded-full bg-white shadow-[0_1px_2px_rgba(0,0,0,0.06),0_2px_8px_rgba(0,0,0,0.06)]"
                    transition={{ type: 'spring', stiffness: 420, damping: 36 }}
                  />
                )}
                {m === 'login' ? 'Вход' : 'Регистрация'}
              </button>
            ))}
          </div>

          <form onSubmit={onSubmit} noValidate className="mt-6 flex flex-col gap-3">
            <AnimatePresence initial={false}>
              {mode === 'register' && (
                <motion.div
                  key="name"
                  initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  transition={spring}
                  className="-m-2 overflow-hidden p-2"
                >
                  <Field label="Имя" value={name} onChange={setName} autoComplete="name" error={submitted ? errors.name : null} />
                </motion.div>
              )}
            </AnimatePresence>

            <Field label="Почта" type="email" value={email} onChange={setEmail} autoComplete="email" inputMode="email" error={submitted ? errors.email : null} />
            <PasswordField
              value={password}
              onChange={setPassword}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              error={submitted ? errors.password : null}
              showStrength={mode === 'register'}
            />

            <AnimatePresence>
              {error && (
                <motion.div
                  key={error.n}
                  initial={{ opacity: 0, x: 0 }}
                  animate={{ opacity: 1, x: [0, -8, 8, -5, 5, 0] }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.4 }}
                  role="alert"
                  className="rounded-[14px] bg-[#ff3b30]/[0.08] px-4 py-3 text-[14px] text-[#d70015]"
                >
                  {error.text}
                </motion.div>
              )}
            </AnimatePresence>

            <motion.button
              type="submit"
              whileTap={{ scale: 0.98 }}
              disabled={loading}
              className="group mt-3 flex h-[52px] items-center justify-center gap-2 rounded-full bg-ink text-[16px] font-medium text-white transition-colors hover:bg-black disabled:opacity-70"
            >
              {loading ? (
                <Loader2 className="size-5 animate-spin" />
              ) : (
                <>
                  {copy.submit}
                  <ArrowRight className="size-[18px] transition-transform duration-300 ease-apple group-hover:translate-x-0.5" />
                </>
              )}
            </motion.button>
          </form>

          <p className="mt-6 text-center text-[14px] text-mute">
            {mode === 'login' ? 'Ещё нет аккаунта? ' : 'Уже есть аккаунт? '}
            <button type="button" onClick={() => switchMode(mode === 'login' ? 'register' : 'login')} className="font-medium text-accent hover:underline">
              {mode === 'login' ? 'Зарегистрируйтесь' : 'Войдите'}
            </button>
          </p>
        </motion.div>
      </main>
    </div>
  )
}

/* ------------------------------------------------------------------ */

interface FieldProps {
  label: string
  value: string
  onChange: (v: string) => void
  type?: string
  autoComplete?: string
  inputMode?: React.HTMLAttributes<HTMLInputElement>['inputMode']
  error?: string | null
  trailing?: React.ReactNode
}

function Field({ label, value, onChange, type = 'text', autoComplete, inputMode, error, trailing }: FieldProps) {
  const id = useId()
  return (
    <div>
      <div
        className={`relative h-[56px] rounded-[14px] bg-white transition-shadow duration-200 ${
          error
            ? 'shadow-[0_0_0_1.5px_#ff3b30]'
            : 'shadow-[0_0_0_1px_rgba(0,0,0,0.1)] focus-within:shadow-[0_0_0_2px_var(--color-accent),0_0_0_6px_rgba(0,113,227,0.12)]'
        }`}
      >
        <input
          id={id}
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoComplete={autoComplete}
          inputMode={inputMode}
          placeholder=" "
          aria-invalid={!!error}
          className={`peer size-full rounded-[14px] bg-transparent px-4 pb-2 pt-6 text-[16px] outline-none ${trailing ? 'pr-12' : ''}`}
        />
        <label
          htmlFor={id}
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[16px] text-faint transition-all duration-200 ease-apple peer-focus:top-[17px] peer-focus:text-[12px] peer-[:not(:placeholder-shown)]:top-[17px] peer-[:not(:placeholder-shown)]:text-[12px]"
        >
          {label}
        </label>
        {trailing && <div className="absolute inset-y-0 right-2 flex items-center">{trailing}</div>}
      </div>
      <AnimatePresence initial={false}>
        {error && (
          <motion.p
            initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden px-1 pt-1.5 text-[13px] text-[#d70015]"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}

function PasswordField({
  value, onChange, autoComplete, error, showStrength,
}: { value: string; onChange: (v: string) => void; autoComplete: string; error?: string | null; showStrength: boolean }) {
  const [visible, setVisible] = useState(false)
  const s = strength(value)
  return (
    <div>
      <Field
        label="Пароль"
        type={visible ? 'text' : 'password'}
        value={value}
        onChange={onChange}
        autoComplete={autoComplete}
        error={error}
        trailing={
          <button
            type="button"
            onClick={() => setVisible((v) => !v)}
            aria-label={visible ? 'Скрыть пароль' : 'Показать пароль'}
            className="grid size-9 place-items-center rounded-full text-faint transition-colors hover:bg-black/[0.05] hover:text-ink"
          >
            {visible ? <EyeOff className="size-[18px]" /> : <Eye className="size-[18px]" />}
          </button>
        }
      />
      <AnimatePresence initial={false}>
        {showStrength && value && !error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="flex items-center gap-3 px-1 pt-2.5">
              <div className="flex flex-1 gap-1">
                {[0, 1, 2, 3].map((i) => (
                  <span key={i} className="h-1 flex-1 overflow-hidden rounded-full bg-black/[0.07]">
                    <span
                      className="block h-full rounded-full transition-all duration-500 ease-apple"
                      style={{ width: i < s.bars ? '100%' : '0%', background: s.color }}
                    />
                  </span>
                ))}
              </div>
              <span className="w-[72px] text-right text-[12px] font-medium" style={{ color: s.color }}>{s.label}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function strength(p: string) {
  const score = [p.length >= 8, p.length >= 12, /[a-zа-я]/.test(p) && /[A-ZА-Я]/.test(p), /\d/.test(p), /[^\p{L}\d]/u.test(p)].filter(Boolean).length
  if (p.length < 8) return { bars: 1, color: '#ff3b30', label: 'Короткий' }
  if (score <= 2) return { bars: 2, color: '#ff9500', label: 'Средний' }
  if (score === 3) return { bars: 3, color: '#34c759', label: 'Хороший' }
  return { bars: 4, color: '#248a3d', label: 'Надёжный' }
}

/* ------------------------------------------------------------------ */
/* Витрина: фасад, на котором по кругу меняется свет пресетов */

function Showcase() {
  const presets = presetsFor('exterior')
  const [i, setI] = useState(2)
  useEffect(() => {
    const t = setInterval(() => setI((x) => (x + 1) % presets.length), 3800)
    return () => clearInterval(t)
  }, [presets.length])

  const p = presets[i]
  const night = p.id === 'evening'
  const glass = night ? '#ffcf8a' : p.id === 'golden' ? 'rgba(255,214,160,0.55)' : 'rgba(255,255,255,0.35)'

  return (
    <aside className="sticky top-0 hidden h-dvh p-4 lg:block">
      <div className="relative size-full overflow-hidden rounded-[32px] shadow-card">
        {presets.map((x, k) => (
          <div key={x.id} className="absolute inset-0 transition-opacity duration-[1400ms] ease-in-out" style={{ background: x.sky, opacity: k === i ? 1 : 0 }} />
        ))}

        {p.sun && (
          <motion.span
            className="absolute rounded-full"
            animate={{
              left: p.sun.x,
              top: `calc(${p.sun.y} * 0.6)`,
              width: p.sun.size * 2.6,
              height: p.sun.size * 2.6,
              backgroundColor: p.sun.color,
              boxShadow: `0 0 ${p.sun.size * 3}px ${p.sun.size}px ${p.sun.color}`,
            }}
            transition={{ duration: 1.4, ease: [0.32, 0.72, 0, 1] }}
            style={{ translateX: '-50%', translateY: '-50%' }}
          />
        )}

        {/* силуэт виллы */}
        <svg viewBox="0 0 800 420" preserveAspectRatio="xMidYMax meet" className="absolute inset-x-0 bottom-0 h-[58%] w-full">
          <g style={{ transition: 'fill 1.4s' }}>
            <circle cx="90" cy="300" r="70" fill="rgba(25,30,25,0.42)" />
            <circle cx="150" cy="320" r="50" fill="rgba(25,30,25,0.36)" />
            <circle cx="720" cy="310" r="60" fill="rgba(25,30,25,0.4)" />
            <rect x="200" y="190" width="420" height="170" fill="rgba(22,22,32,0.38)" />
            <rect x="160" y="168" width="500" height="24" fill="rgba(22,22,32,0.55)" />
            <rect x="380" y="70" width="260" height="100" fill="rgba(22,22,32,0.42)" />
            <rect x="360" y="58" width="300" height="14" fill="rgba(22,22,32,0.55)" />
            <rect x="225" y="215" width="150" height="120" fill={glass} style={{ transition: 'fill 1.4s' }} />
            <rect x="395" y="215" width="200" height="120" fill={glass} style={{ transition: 'fill 1.4s' }} />
            <rect x="400" y="88" width="220" height="62" fill={glass} style={{ transition: 'fill 1.4s' }} />
            <rect x="0" y="358" width="800" height="62" fill="rgba(20,22,20,0.45)" />
            <rect x="240" y="358" width="340" height="8" fill="rgba(140,200,255,0.25)" />
          </g>
        </svg>

        <div className="absolute inset-0 bg-gradient-to-t from-black/55 via-black/5 to-transparent" />

        <div className="absolute left-10 top-9 flex items-center gap-2.5 text-white">
          <span className="rounded-[9px] ring-1 ring-white/30"><Logo /></span>
          <span className="text-[17px] font-semibold tracking-[-0.02em]">
            INEGO<span className="text-white/60">.dev</span>
          </span>
        </div>

        <div className="absolute inset-x-10 bottom-10 text-white">
          <h2 className="text-[44px] font-semibold leading-[1.05] tracking-[-0.035em] xl:text-[52px]">
            Свет, который
            <br />
            продаёт архитектуру.
          </h2>
          <div className="mt-6 flex items-center gap-4">
            <AnimatePresence mode="wait">
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.35 }}
                className="rounded-full bg-white/15 px-4 py-2 text-[13px] font-medium ring-1 ring-white/20 backdrop-blur-xl"
              >
                {p.title} <span className="text-white/60">· {p.meta}</span>
              </motion.div>
            </AnimatePresence>
            <div className="flex gap-1.5">
              {presets.map((x, k) => (
                <span key={x.id} className={`h-1.5 rounded-full transition-all duration-500 ease-apple ${k === i ? 'w-5 bg-white' : 'w-1.5 bg-white/40'}`} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}
