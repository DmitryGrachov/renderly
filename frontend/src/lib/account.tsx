import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { useAuth } from './auth'

/*
 * Личный кабинет: модели, токены, расход.
 * Настоящей биллинговой системы пока нет — баланс и история живут в localStorage,
 * а расход за прошлые дни детерминированно генерируется из id пользователя.
 */

export type ModelId = 'klein-4b' | 'klein-9b' | 'qwen-edit-2511' | 'qwen-image-2.1'

export interface Model {
  id: ModelId
  name: string
  tag: string
  description: string
  cost: number // токенов за рендер
  speed: number // 1..5
  quality: number // 1..5
  time: string
  badge?: string
  /** Модель есть только в Inpainting (в Img to Img рендер идёт на запасной) */
  inpaintOnly?: boolean
}

// id совпадают с ключами моделей на бэкенде (поле model в /api/img2img)
export const MODELS: Model[] = [
  {
    id: 'klein-4b',
    name: 'FLUX.2 klein 4B',
    tag: '4B · distilled',
    description: 'Мгновенные итерации. Идеальна для поиска света и быстрых правок.',
    cost: 2,
    speed: 5,
    quality: 3,
    time: '≈ 15 с',
  },
  {
    id: 'klein-9b',
    name: 'FLUX.2 klein 9B',
    tag: '9B · distilled',
    description: 'Вдвое больше параметров: чище материалы, точнее мелкая геометрия и свет.',
    cost: 4,
    speed: 4,
    quality: 4,
    time: '≈ 20 с',
    badge: 'Детальнее',
  },
  {
    id: 'qwen-edit-2511',
    name: 'Qwen-Image-Edit 2511',
    tag: '20B · Lightning',
    description: 'Редактор на 20B: точнее понимает запрос и вписывает объекты. Только Inpainting.',
    cost: 6,
    speed: 2,
    quality: 5,
    time: '≈ 40 с',
    badge: 'Inpainting',
    inpaintOnly: true,
  },
  {
    id: 'qwen-image-2.1',
    name: 'Qwen-Image 2.1 Turbo',
    tag: '7B · 8 шагов',
    description: 'Новейшая Qwen (сентябрь 2026): нейрорендер Img to Img в 2K и правка по маске, компактнее и быстрее 2511. Лицензия некоммерческая.',
    cost: 5,
    speed: 3,
    quality: 5,
    time: '≈ 30 с',
    badge: 'Новая',
  },
]

/** В Img to Img модели «только Inpainting» заменяются этой */
export const IMG2IMG_FALLBACK: ModelId = 'klein-9b'

const ZERO: Record<ModelId, number> = { 'klein-4b': 0, 'klein-9b': 0, 'qwen-edit-2511': 0, 'qwen-image-2.1': 0 }

/** Сумма токенов за день по всем моделям. */
export const dayTotal = (d: UsageDay) => MODELS.reduce((s, m) => s + (d.byModel[m.id] ?? 0), 0)

export const modelById = (id: ModelId) => MODELS.find((m) => m.id === id) ?? MODELS[0]

export interface Plan {
  name: string
  monthly: number
  renews: Date
}

export interface UsageDay {
  date: Date
  byModel: Record<ModelId, number> // токенов
}

export interface Activity {
  id: string
  at: number
  kind: 'render' | 'inpaint' | 'topup' | 'bonus'
  title: string
  model?: ModelId
  amount: number // + начисление, − списание
}

export interface Pack {
  id: string
  tokens: number
  price: number
  bonus?: number
  highlight?: boolean
}

export const PACKS: Pack[] = [
  { id: 's', tokens: 500, price: 490 },
  { id: 'm', tokens: 2000, price: 1690, bonus: 200, highlight: true },
  { id: 'l', tokens: 6000, price: 4490, bonus: 1000 },
]

// ---------- детерминированный «шум» ----------
function hash(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const DAYS = 90
const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate())

function fakeUsage(seed: string): UsageDay[] {
  const r = rng(hash(seed))
  const today = startOfDay(new Date())
  const out: UsageDay[] = []
  for (let i = DAYS - 1; i >= 0; i--) {
    const date = new Date(today)
    date.setDate(today.getDate() - i)
    const weekend = date.getDay() === 0 || date.getDay() === 6
    // плавный рост активности + выходные провалы + редкие «дедлайны»
    const trend = 0.55 + 0.45 * ((DAYS - i) / DAYS)
    const burst = r() < 0.07 ? 2.2 : 1
    const base = (weekend ? 0.35 : 1) * trend * burst
    const renders = (k: number) => Math.max(0, Math.round(base * k * (0.4 + r() * 1.2)))
    out.push({
      date,
      byModel: {
        'klein-4b': renders(16) * 2,
        'klein-9b': renders(6) * 4,
        'qwen-edit-2511': 0,
        'qwen-image-2.1': 0,
      },
    })
  }
  return out
}

function fakeHistory(seed: string, usage: UsageDay[]): Activity[] {
  const r = rng(hash(seed + ':h'))
  const out: Activity[] = []
  const now = Date.now()
  const titles: Record<'render' | 'inpaint', string[]> = {
    render: ['Img to Img · Golden hour', 'Img to Img · Вечер', 'Img to Img · День', 'Img to Img · Свой референс'],
    inpaint: ['Inpainting · Замена материала', 'Inpainting · Удаление объекта', 'Inpainting · Озеленение'],
  }
  let t = now - 1000 * 60 * 14
  for (let i = 0; i < 14; i++) {
    const kind = r() < 0.7 ? 'render' : 'inpaint'
    const model: ModelId = r() < 0.6 ? 'klein-4b' : 'klein-9b'
    out.push({
      id: `h${i}`,
      at: t,
      kind,
      title: titles[kind][Math.floor(r() * titles[kind].length)],
      model,
      amount: -modelById(model).cost,
    })
    t -= 1000 * 60 * (20 + r() * 400)
    if (i === 6) {
      out.push({ id: 'topup', at: t, kind: 'topup', title: 'Пополнение · 2 000 + 200 бонус', amount: 2200 })
      t -= 1000 * 60 * 60 * 30
    }
  }
  out.push({ id: 'welcome', at: usage[0].date.getTime(), kind: 'bonus', title: 'Приветственный бонус', amount: 300 })
  return out
}

// ---------- DeepSeek (разбор сцены, размер и место объекта, инструкции) ----------
export type LlmId = 'deepseek-flash' | 'deepseek-v4-pro'

export const LLMS: { id: LlmId; label: string; hint: string }[] = [
  { id: 'deepseek-flash', label: 'Flash', hint: 'DeepSeek Flash — быстрее' },
  { id: 'deepseek-v4-pro', label: 'Pro', hint: 'DeepSeek V4 Pro — точнее размер, место и описание, но медленнее' },
]

// ---------- контекст ----------
interface AccountContextValue {
  balance: number
  plan: Plan
  model: ModelId
  setModel: (m: ModelId) => void
  /** Модель DeepSeek для разбора сцены и инструкций (переключатель в шапке) */
  llm: LlmId
  setLlm: (m: LlmId) => void
  usage: UsageDay[]
  history: Activity[]
  /** Демо-пополнение без оплаты. */
  topUp: (pack: Pack) => void
  /** Списать стоимость рендера текущей модели. */
  charge: (kind: 'render' | 'inpaint', title: string, used?: ModelId) => void
}

const AccountContext = createContext<AccountContextValue | null>(null)

const MODEL_KEY = 'inego.model'
const readModel = (): ModelId => {
  try {
    const v = localStorage.getItem(MODEL_KEY)
    if (MODELS.some((m) => m.id === v)) return v as ModelId
  } catch { /* storage недоступен */ }
  return 'klein-4b'
}

const LLM_KEY = 'inego.llm'
const readLlm = (): LlmId => {
  try {
    const v = localStorage.getItem(LLM_KEY)
    if (LLMS.some((m) => m.id === v)) return v as LlmId
  } catch { /* storage недоступен */ }
  return 'deepseek-flash'
}

interface Local {
  balance: number
  events: Activity[]
}

export function AccountProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  const seed = user?.id ?? 'guest'
  const key = `inego.wallet.${seed}`

  const [model, setModelState] = useState<ModelId>(readModel)
  const [llm, setLlmState] = useState<LlmId>(readLlm)
  const [local, setLocal] = useState<Local>({ balance: 0, events: [] })

  useEffect(() => {
    const r = rng(hash(seed + ':b'))
    let next: Local = { balance: 1200 + Math.round(r() * 900), events: [] }
    try {
      const raw = localStorage.getItem(key)
      if (raw) next = JSON.parse(raw)
    } catch { /* storage недоступен */ }
    setLocal(next)
  }, [key, seed])

  const persist = useCallback((fn: (l: Local) => Local) => {
    setLocal((prev) => {
      const next = fn(prev)
      try { localStorage.setItem(key, JSON.stringify(next)) } catch { /* storage недоступен */ }
      return next
    })
  }, [key])

  const setModel = useCallback((m: ModelId) => {
    setModelState(m)
    try { localStorage.setItem(MODEL_KEY, m) } catch { /* storage недоступен */ }
  }, [])

  const setLlm = useCallback((m: LlmId) => {
    setLlmState(m)
    try { localStorage.setItem(LLM_KEY, m) } catch { /* storage недоступен */ }
  }, [])

  const baseUsage = useMemo(() => fakeUsage(seed), [seed])
  const baseHistory = useMemo(() => fakeHistory(seed, baseUsage), [seed, baseUsage])

  // реальные списания этой сессии добавляются в сегодняшний столбик
  const usage = useMemo(() => {
    const today = startOfDay(new Date()).getTime()
    const extra = { ...ZERO }
    for (const e of local.events) {
      if (e.amount < 0 && e.model && e.model in extra && startOfDay(new Date(e.at)).getTime() === today) extra[e.model] -= e.amount
    }
    return baseUsage.map((d, i) =>
      i === baseUsage.length - 1
        ? { ...d, byModel: Object.fromEntries(MODELS.map((m) => [m.id, (d.byModel[m.id] ?? 0) + extra[m.id]])) as Record<ModelId, number> }
        : d,
    )
  }, [baseUsage, local.events])

  const history = useMemo(() => [...local.events, ...baseHistory].sort((a, b) => b.at - a.at), [local.events, baseHistory])

  const topUp = useCallback((pack: Pack) => {
    const amount = pack.tokens + (pack.bonus ?? 0)
    persist((l) => ({
      balance: l.balance + amount,
      events: [{ id: crypto.randomUUID(), at: Date.now(), kind: 'topup' as const, title: `Пополнение · ${pack.tokens.toLocaleString('ru-RU')}${pack.bonus ? ` + ${pack.bonus} бонус` : ''}`, amount }, ...l.events].slice(0, 50),
    }))
  }, [persist])

  const charge = useCallback((kind: 'render' | 'inpaint', title: string, used: ModelId = model) => {
    const cost = modelById(used).cost
    persist((l) => ({
      balance: Math.max(0, l.balance - cost),
      events: [{ id: crypto.randomUUID(), at: Date.now(), kind, title, model: used, amount: -cost }, ...l.events].slice(0, 50),
    }))
  }, [model, persist])

  const plan = useMemo<Plan>(() => {
    const renews = startOfDay(new Date())
    renews.setDate(renews.getDate() + 12)
    return { name: 'Studio', monthly: 3000, renews }
  }, [])

  return (
    <AccountContext.Provider value={{ balance: local.balance, plan, model, setModel, llm, setLlm, usage, history, topUp, charge }}>
      {children}
    </AccountContext.Provider>
  )
}

export function useAccount() {
  const ctx = useContext(AccountContext)
  if (!ctx) throw new Error('useAccount must be used inside AccountProvider')
  return ctx
}

export const fmt = (n: number) => n.toLocaleString('ru-RU')
