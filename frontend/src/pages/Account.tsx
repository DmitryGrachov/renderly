import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, animate, motion, useMotionValue, useTransform } from 'motion/react'
import {
  ArrowDownRight, ArrowUpRight, Check, Coins, Gauge, Gift, ImageIcon, Paintbrush, Plus, Sparkles, Timer, Zap,
} from 'lucide-react'
import { useAuth } from '../lib/auth'
import { MODELS, PACKS, dayTotal, fmt, modelById, useAccount, type Activity, type ModelId, type Pack } from '../lib/account'
import { UsageChart } from '../components/UsageChart'

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const

const RANGES = [
  { id: '7', label: '7 дней', days: 7 },
  { id: '30', label: '30 дней', days: 30 },
  { id: '90', label: '90 дней', days: 90 },
] as const
type RangeId = (typeof RANGES)[number]['id']

interface Props {
  onToast: (msg: string) => void
}

export function Account({ onToast }: Props) {
  const { user } = useAuth()
  const { balance, plan, model, setModel, usage, history, topUp } = useAccount()
  const [range, setRange] = useState<RangeId>('30')

  const days = RANGES.find((r) => r.id === range)!.days
  const slice = usage.slice(-days)
  const prev = usage.slice(-days * 2, -days)

  const stats = useMemo(() => {
    const sum = (arr: typeof usage) => arr.reduce((s, d) => s + dayTotal(d), 0)
    const renders = (arr: typeof usage) =>
      arr.reduce((s, d) => s + MODELS.reduce((a, m) => a + (d.byModel[m.id] ?? 0) / m.cost, 0), 0)
    const byModel = Object.fromEntries(MODELS.map((m) => [m.id, slice.reduce((s, d) => s + (d.byModel[m.id] ?? 0), 0)])) as Record<ModelId, number>
    const spent = sum(slice)
    const spentPrev = sum(prev)
    const r = renders(slice)
    const rPrev = renders(prev)
    return {
      spent,
      spentDelta: spentPrev ? (spent - spentPrev) / spentPrev : null,
      renders: Math.round(r),
      rendersDelta: rPrev ? (r - rPrev) / rPrev : null,
      perDay: Math.round(spent / slice.length),
      byModel,
    }
  }, [slice, prev])

  // расход в текущем «месяце» тарифа — последние 18 дней (до продления 12)
  const monthSpent = usage.slice(-18).reduce((s, d) => s + dayTotal(d), 0)
  const firstName = user?.name.split(/\s+/)[0] ?? ''
  const current = modelById(model)

  const buy = (p: Pack) => {
    topUp(p)
    onToast(`Демо-режим: начислено ${fmt(p.tokens + (p.bonus ?? 0))} токенов`)
  }

  return (
    <div className="mx-auto max-w-[1280px] px-5 pb-24 sm:px-8">
      {/* Hero */}
      <section className="pb-10 pt-12 sm:pb-12 sm:pt-16">
        <motion.p
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.05 }}
          className="text-[15px] font-semibold text-gradient sm:text-[17px]"
        >
          Личный кабинет
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.1 }}
          className="mt-3 text-[40px] font-semibold leading-[1.04] tracking-[-0.035em] sm:text-[56px]"
        >
          {firstName ? `${firstName}, ` : ''}добро пожаловать.
          <br />
          <span className="text-faint">Ваша студия под контролем.</span>
        </motion.h1>
      </section>

      {/* Баланс + метрики */}
      <motion.section
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.18 }}
        className="grid gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:gap-6"
      >
        <BalanceCard balance={balance} planName={plan.name} monthly={plan.monthly} monthSpent={monthSpent} renews={plan.renews} perRender={current.cost} modelName={current.name} />

        <div className="grid grid-cols-2 gap-5 lg:gap-6">
          <StatTile icon={Coins} label="Потрачено" value={fmt(stats.spent)} unit="токенов" delta={stats.spentDelta} invert />
          <StatTile icon={ImageIcon} label="Рендеров" value={fmt(stats.renders)} delta={stats.rendersDelta} />
          <StatTile icon={Gauge} label="В среднем за день" value={fmt(stats.perDay)} unit="токенов" />
          <StatTile icon={Timer} label="Хватит ещё на" value={fmt(Math.floor(balance / Math.max(1, stats.perDay)))} unit="дней" />
        </div>
      </motion.section>

      {/* График */}
      <motion.section
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.26 }}
        className="mt-5 rounded-[28px] bg-surface p-5 shadow-card ring-1 ring-fg/[0.04] sm:p-7 lg:mt-6"
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-[22px] font-semibold tracking-[-0.025em]">Расход токенов</h2>
            <p className="mt-1 text-[14px] text-mute">По дням и моделям · наведите на столбец для деталей</p>
          </div>
          <Segmented value={range} onChange={setRange} options={RANGES.map((r) => ({ id: r.id, label: r.label }))} layoutId="range-pill" />
        </div>

        {/* легенда с итогами */}
        <div className="mt-6 flex flex-wrap gap-x-8 gap-y-3">
          {MODELS.map((m) => (
            <div key={m.id} className="flex items-center gap-2.5">
              <span className="size-2.5 rounded-[4px]" style={{ background: `var(--color-m-${m.id.replace('.', '-')})` }} />
              <span className="text-[13px] text-mute">{m.name}</span>
              <span className="text-[13px] font-semibold tabular-nums">{fmt(stats.byModel[m.id])}</span>
            </div>
          ))}
        </div>

        <div className="mt-4 -mx-1">
          <UsageChart data={slice} rangeKey={range} />
        </div>
      </motion.section>

      {/* Модели */}
      <motion.section
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.32 }}
        className="mt-14"
      >
        <SectionHead title="Модель генерации" sub="Используется по умолчанию в Img to Img и Inpainting. Стоимость списывается за каждый рендер." />
        <div className="mt-6 grid gap-5 md:grid-cols-2 lg:gap-6" role="radiogroup" aria-label="Модель генерации">
          {MODELS.map((m) => (
            <ModelCard key={m.id} id={m.id} active={m.id === model} onSelect={() => { setModel(m.id); onToast(`Модель по умолчанию: ${m.name}`) }} />
          ))}
        </div>
      </motion.section>

      {/* Пакеты + история */}
      <motion.section
        id="topup"
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.38 }}
        className="mt-14 grid scroll-mt-24 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-6"
      >
        <div>
          <SectionHead title="Пополнить баланс" sub="Токены не сгорают и суммируются с тарифом." />
          <div className="mt-6 space-y-3">
            {PACKS.map((p) => <PackRow key={p.id} pack={p} onBuy={() => buy(p)} />)}
          </div>
          <p className="mt-4 text-[12.5px] text-faint">Оплата пока работает в демо-режиме — токены начисляются без списания средств.</p>
        </div>

        <div>
          <SectionHead title="История операций" sub="Последние списания и начисления." />
          <div className="mt-6 overflow-hidden rounded-[24px] bg-surface shadow-card ring-1 ring-fg/[0.04]">
            <div className="max-h-[392px] overflow-y-auto">
              <AnimatePresence initial={false}>
                {history.slice(0, 24).map((a) => <ActivityRow key={a.id} a={a} />)}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </motion.section>
    </div>
  )
}

/* ---------------- баланс ---------------- */

function CountUp({ value }: { value: number }) {
  const mv = useMotionValue(0)
  const text = useTransform(mv, (v) => fmt(Math.round(v)))
  useEffect(() => {
    const c = animate(mv, value, { duration: 1.1, ease: [0.32, 0.72, 0, 1] })
    return () => c.stop()
  }, [mv, value])
  return <motion.span>{text}</motion.span>
}

const dateFmt = new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'long' })

function BalanceCard(props: {
  balance: number; planName: string; monthly: number; monthSpent: number; renews: Date; perRender: number; modelName: string
}) {
  const { balance, planName, monthly, monthSpent, renews, perRender, modelName } = props
  const used = Math.min(1, monthSpent / monthly)
  return (
    <div className="relative isolate overflow-hidden rounded-[28px] bg-[#0a0a0c] p-6 text-white shadow-float ring-1 ring-white/[0.06] sm:p-8">
      {/* свечение */}
      <motion.div
        aria-hidden
        className="absolute -right-24 -top-32 -z-10 size-[420px] rounded-full opacity-70 blur-[80px]"
        style={{ background: 'conic-gradient(from 200deg, #ff8a3d, #ff4f7b, #6a5cff, #2f7bff, #ff8a3d)' }}
        animate={{ rotate: 360 }}
        transition={{ duration: 28, repeat: Infinity, ease: 'linear' }}
      />
      <div aria-hidden className="absolute inset-0 -z-10 bg-[radial-gradient(120%_80%_at_0%_100%,rgba(10,10,12,1)_30%,rgba(10,10,12,0.4)_70%,transparent)]" />
      <div
        aria-hidden
        className="absolute inset-0 -z-10 opacity-[0.07]"
        style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,.6) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.6) 1px,transparent 1px)', backgroundSize: '32px 32px', maskImage: 'radial-gradient(80% 70% at 70% 0%, black, transparent)' }}
      />

      <div className="flex items-center justify-between">
        <span className="text-[13px] font-medium text-white/60">Баланс</span>
        <span className="flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1 text-[12px] font-semibold ring-1 ring-white/15 backdrop-blur">
          <Sparkles className="size-3.5" /> {planName}
        </span>
      </div>

      <div className="mt-8 flex items-end gap-3">
        <span className="text-[56px] font-semibold leading-none tracking-[-0.045em] tabular-nums sm:text-[76px]">
          <CountUp value={balance} />
        </span>
        <span className="mb-2 text-[15px] font-medium text-white/55 sm:mb-3">токенов</span>
      </div>
      <p className="mt-3 text-[14px] text-white/60">
        ≈ {fmt(Math.floor(balance / perRender))} рендеров на {modelName}
      </p>

      <div className="mt-8">
        <div className="flex justify-between text-[12.5px] text-white/55">
          <span>Тариф в этом месяце</span>
          <span className="tabular-nums"><span className="text-white">{fmt(monthSpent)}</span> / {fmt(monthly)}</span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
          <motion.div
            className="h-full rounded-full"
            style={{ background: 'linear-gradient(90deg,#ff8a3d,#ff4f7b 45%,#8f5cff)' }}
            initial={{ width: 0 }}
            animate={{ width: `${used * 100}%` }}
            transition={{ duration: 1.2, ease: [0.32, 0.72, 0, 1], delay: 0.3 }}
          />
        </div>
        <div className="mt-2 text-[12.5px] text-white/45">Продление {dateFmt.format(renews)}</div>
      </div>

      <div className="mt-8 flex flex-wrap gap-2.5">
        <a
          href="#topup"
          onClick={(e) => { e.preventDefault(); document.getElementById('topup')?.scrollIntoView({ behavior: 'smooth' }) }}
          className="flex items-center gap-1.5 rounded-full bg-white px-5 py-2.5 text-[14px] font-semibold text-[#0a0a0c] transition-transform duration-200 hover:scale-[1.03] active:scale-[0.98]"
        >
          <Plus className="size-4" strokeWidth={2.5} /> Пополнить
        </a>
        <a
          href="#/img2img"
          className="flex items-center gap-1.5 rounded-full bg-white/10 px-5 py-2.5 text-[14px] font-semibold ring-1 ring-white/15 backdrop-blur transition-colors hover:bg-white/15"
        >
          <Zap className="size-4" /> К генерации
        </a>
      </div>
    </div>
  )
}

/* ---------------- метрики ---------------- */

function StatTile({ icon: Icon, label, value, unit, delta, invert }: {
  icon: typeof Coins; label: string; value: string; unit?: string; delta?: number | null; invert?: boolean
}) {
  const up = (delta ?? 0) >= 0
  // для расхода рост — нейтральная информация, не «хорошо»; цвет даём только росту рендеров
  const good = invert ? null : up
  return (
    <div className="flex flex-col justify-between rounded-[24px] bg-surface p-5 shadow-card ring-1 ring-fg/[0.04] sm:p-6">
      <div className="flex items-center gap-2 text-[13px] text-mute">
        <span className="grid size-7 place-items-center rounded-full bg-fg/[0.05]"><Icon className="size-[15px] text-ink-2" strokeWidth={1.9} /></span>
        {label}
      </div>
      <div className="mt-6">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={value}
            initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.18 }}
            className="text-[30px] font-semibold leading-none tracking-[-0.035em] tabular-nums sm:text-[36px]"
          >
            {value}
          </motion.div>
        </AnimatePresence>
        <div className="mt-2 flex items-center gap-2 text-[12.5px]">
          {unit && <span className="text-faint">{unit}</span>}
          {delta != null && (
            <span className={`flex items-center gap-0.5 font-medium tabular-nums ${good === null ? 'text-mute' : good ? 'text-emerald-600 dark:text-emerald-400' : 'text-danger'}`}>
              {up ? <ArrowUpRight className="size-3.5" /> : <ArrowDownRight className="size-3.5" />}
              {Math.abs(Math.round(delta * 100))}%
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

/* ---------------- модели ---------------- */

function Meter({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-[72px] text-[12.5px] text-mute">{label}</span>
      <div className="flex flex-1 gap-1" aria-label={`${label}: ${value} из 5`}>
        {[1, 2, 3, 4, 5].map((i) => (
          <span key={i} className={`h-1.5 flex-1 rounded-full ${i <= value ? 'bg-ink' : 'bg-fg/[0.08]'}`} />
        ))}
      </div>
    </div>
  )
}

function ModelCard({ id, active, onSelect }: { id: ModelId; active: boolean; onSelect: () => void }) {
  const m = modelById(id)
  return (
    <button
      role="radio"
      aria-checked={active}
      onClick={onSelect}
      className={`group relative flex flex-col rounded-[24px] bg-surface p-6 text-left shadow-card transition-all duration-300 ease-apple hover:-translate-y-0.5 ${
        active ? 'ring-2 ring-ink' : 'ring-1 ring-fg/[0.06] hover:ring-fg/[0.14]'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <span
          className="grid size-11 place-items-center rounded-[14px] text-white shadow-soft"
          style={{ background: `linear-gradient(140deg, var(--color-m-${id.replace('.', '-')}), color-mix(in oklab, var(--color-m-${id.replace('.', '-')}) 55%, #000))` }}
        >
          <Sparkles className="size-5" strokeWidth={1.9} />
        </span>
        <span className={`grid size-6 place-items-center rounded-full transition-colors ${active ? 'bg-ink text-on-ink' : 'ring-1 ring-fg/[0.15]'}`}>
          <AnimatePresence>
            {active && (
              <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: 'spring', stiffness: 500, damping: 28 }}>
                <Check className="size-3.5" strokeWidth={3} />
              </motion.span>
            )}
          </AnimatePresence>
        </span>
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <span className="text-[19px] font-semibold tracking-[-0.02em]">{m.name}</span>
        {m.badge && <span className="rounded-full bg-fg/[0.05] px-2 py-0.5 text-[11px] font-semibold text-ink-2">{m.badge}</span>}
      </div>
      <div className="mt-0.5 text-[12.5px] text-faint">{m.tag}</div>
      <p className="mt-3 text-[14px] leading-relaxed text-mute">{m.description}</p>

      <div className="mt-5 space-y-2">
        <Meter label="Скорость" value={m.speed} />
        <Meter label="Качество" value={m.quality} />
      </div>

      <div className="mt-6 flex items-end justify-between border-t border-fg/[0.06] pt-4">
        <div>
          <span className="text-[26px] font-semibold tracking-[-0.03em] tabular-nums">{m.cost}</span>
          <span className="ml-1 text-[13px] text-mute">токенов / рендер</span>
        </div>
        <span className="text-[12.5px] text-faint">{m.time}</span>
      </div>
    </button>
  )
}

/* ---------------- пакеты ---------------- */

function PackRow({ pack, onBuy }: { pack: Pack; onBuy: () => void }) {
  const total = pack.tokens + (pack.bonus ?? 0)
  const perToken = (pack.price / total).toFixed(2).replace('.', ',')
  return (
    <div
      className={`relative flex items-center gap-4 overflow-hidden rounded-[22px] bg-surface p-4 pl-5 shadow-card sm:p-5 sm:pl-6 ${
        pack.highlight ? 'border-2 border-transparent' : 'ring-1 ring-fg/[0.04]'
      }`}
      style={pack.highlight ? { background: 'linear-gradient(var(--color-surface),var(--color-surface)) padding-box, linear-gradient(110deg,#ff8a3d,#ff4f7b 40%,#8f5cff) border-box' } : undefined}
    >
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[22px] font-semibold tracking-[-0.03em] tabular-nums">{fmt(pack.tokens)}</span>
          <span className="text-[13px] text-mute">токенов</span>
          {pack.bonus && (
            <span className="flex items-center gap-1 rounded-full bg-fg/[0.05] px-2 py-0.5 text-[11.5px] font-semibold text-ink-2">
              <Gift className="size-3" /> +{fmt(pack.bonus)}
            </span>
          )}
          {pack.highlight && <span className="text-[11.5px] font-semibold text-gradient">Выгодно</span>}
        </div>
        <div className="mt-0.5 text-[12.5px] text-faint">{perToken} ₽ за токен</div>
      </div>
      <button
        onClick={onBuy}
        className="shrink-0 rounded-full bg-ink px-5 py-2.5 text-[14px] font-semibold text-on-ink transition-all duration-200 hover:bg-ink-hover hover:scale-[1.03] active:scale-[0.98]"
      >
        {fmt(pack.price)} ₽
      </button>
    </div>
  )
}

/* ---------------- история ---------------- */

const timeFmt = new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })

function ActivityRow({ a }: { a: Activity }) {
  const Icon = a.kind === 'render' ? ImageIcon : a.kind === 'inpaint' ? Paintbrush : a.kind === 'topup' ? Plus : Gift
  const credit = a.amount > 0
  return (
    <motion.div
      layout
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      transition={{ type: 'spring', stiffness: 400, damping: 36 }}
      className="border-b border-fg/[0.05] last:border-0"
    >
      <div className="flex items-center gap-3.5 px-5 py-3.5">
        <span className={`grid size-9 shrink-0 place-items-center rounded-full ${credit ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' : 'bg-fg/[0.05] text-ink-2'}`}>
          <Icon className="size-4" strokeWidth={1.9} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[14px] font-medium">{a.title}</div>
          <div className="flex items-center gap-1.5 text-[12px] text-faint">
            {a.model && (
              <>
                <span className="size-1.5 rounded-full" style={{ background: `var(--color-m-${a.model.replace('.', '-')})` }} />
                {modelById(a.model).name} ·
              </>
            )}
            <span>{timeFmt.format(a.at)}</span>
          </div>
        </div>
        <span className={`text-[14px] font-semibold tabular-nums ${credit ? 'text-emerald-600 dark:text-emerald-400' : 'text-ink'}`}>
          {credit ? '+' : '−'}{fmt(Math.abs(a.amount))}
        </span>
      </div>
    </motion.div>
  )
}

/* ---------------- общее ---------------- */

function SectionHead({ title, sub }: { title: string; sub: string }) {
  return (
    <div>
      <h2 className="text-[28px] font-semibold tracking-[-0.03em] sm:text-[32px]">{title}</h2>
      <p className="mt-1.5 max-w-[560px] text-[15px] text-mute">{sub}</p>
    </div>
  )
}

function Segmented<T extends string>({ value, onChange, options, layoutId }: {
  value: T; onChange: (v: T) => void; options: { id: T; label: string }[]; layoutId: string
}) {
  return (
    <div className="relative flex rounded-full bg-fg/[0.05] p-1" role="tablist">
      {options.map((o) => {
        const active = o.id === value
        return (
          <button
            key={o.id}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(o.id)}
            className={`relative z-10 rounded-full px-3.5 py-1.5 text-[13px] font-medium transition-colors duration-300 ${active ? 'text-ink' : 'text-mute hover:text-ink'}`}
          >
            {active && (
              <motion.span
                layoutId={layoutId}
                className="absolute inset-0 -z-10 rounded-full bg-pill shadow-[0_1px_2px_rgba(0,0,0,0.06),0_2px_8px_rgba(0,0,0,0.06)]"
                transition={{ type: 'spring', stiffness: 420, damping: 36 }}
              />
            )}
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
