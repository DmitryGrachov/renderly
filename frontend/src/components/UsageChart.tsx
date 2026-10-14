import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { MODELS, fmt, type ModelId, type UsageDay } from '../lib/account'

const H = 260
const PAD = { top: 12, right: 8, bottom: 28, left: 44 }
const ORDER: ModelId[] = MODELS.map((m) => m.id) // снизу вверх, порядок фиксирован

const dayFmt = new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'short' })
const longFmt = new Intl.DateTimeFormat('ru-RU', { weekday: 'short', day: 'numeric', month: 'long' })

/** «Красивый» верх шкалы: 1/2/2.5/5 × 10^n. */
function niceMax(v: number) {
  if (v <= 0) return 10
  const p = 10 ** Math.floor(Math.log10(v))
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p
  return 10 * p
}

/** Прямоугольник со скруглённым верхом — «шапка» столбика. */
function topRounded(x: number, y: number, w: number, h: number, r: number) {
  r = Math.min(r, w / 2, h)
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`
}

export function UsageChart({ data, rangeKey }: { data: UsageDay[]; rangeKey: string }) {
  const wrap = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(800)
  const [hover, setHover] = useState<number | null>(null)

  useLayoutEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const totals = data.map((d) => ORDER.reduce((s, m) => s + d.byModel[m], 0))
  const max = niceMax(Math.max(...totals))
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max)

  const iw = Math.max(0, w - PAD.left - PAD.right)
  const ih = H - PAD.top - PAD.bottom
  const step = iw / data.length
  const bw = Math.max(2, Math.min(28, step * (data.length > 40 ? 0.62 : 0.56)))
  const y = (v: number) => PAD.top + ih - (v / max) * ih

  // подписи по оси X — не больше ~7, чтобы не слипались
  const labelEvery = Math.ceil(data.length / Math.max(2, Math.floor(iw / 90)))
  const labels = useMemo(
    () => data.map((_, i) => ((data.length - 1 - i) % labelEvery === 0 ? i : -1)).filter((i) => i >= 0),
    [data, labelEvery],
  )

  const hv = hover !== null ? data[hover] : null
  const tipLeft = hover !== null ? PAD.left + step * hover + step / 2 : 0
  const tipFlip = tipLeft > w - 200

  return (
    <div ref={wrap} className="relative w-full select-none" onPointerLeave={() => setHover(null)}>
      <svg width={w} height={H} className="block overflow-visible" role="img" aria-label="Расход токенов по дням, по моделям">
        {/* сетка */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={w - PAD.right} y1={y(t)} y2={y(t)} className={t === 0 ? 'stroke-fg/[0.14]' : 'stroke-fg/[0.06]'} strokeDasharray={t === 0 ? undefined : '2 4'} />
            <text x={PAD.left - 10} y={y(t)} dy="0.32em" textAnchor="end" className="fill-faint text-[11px] tabular-nums">
              {t >= 1000 ? `${(t / 1000).toLocaleString('ru-RU', { maximumFractionDigits: 1 })}k` : t}
            </text>
          </g>
        ))}

        {/* подсветка колонки */}
        {hover !== null && (
          <rect x={PAD.left + step * hover} y={PAD.top} width={step} height={ih} rx={6} className="fill-fg/[0.04]" />
        )}

        {/* столбики */}
        <AnimatePresence mode="wait" initial={false}>
          <motion.g
            key={rangeKey}
            initial={{ scaleY: 0, opacity: 0 }}
            animate={{ scaleY: 1, opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.12 } }}
            transition={{ type: 'spring', stiffness: 160, damping: 24 }}
            style={{ originY: 1 }}
          >
            {data.map((d, i) => {
              const x = PAD.left + step * i + (step - bw) / 2
              let acc = 0
              const segs = ORDER.filter((m) => d.byModel[m] > 0)
              const dim = hover !== null && hover !== i
              return (
                <g key={i} opacity={dim ? 0.38 : 1} style={{ transition: 'opacity .2s' }}>
                  {segs.map((m, k) => {
                    const v = d.byModel[m]
                    const y0 = y(acc)
                    acc += v
                    const y1 = y(acc)
                    // 2px зазор цвета поверхности между сегментами
                    const gap = k > 0 ? 2 : 0
                    const h = Math.max(0, y0 - y1 - gap)
                    const top = k === segs.length - 1
                    const fill = `var(--color-m-${m})`
                    return top ? (
                      <path key={m} d={topRounded(x, y1, bw, h, 4)} fill={fill} />
                    ) : (
                      <rect key={m} x={x} y={y1} width={bw} height={h} fill={fill} />
                    )
                  })}
                </g>
              )
            })}
          </motion.g>
        </AnimatePresence>

        {/* подписи X */}
        {labels.map((i) => (
          <text key={i} x={PAD.left + step * i + step / 2} y={H - 8} textAnchor="middle" className="fill-faint text-[11px]">
            {dayFmt.format(data[i].date).replace('.', '')}
          </text>
        ))}

        {/* зоны наведения — шире самих столбиков */}
        {data.map((_, i) => (
          <rect
            key={i}
            x={PAD.left + step * i}
            y={0}
            width={step}
            height={H}
            fill="transparent"
            onPointerEnter={() => setHover(i)}
            onPointerDown={() => setHover(i)}
          />
        ))}
      </svg>

      <AnimatePresence>
        {hv && (
          <motion.div
            key="tip"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0, left: tipLeft }}
            exit={{ opacity: 0, transition: { duration: 0.1 } }}
            transition={{ type: 'spring', stiffness: 600, damping: 40 }}
            className="pointer-events-none absolute top-2 z-10"
          >
            <div
              className="w-[188px] rounded-2xl bg-surface/[0.94] p-3 shadow-float ring-1 ring-fg/[0.06] backdrop-blur-xl transition-transform duration-200"
              style={{ transform: tipFlip ? 'translateX(calc(-100% - 14px))' : 'translateX(14px)' }}
            >
            <div className="text-[12px] font-medium text-faint first-letter:uppercase">{longFmt.format(hv.date)}</div>
            <div className="mt-0.5 text-[20px] font-semibold tabular-nums tracking-[-0.02em]">
              {fmt(ORDER.reduce((s, m) => s + hv.byModel[m], 0))}
              <span className="ml-1 text-[12px] font-medium text-mute">токенов</span>
            </div>
            <div className="mt-2 space-y-1">
              {[...ORDER].reverse().map((m) => (
                <div key={m} className="flex items-center gap-2 text-[12.5px]">
                  <span className="size-2 rounded-[3px]" style={{ background: `var(--color-m-${m})` }} />
                  <span className="flex-1 text-ink-2">{MODELS.find((x) => x.id === m)!.name}</span>
                  <span className="tabular-nums text-ink">{fmt(hv.byModel[m])}</span>
                </div>
              ))}
            </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* табличное представление для скринридеров */}
      <table className="sr-only">
        <caption>Расход токенов по дням</caption>
        <thead>
          <tr><th>Дата</th>{ORDER.map((m) => <th key={m}>{m}</th>)}</tr>
        </thead>
        <tbody>
          {data.map((d, i) => (
            <tr key={i}><td>{dayFmt.format(d.date)}</td>{ORDER.map((m) => <td key={m}>{d.byModel[m]}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
