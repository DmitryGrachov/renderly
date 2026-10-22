import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { Eye, EyeOff, Loader2, RotateCcw, X } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { toStudioImage, type StudioImage } from '../lib/types'
import { buildZones, zoneColor, type Zone, type ZoneKind, type ZoneMap } from '../lib/zones'

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const

const SECTIONS: { kinds: ZoneKind[]; title: string }[] = [
  { kinds: ['material'], title: 'Материалы стен' },
  { kinds: ['part'], title: 'Здание' },
  { kinds: ['veg'], title: 'Растительность' },
  { kinds: ['scene'], title: 'Окружение' },
]

export function RenderPass() {
  const [image, setImage] = useState<StudioImage | null>(null)
  const [map, setMap] = useState<ZoneMap | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  const start = async (file: File) => {
    abort.current?.abort()
    const ctrl = new AbortController()
    abort.current = ctrl
    setError(null)
    setMap(null)
    setImage(await toStudioImage(file))
    try {
      setMap(await buildZones(file, ctrl.signal))
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    }
  }

  const reset = () => {
    abort.current?.abort()
    setImage(null)
    setMap(null)
    setError(null)
  }

  return (
    <div className="mx-auto max-w-[1280px] px-5 pb-24 sm:px-8">
      <section className="pb-10 pt-12 text-center sm:pb-12 sm:pt-16">
        <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="text-[15px] font-semibold text-gradient sm:text-[17px]">
          Render pass
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.06 }}
          className="mt-3 text-[44px] font-semibold leading-[1.04] tracking-[-0.035em] sm:text-[64px]"
        >
          Кадр по зонам.
          <br />
          <span className="text-faint">Каждая — готовая маска.</span>
        </motion.h1>
      </section>

      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.12 }}>
        {!image ? (
          <div className="relative h-[400px] overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04] sm:aspect-[16/9] sm:h-auto">
            <Dropzone onFile={start} listenPaste />
          </div>
        ) : (
          <Viewer image={image} map={map} error={error} onReset={reset} onRetry={() => start(image.file)} />
        )}
      </motion.div>
    </div>
  )
}

interface ViewerProps {
  image: StudioImage
  map: ZoneMap | null
  error: string | null
  onReset: () => void
  onRetry: () => void
}

function Viewer({ image, map, error, onReset, onRetry }: ViewerProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<number>(0)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [showAll, setShowAll] = useState(true)
  const [tip, setTip] = useState<{ x: number; y: number } | null>(null)

  const byId = useMemo(() => new Map((map?.zones ?? []).map((z) => [z.id, z])), [map])
  const isVeg = useMemo(() => {
    const v = new Uint8Array(256)
    for (const z of map?.zones ?? []) if (z.kind === 'veg') v[z.id] = 1
    return v
  }, [map])
  const colors = useMemo(() => {
    const c = new Uint8Array(256 * 3)
    for (const z of map?.zones ?? []) c.set(zoneColor(z), z.id * 3)
    return c
  }, [map])

  // перерисовка оверлея: все зоны полупрозрачно, наведённая и выбранные — ярче, их край — белым;
  // растительность — по мягкой альфе поверх (ветки, а не пятно вместе с небом между ними)
  useEffect(() => {
    const cv = canvas.current
    if (!cv || !map) return
    const { width: w, height: h, ids, alpha, veg } = map
    cv.width = w
    cv.height = h
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(w, h)
    const px = img.data
    const base = showAll ? 105 : 0
    for (let i = 0; i < ids.length; i++) {
      const v = veg[i]
      if (v && alpha[i] > 3) {
        const k = alpha[i] / 255
        const a = (v === hover || selected.has(v) ? 215 : base) * k
        const id0 = isVeg[ids[i]] ? 0 : ids[i]
        const under = id0 && (showAll || id0 === hover || selected.has(id0))
        const o = i * 4
        const ua = under ? (id0 === hover || selected.has(id0) ? 185 : base) : 0
        const t = a / Math.max(1, Math.max(a, ua))
        for (let c = 0; c < 3; c++) {
          px[o + c] = under ? colors[id0 * 3 + c] * (1 - k) + colors[v * 3 + c] * k : colors[v * 3 + c]
        }
        px[o + 3] = Math.max(a, ua * (1 - k * t))
        continue
      }
      const id = ids[i]
      if (!id || isVeg[id]) continue
      const on = id === hover || selected.has(id)
      const a = on ? 185 : base
      if (!a) continue
      const o = i * 4
      px[o] = colors[id * 3]
      px[o + 1] = colors[id * 3 + 1]
      px[o + 2] = colors[id * 3 + 2]
      px[o + 3] = a
      if (on) {
        const x = i % w
        const edge =
          (x > 0 && ids[i - 1] !== id) || (x < w - 1 && ids[i + 1] !== id) ||
          (i >= w && ids[i - w] !== id) || (i < ids.length - w && ids[i + w] !== id)
        if (edge) {
          px[o] = px[o + 1] = px[o + 2] = 255
          px[o + 3] = 255
        }
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [map, colors, isVeg, hover, selected, showAll])

  const idAt = (e: React.MouseEvent) => {
    if (!map) return 0
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const x = Math.floor(((e.clientX - r.left) / r.width) * map.width)
    const y = Math.floor(((e.clientY - r.top) / r.height) * map.height)
    if (x < 0 || y < 0 || x >= map.width || y >= map.height) return 0
    const p = y * map.width + x
    // под курсором ветка (альфа > 0.5) — дерево; небо между ветками — небо
    if (map.veg[p] && map.alpha[p] > 127) return map.veg[p]
    return isVeg[map.ids[p]] ? 0 : map.ids[p]
  }

  const toggle = (ids: number[]) =>
    setSelected((s) => {
      const n = new Set(s)
      const all = ids.every((i) => n.has(i))
      ids.forEach((i) => (all ? n.delete(i) : n.add(i)))
      return n
    })

  const hovered = byId.get(hover)

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="relative overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04]">
        <div
          className={`relative ${map ? 'cursor-crosshair' : ''}`}
          onMouseMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect()
            setHover(idAt(e))
            setTip({ x: e.clientX - r.left, y: e.clientY - r.top })
          }}
          onMouseLeave={() => { setHover(0); setTip(null) }}
          onClick={(e) => { const id = idAt(e); if (id) toggle([id]) }}
        >
          <img src={image.url} alt={image.name} className="block w-full select-none" draggable={false} />
          {map && <canvas ref={canvas} className="pointer-events-none absolute inset-0 size-full" />}
          {hovered && tip && (
            <div
              className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[calc(100%+12px)] whitespace-nowrap rounded-full bg-ink/85 px-3 py-1 text-[12px] font-medium text-on-ink shadow-soft backdrop-blur"
              style={{ left: tip.x, top: tip.y }}
            >
              {hovered.label}
            </div>
          )}
          {!map && !error && (
            <div className="absolute inset-0 grid place-items-center bg-black/25 backdrop-blur-[2px]">
              <div className="flex items-center gap-2.5 rounded-full bg-surface/90 px-4 py-2 text-[14px] font-medium shadow-card">
                <Loader2 className="size-4 animate-spin text-accent" />
                Строим карту зон… обычно 20–60 с
              </div>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 grid place-items-center bg-black/35">
              <div className="max-w-[420px] rounded-2xl bg-surface px-5 py-4 text-center shadow-card">
                <p className="text-[14px] text-ink">{error}</p>
                <button onClick={onRetry} className="mt-3 rounded-full bg-ink px-4 py-1.5 text-[13px] font-medium text-on-ink">
                  Повторить
                </button>
              </div>
            </div>
          )}
        </div>
        <button
          onClick={onReset}
          aria-label="Другой кадр"
          className="absolute right-3 top-3 grid size-9 place-items-center rounded-full bg-surface/85 text-ink shadow-soft backdrop-blur transition-colors hover:bg-surface"
        >
          <X className="size-4" />
        </button>
      </div>

      <aside className="h-fit rounded-[28px] bg-surface p-5 shadow-card ring-1 ring-fg/[0.04] lg:sticky lg:top-20">
        <div className="flex items-center justify-between">
          <div className="text-[17px] font-semibold tracking-[-0.02em]">Зоны{map ? ` · ${map.zones.length}` : ''}</div>
          <div className="flex gap-1">
            <button
              onClick={() => setShowAll((v) => !v)}
              title={showAll ? 'Скрыть заливку' : 'Показать все зоны'}
              className="grid size-8 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink"
            >
              {showAll ? <Eye className="size-4" /> : <EyeOff className="size-4" />}
            </button>
            <button
              onClick={() => setSelected(new Set())}
              disabled={!selected.size}
              title="Сбросить выбор"
              className="grid size-8 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink disabled:opacity-30"
            >
              <RotateCcw className="size-4" />
            </button>
          </div>
        </div>
        <p className="mt-1 text-[13px] text-mute">
          {selected.size ? `Выбрано: ${selected.size}` : 'Наведите на кадр — клик выбирает зону'}
        </p>

        {map && (
          <div className="mt-4 max-h-[60vh] space-y-5 overflow-y-auto pr-1">
            {SECTIONS.map(({ kinds, title }) => {
              const list = map.zones.filter((z) => kinds.includes(z.kind))
              if (!list.length) return null
              const groups = kinds.includes('finish') ? [...new Set(list.map((z) => z.group).filter(Boolean))] as string[] : []
              return (
                <div key={title}>
                  <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-faint">{title}</div>
                  {groups.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-1.5">
                      {groups.map((g) => {
                        const ids = list.filter((z) => z.group === g).map((z) => z.id)
                        const [r, gg, b] = zoneColor(list.find((z) => z.group === g)!)
                        return (
                          <button
                            key={g}
                            onClick={() => toggle(ids)}
                            onMouseEnter={() => setHover(ids[0])}
                            onMouseLeave={() => setHover(0)}
                            className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[12px] font-medium ring-1 ring-fg/10 transition-colors hover:bg-fg/[0.04]"
                          >
                            <span className="size-2.5 rounded-full" style={{ background: `rgb(${r},${gg},${b})` }} />
                            Вся отделка {g}
                          </button>
                        )
                      })}
                    </div>
                  )}
                  <ul className="space-y-0.5">
                    {list.map((z) => (
                      <ZoneRow
                        key={z.id}
                        zone={z}
                        active={selected.has(z.id)}
                        hovered={hover === z.id}
                        onHover={(on) => setHover(on ? z.id : 0)}
                        onClick={() => toggle([z.id])}
                      />
                    ))}
                  </ul>
                </div>
              )
            })}
          </div>
        )}
      </aside>
    </div>
  )
}

function ZoneRow({ zone, active, hovered, onHover, onClick }: {
  zone: Zone
  active: boolean
  hovered: boolean
  onHover: (on: boolean) => void
  onClick: () => void
}) {
  const [r, g, b] = zoneColor(zone)
  return (
    <li>
      <button
        onClick={onClick}
        onMouseEnter={() => onHover(true)}
        onMouseLeave={() => onHover(false)}
        className={`flex w-full items-center gap-2.5 rounded-xl px-2.5 py-1.5 text-left text-[13px] transition-colors ${
          active ? 'bg-accent/10 text-ink' : hovered ? 'bg-fg/[0.05]' : 'hover:bg-fg/[0.04]'
        }`}
      >
        <span className="size-3 shrink-0 rounded-[4px] ring-1 ring-black/10" style={{ background: `rgb(${r},${g},${b})` }} />
        <span className="truncate">{zone.label}</span>
        {active && <span className="ml-auto text-[11px] font-semibold text-accent">выбрано</span>}
      </button>
    </li>
  )
}
