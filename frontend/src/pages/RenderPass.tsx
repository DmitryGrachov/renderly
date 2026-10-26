import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { Download, Eye, EyeOff, Layers, Loader2, RotateCcw, X } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { toStudioImage, type StudioImage } from '../lib/types'
import { buildMaterials, DIVIDER_ID, groupColor, maskPng, type MaterialGroup, type MaterialMap } from '../lib/materials'

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const

export function RenderPass() {
  const [image, setImage] = useState<StudioImage | null>(null)
  const [map, setMap] = useState<MaterialMap | null>(null)
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
      setMap(await buildMaterials(file, ctrl.signal))
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
    <div className="mx-auto max-w-[1880px] px-5 pb-24 sm:px-8">
      <section className="pb-10 pt-12 text-center sm:pb-12 sm:pt-16">
        <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="text-[15px] font-semibold text-gradient sm:text-[17px]">
          Render Pass
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.06 }}
          className="mt-3 text-[44px] font-semibold leading-[1.04] tracking-[-0.035em] sm:text-[64px]"
        >
          Кадр по схеме.
          <br />
          <span className="text-faint">Область — маска, линии — разделитель.</span>
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
  map: MaterialMap | null
  error: string | null
  onReset: () => void
  onRetry: () => void
}

function Viewer({ image, map, error, onReset, onRetry }: ViewerProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState(0)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [showAll, setShowAll] = useState(true)
  const [showSchema, setShowSchema] = useState(false)
  const [tip, setTip] = useState<{ x: number; y: number } | null>(null)

  const byId = useMemo(() => new Map((map?.groups ?? []).map((g) => [g.id, g])), [map])
  const colors = useMemo(() => {
    const n = Math.max(0, ...(map?.groups ?? []).map((g) => g.id)) + 1
    const c = new Uint8Array(n * 3)
    for (const g of map?.groups ?? []) c.set(groupColor(g.id), g.id * 3)
    return c
  }, [map])

  useEffect(() => setSelected(new Set()), [map])

  // оверлей: все маски полупрозрачно, наведённая и выбранные — ярче, без обводки; линии схемы (разделитель)
  // не рисуются никогда — на кадре они только мешают, маска разделителя берётся из списка
  useEffect(() => {
    const cv = canvas.current
    if (!cv || !map) return
    const { width: w, height: h, ids } = map
    cv.width = w
    cv.height = h
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(w, h)
    const px = img.data
    const base = showAll ? 105 : 0
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]
      if (!id || id === DIVIDER_ID) continue
      const on = id === hover || selected.has(id)
      const a = on ? 185 : base
      if (!a) continue
      const o = i * 4
      px[o] = colors[id * 3]
      px[o + 1] = colors[id * 3 + 1]
      px[o + 2] = colors[id * 3 + 2]
      px[o + 3] = a
    }
    ctx.putImageData(img, 0, 0)
  }, [map, colors, hover, selected, showAll])

  const idAt = (e: React.MouseEvent) => {
    if (!map) return 0
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const x = Math.floor(((e.clientX - r.left) / r.width) * map.width)
    const y = Math.floor(((e.clientY - r.top) / r.height) * map.height)
    if (x < 0 || y < 0 || x >= map.width || y >= map.height) return 0
    const id = map.ids[y * map.width + x]
    return id === DIVIDER_ID ? 0 : id // на линии наведение и клик ничего не выбирают
  }

  const toggle = (id: number) =>
    setSelected((s) => {
      const n = new Set(s)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })

  const download = async () => {
    if (!map || !selected.size) return
    const blob = await maskPng(map, selected, image.width, image.height)
    const names = [...selected].sort((a, b) => a - b).join('-')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `inego_mask_${names}_${image.name.replace(/\.[^.]+$/, '')}.png`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  const hovered = byId.get(hover)

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="relative overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04]">
        <div
          className={`relative mx-auto w-fit max-w-full ${map ? 'cursor-crosshair' : ''}`}
          onMouseMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect()
            setHover(idAt(e))
            setTip({ x: e.clientX - r.left, y: e.clientY - r.top })
          }}
          onMouseLeave={() => { setHover(0); setTip(null) }}
          onClick={(e) => { const id = idAt(e); if (id) toggle(id) }}
        >
          <img src={showSchema && map ? map.schema : image.url} alt={image.name} className="block max-h-[calc(100dvh-7rem)] w-auto max-w-full select-none" draggable={false} />
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
                Рисуем схему и режем маски… около минуты
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
          <div className="text-[17px] font-semibold tracking-[-0.02em]">Маски{map ? ` · ${map.groups.length}` : ''}</div>
          <div className="flex gap-1">
            {map && (
              <button
                onClick={() => setShowSchema((v) => !v)}
                title={showSchema ? 'Показать кадр' : 'Показать схему, по которой режутся маски'}
                className={`grid size-8 place-items-center rounded-full transition-colors hover:bg-fg/[0.05] hover:text-ink ${showSchema ? 'text-accent' : 'text-mute'}`}
              >
                <Layers className="size-4" />
              </button>
            )}
            <button
              onClick={() => setShowAll((v) => !v)}
              title={showAll ? 'Скрыть заливку' : 'Показать все маски'}
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
          {selected.size ? `Выбрано: ${selected.size}` : 'Наведите на кадр — клик выбирает маску'}
        </p>

        {map && (
          <>
            <ul className="mt-4 max-h-[52vh] space-y-0.5 overflow-y-auto pr-1">
              {map.groups.map((g) => (
                <GroupRow
                  key={g.id}
                  group={g}
                  active={selected.has(g.id)}
                  hovered={hover === g.id}
                  onHover={(on) => setHover(on ? g.id : 0)}
                  onClick={() => toggle(g.id)}
                />
              ))}
            </ul>
            <button
              onClick={download}
              disabled={!selected.size}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-full bg-ink px-4 py-2 text-[13px] font-medium text-on-ink transition-opacity disabled:opacity-30"
            >
              <Download className="size-4" /> Скачать маску
            </button>
          </>
        )}
      </aside>
    </div>
  )
}

function GroupRow({ group, active, hovered, onHover, onClick }: {
  group: MaterialGroup
  active: boolean
  hovered: boolean
  onHover: (on: boolean) => void
  onClick: () => void
}) {
  const [r, g, b] = groupColor(group.id)
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
        {/* цвет заливки на кадре — контрастный; кружок — цвет самой области на схеме */}
        <span className="size-3 shrink-0 rounded-[4px] ring-1 ring-black/10" style={{ background: `rgb(${r},${g},${b})` }} />
        <span className="size-3 shrink-0 rounded-full ring-1 ring-black/10" style={{ background: group.color }} />
        <span className="truncate">{group.label}</span>
        <span className="ml-auto text-[11px] tabular-nums text-faint">
          {active ? <span className="font-semibold text-accent">выбрано</span> : `${(group.area * 100).toFixed(group.area < 0.01 ? 1 : 0)}%`}
        </span>
      </button>
    </li>
  )
}
