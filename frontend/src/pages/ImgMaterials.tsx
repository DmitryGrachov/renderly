import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Brush, Check, Download, Eraser, SunMedium, Pentagon, SquareDashed, Plus, RefreshCw, RotateCcw, Sparkles, Trash2, Undo2, X } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { CompareSlider } from '../components/CompareSlider'
import { ZoomView } from '../components/ZoomView'
import { GeneratingOverlay } from '../components/GeneratingOverlay'
import { RestoreBrush, type RestoreBrushHandle } from '../components/RestoreBrush'
import { TextureDrop } from '../components/inpaint/ActionPopover'
import { IconButton, SecondaryButton, Step } from '../components/StudioControls'
import { ToolButton } from '../components/ToolButton'
import { BrushOptions } from '../components/inpaint/Toolbar'
import { HARDNESS_DEFAULT, type DrawTool, type Shape } from '../lib/brush'
import { changedRegion, colorize, isNeutral, NEUTRAL, regionOverlay, type ColorValues } from '../lib/colorAdjust'
import { ColorPanel } from '../components/ColorPanel'
import { useScene, type Scene } from '../lib/scene'
import { img2imgMaterials, MATERIAL_STAGES } from '../lib/api'
import { fmt, modelById, useAccount } from '../lib/account'
import { formatSize, toStudioImage, urlToStudioImage, type StudioImage } from '../lib/types'

type Status = 'idle' | 'running' | 'done' | 'error'

interface Row {
  id: number
  texture: File | null
  target: string
  /** ползунок «Размер»: log2 множителя размера элементов, 0 — как на образце */
  size: number
}

interface Props {
  /** кадр, переданный из рендера («В Материалы»): становится исходником */
  incoming: StudioImage | null
  onSendToInpaint: (image: StudioImage) => void
}

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const
const MAX_ROWS = 3

const TARGET_CHIPS: Record<Scene, string[]> = {
  exterior: ['Стены дома', 'Цоколь', 'Кровля', 'Дорожки', 'Забор'],
  interior: ['Пол', 'Стены', 'Потолок', 'Столешница', 'Фартук'],
}

const TARGET_PLACEHOLDER: Record<Scene, string> = {
  exterior: 'Куда положить: например, на стены дома, кроме цоколя',
  interior: 'Куда положить: например, на пол во всей комнате',
}

let nextId = 1
const newRow = (): Row => ({ id: nextId++, texture: null, target: '', size: 0 })
// размер элементов ×0.25…×3 (как на бэкенде, MIN_SCALE / MAX_SCALE): шкала логарифмическая, ×1 посередине по смыслу
const SIZE_MIN = -2
const SIZE_MAX = Math.log2(3)
const sizeScale = (v: number) => Math.round(2 ** v * 100) / 100

export function ImgMaterials({ incoming, onSendToInpaint }: Props) {
  const { scene } = useScene()
  // модель — из селектора в шапке: у каждой свой пайплайн на бэкенде (texturing.py)
  const { charge, llm, model } = useAccount()
  const activeModel = modelById(model)
  const [source, setSource] = useState<StudioImage | null>(null)
  // предыдущие исходники — «Продолжить с результатом» можно откатить
  const [history, setHistory] = useState<StudioImage[]>([])
  const [rows, setRows] = useState<Row[]>(() => [newRow()])
  const [status, setStatus] = useState<Status>('idle')
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('')
  const [result, setResult] = useState<{ url: string; seed: number } | null>(null)
  const [view, setView] = useState<'compare' | 'result'>('compare')
  const [error, setError] = useState<string | null>(null)
  // кисть отката: под мазками результата проявляется кадр до материала
  const [brushing, setBrushing] = useState(false)
  const [brushSize, setBrushSize] = useState(80)
  const [brushTool, setBrushTool] = useState<DrawTool>('brush')
  const [brushShape, setBrushShape] = useState<Shape>('circle')
  const [hardness, setHardness] = useState(HARDNESS_DEFAULT)
  const [undoable, setUndoable] = useState(0)
  // применённый откат можно открыть снова: результат модели до отката и маска мазков
  const [raw, setRaw] = useState<string | null>(null)
  const [brushMask, setBrushMask] = useState<HTMLCanvasElement | null>(null)
  // цепочка результата: модель (raw) → откат кистью (restored) → цвет (color); result.url — итог
  const [restored, setRestored] = useState<string | null>(null)
  const [color, setColor] = useState<ColorValues>(NEUTRAL)
  const [colorOpen, setColorOpen] = useState(false)
  const [colorWorking, setColorWorking] = useState(false)
  const [showRegion, setShowRegion] = useState(false)
  const [regionUrl, setRegionUrl] = useState<string | null>(null)
  // область изменений — для того кадра, по которому посчитана (откат её меняет)
  const region = useRef<{ url: string; mask: HTMLCanvasElement } | null>(null)
  const colorTimer = useRef(0)
  // сдвиг прежнего кадра под кистью: стрелки — на 1 px, с Shift — на 5
  const [nudge, setNudge] = useState({ x: 0, y: 0 })
  const brush = useRef<RestoreBrushHandle>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => {
    if (!brushing) return
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea')) return
      if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') { e.preventDefault(); brush.current?.undo(); return }
      if (e.ctrlKey || e.metaKey) return
      const tools: Record<string, DrawTool> = { KeyB: 'brush', KeyU: 'rect', KeyP: 'poly' }
      if (tools[e.code]) { setBrushTool(tools[e.code]); return }
      if (e.code === 'BracketLeft') { setBrushSize((v) => Math.max(10, v - 8)); return }
      if (e.code === 'BracketRight') { setBrushSize((v) => Math.min(300, v + 8)); return }
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]
      if (!d) return
      e.preventDefault()
      const step = e.shiftKey ? 5 : 1
      setNudge((n) => ({ x: n.x + d[0] * step, y: n.y + d[1] * step }))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [brushing])
  useEffect(() => {
    if (incoming) takeSource(incoming)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incoming])

  const running = status === 'running'
  const ready = rows.filter((r) => r.texture && r.target.trim())
  const incomplete = rows.some((r) => !!r.texture !== !!r.target.trim())
  const canGenerate = !!source && !running && ready.length > 0 && !incomplete

  function takeSource(img: StudioImage) {
    abort.current?.abort()
    setBrushing(false)
    setSource(img)
    setHistory([])
    setResult(null); setRaw(null); setBrushMask(null); resetColor()
    setStatus('idle')
    setError(null)
  }

  function update(id: number, patch: Partial<Row>) {
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)))
  }

  function addChip(id: number, chip: string) {
    const row = rows.find((r) => r.id === id)
    if (!row) return
    const base = row.target.trim().replace(/[,;]$/, '')
    update(id, { target: (base ? `${base}, ${chip.toLowerCase()}` : chip).slice(0, 300) })
  }

  async function generate() {
    if (!source || !canGenerate) return
    const ctrl = new AbortController()
    abort.current = ctrl
    setBrushing(false)
    setStatus('running')
    setProgress(0)
    setError(null)
    try {
      const res = await img2imgMaterials(
        { image: source.file, imageUrl: source.url, llm, model, items: ready.map((r) => ({ texture: r.texture!, target: r.target, scale: sizeScale(r.size) })) },
        (p, s) => { setProgress(p); setStage(s) },
        ctrl.signal,
      )
      setResult(res)
      setRaw(null) // новый результат — прежний откат и цвет к нему не относятся
      setBrushMask(null)
      resetColor()
      for (const r of ready) charge('render', `Img to Img · Материалы · ${r.target.trim()}`, model)
      setView('compare')
      setStatus('done')
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        setStatus(result ? 'done' : 'idle')
        return
      }
      setError((e as Error).message)
      setStatus('error')
    }
  }

  const fileName = () => `inego_materials_${source?.name.replace(/\.[^.]+$/, '') ?? 'frame'}.jpg`

  function download() {
    if (!result) return
    const a = document.createElement('a')
    a.href = result.url
    a.download = fileName()
    a.click()
  }

  // следующий материал — поверх уже положенного
  async function continueFromResult() {
    if (!result || !source) return
    const img = await urlToStudioImage(result.url, fileName())
    setHistory((h) => [...h, source])
    setSource(img)
    setResult(null); setRaw(null); setBrushMask(null); resetColor()
    setStatus('idle')
    setRows([newRow()])
  }

  function resetColor() {
    setRestored(null)
    setColor(NEUTRAL)
    setColorOpen(false)
    setShowRegion(false)
    setRegionUrl(null)
    region.current = null
  }

  // кадр без цвета: откат, иначе результат модели
  const plain = restored ?? raw ?? result?.url ?? null

  async function regionFor(url: string) {
    if (region.current?.url !== url) region.current = { url, mask: await changedRegion(source!.url, url) }
    return region.current.mask
  }

  /** итог цепочки: цвет поверх кадра без цвета (только в области изменений) */
  async function finalUrl(base: string, v: ColorValues) {
    return isNeutral(v) ? base : colorize(base, await regionFor(base), v)
  }

  // мазки кисти отката — в результат: дальше скачивается, продолжается и уходит в Inpainting уже он; цвет, если
  // был, ложится поверх заново — область по откату другая
  async function applyBrush() {
    if (!result || !brush.current) return
    const blob = await brush.current.toBlob()
    const url = URL.createObjectURL(blob)
    setBrushMask(brush.current.mask())
    setRaw((r) => r ?? result.url)
    setRestored(url)
    setResult({ ...result, url: await finalUrl(url, color) })
    setBrushing(false)
    setView('result')
  }

  // ползунки цвета — пересборка с задержкой, пока тянут
  function changeColor(v: ColorValues) {
    if (!result || !plain) return
    setColor(v)
    setRaw((r) => r ?? plain)
    clearTimeout(colorTimer.current)
    const base = plain
    colorTimer.current = window.setTimeout(async () => {
      setColorWorking(true)
      try {
        const url = await finalUrl(base, v)
        setResult((r) => (r ? { ...r, url } : r))
      } finally {
        setColorWorking(false)
      }
    }, 120)
  }

  async function toggleRegion(on: boolean) {
    setShowRegion(on)
    if (on && plain) setRegionUrl(await regionOverlay(await regionFor(plain)))
  }

  function undo() {
    const prev = history[history.length - 1]
    if (!prev) return
    setHistory((h) => h.slice(0, -1))
    setSource(prev)
    setResult(null); setRaw(null); setBrushMask(null); resetColor()
    setStatus('idle')
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.1 }}
      className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px] lg:gap-6"
    >
      {/* Canvas */}
      <div className="flex min-w-0 flex-col gap-4">
        <div className="relative h-[400px] overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04] sm:aspect-[16/10] sm:h-auto sm:min-h-0">
          <AnimatePresence mode="wait" initial={false}>
            {!source ? (
              <motion.div key="drop" className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
                <Dropzone onFile={async (f) => !running && takeSource(await toStudioImage(f))} />
              </motion.div>
            ) : (
              <motion.div
                key={source.url}
                className="checker absolute inset-0"
                initial={{ opacity: 0, scale: 1.02 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
              >
                <ZoomView aspect={source.width / source.height} panWithLeft={!(result && (view === 'compare' || brushing))}>
                  {(imageStyle) =>
                    result && !running ? (
                      brushing ? (
                        <RestoreBrush
                          ref={brush} before={source.url} after={raw ?? result.url} initialMask={brushMask} imageStyle={imageStyle} size={brushSize}
                          onHistory={setUndoable} offset={nudge} tool={brushTool} shape={brushShape} hardness={hardness}
                        />
                      ) : colorOpen ? (
                        <>
                          <img src={result.url} alt="Результат" className="absolute inset-0 size-full object-contain" style={imageStyle} draggable={false} />
                          {showRegion && regionUrl && (
                            <img src={regionUrl} alt="" className="pointer-events-none absolute inset-0 size-full object-contain" style={imageStyle} draggable={false} />
                          )}
                        </>
                      ) : view === 'compare' ? (
                        <CompareSlider before={source.url} after={result.url} imageStyle={imageStyle} />
                      ) : (
                        <img src={result.url} alt="Результат" className="absolute inset-0 size-full object-contain" style={imageStyle} draggable={false} />
                      )
                    ) : (
                      <img src={source.url} alt={source.name} className="absolute inset-0 size-full object-contain" style={imageStyle} draggable={false} />
                    )
                  }
                </ZoomView>

                <AnimatePresence>
                  {running && (
                    <motion.div key="gen" className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>
                      <GeneratingOverlay
                        image={source.url}
                        progress={progress}
                        stage={stage}
                        stages={MATERIAL_STAGES}
                        presetSky="linear-gradient(135deg, #c8b49a, #8a7a6a)"
                        onCancel={() => abort.current?.abort()}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>

                {colorOpen && result && !running && !brushing && (
                  <div className="absolute left-4 top-4 z-20">
                    <ColorPanel
                      values={color} onChange={changeColor} working={colorWorking}
                      showRegion={showRegion} onShowRegion={toggleRegion}
                      onDone={() => { setColorOpen(false); setShowRegion(false) }}
                    />
                  </div>
                )}

                {brushing && result && !running && (
                  <BrushOverlay
                    tool={brushTool} onTool={setBrushTool}
                    shape={brushShape} onShape={setBrushShape}
                    size={brushSize} onSize={setBrushSize}
                    hardness={hardness} onHardness={setHardness}
                    undoable={undoable} onUndo={() => brush.current?.undo()} onClear={() => brush.current?.clear()}
                    nudge={nudge} onResetNudge={() => setNudge({ x: 0, y: 0 })}
                    onClose={() => setBrushing(false)} onApply={applyBrush}
                  />
                )}

                {!running && !brushing && (
                  <div className="pointer-events-none absolute inset-x-4 bottom-4 flex items-end justify-between gap-3">
                    <div className="glass pointer-events-auto flex min-w-0 items-center gap-2.5 rounded-full py-1.5 pl-1.5 pr-4 shadow-soft ring-1 ring-fg/5">
                      <img src={source.url} alt="" className="size-7 rounded-full object-cover" />
                      <div className="min-w-0 text-[12px] leading-tight">
                        <div className="truncate font-medium">{source.name}</div>
                        <div className="text-faint">{source.width}×{source.height} · {formatSize(source.file.size)}</div>
                      </div>
                    </div>
                    <div className="pointer-events-auto flex gap-2">
                      {result && !brushing && (
                        <div className="glass flex rounded-full p-1 shadow-soft ring-1 ring-fg/5">
                          {(['compare', 'result'] as const).map((v) => (
                            <button
                              key={v}
                              onClick={() => setView(v)}
                              className={`relative rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors ${view === v ? 'text-on-ink' : 'text-ink-2 hover:text-ink'}`}
                            >
                              {view === v && <motion.span layoutId="mat-view-pill" className="absolute inset-0 -z-10 rounded-full bg-ink" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
                              {v === 'compare' ? 'Сравнить' : 'Результат'}
                            </button>
                          ))}
                        </div>
                      )}
                      {history.length > 0 && !result && (
                        <IconButton label="Вернуть предыдущий кадр" onClick={undo}>
                          <Undo2 className="size-4" />
                        </IconButton>
                      )}
                      <IconButton label="Убрать изображение" onClick={() => { abort.current?.abort(); setSource(null); setHistory([]); setResult(null); setRaw(null); setBrushMask(null); resetColor(); setStatus('idle') }}>
                        <X className="size-4" />
                      </IconButton>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Result actions */}
        <AnimatePresence>
          {result && status === 'done' && !brushing && (
            <motion.div
              key="actions"
              initial={{ opacity: 0, y: -8, height: 0 }} animate={{ opacity: 1, y: 0, height: 'auto' }} exit={{ opacity: 0, y: -8, height: 0 }}
              transition={spring}
            >
              <div className="flex flex-col gap-4 rounded-[22px] bg-surface p-4 shadow-card ring-1 ring-fg/[0.04] sm:flex-row sm:items-center sm:justify-between sm:pl-5">
                <div>
                  <div className="text-[15px] font-semibold tracking-[-0.01em]">Готово · {ready.length === 1 ? '1 материал' : `${ready.length} материала`}</div>
                  <div className="text-[13px] text-mute">Сид <span className="font-medium tabular-nums text-ink-2">{result.seed}</span></div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <SecondaryButton onClick={download}>
                    <Download className="size-4" /> Скачать
                  </SecondaryButton>
                  {/* откат уже применён — открывается он же: те же мазки, край и сдвиг можно поменять */}
                  <SecondaryButton onClick={() => { setColorOpen(!colorOpen); setBrushing(false); setView('result') }}>
                    <SunMedium className="size-4" /> {isNeutral(color) ? 'Цвет' : 'Править цвет'}
                  </SecondaryButton>
                  <SecondaryButton onClick={() => { setColorOpen(false); setUndoable(0); if (!brushMask) setNudge({ x: 0, y: 0 }); setBrushing(true) }}>
                    <Eraser className="size-4" /> {brushMask ? 'Править откат' : 'Кисть отката'}
                  </SecondaryButton>
                  <SecondaryButton onClick={continueFromResult}>
                    <Plus className="size-4" /> Ещё материал
                  </SecondaryButton>
                  <motion.button
                    whileTap={{ scale: 0.97 }}
                    onClick={async () => onSendToInpaint(await urlToStudioImage(result.url, fileName()))}
                    className="group flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-5 py-2.5 text-[14px] font-medium text-on-ink transition-colors hover:bg-ink-hover sm:flex-none"
                  >
                    В Inpainting
                    <ArrowRight className="size-4 transition-transform duration-300 ease-apple group-hover:translate-x-0.5" />
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Control panel */}
      <aside className="lg:sticky lg:top-20 lg:self-start">
        <div className="rounded-[28px] bg-surface p-5 shadow-card ring-1 ring-fg/[0.04] sm:p-6">
          <Step n={1} title="Образцы" hint="Текстура и куда её положить — нейросеть найдёт поверхность сама" />

          <div className="mt-2 flex flex-col gap-3">
            <AnimatePresence initial={false}>
              {rows.map((r, i) => (
                <motion.div
                  key={r.id}
                  initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  transition={spring} className="overflow-hidden"
                >
                  <div className={`relative ${i > 0 ? 'border-t border-line pt-1' : ''}`}>
                    {rows.length > 1 && (
                      <button
                        aria-label="Убрать образец"
                        title="Убрать образец"
                        disabled={running}
                        onClick={() => setRows((rs) => rs.filter((x) => x.id !== r.id))}
                        className="absolute right-0 top-2 z-10 grid size-7 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink disabled:opacity-40"
                      >
                        <Trash2 className="size-3.5" />
                      </button>
                    )}
                    <div className={rows.length > 1 ? 'pr-8' : ''}>
                      <TextureDrop texture={r.texture} onTexture={(f) => update(r.id, { texture: f })} width="" />
                    </div>
                    {r.texture && <SizeSlider value={r.size} onChange={(v) => update(r.id, { size: v })} disabled={running} />}
                    <textarea
                      value={r.target}
                      onChange={(e) => update(r.id, { target: e.target.value })}
                      disabled={running}
                      rows={2}
                      maxLength={300}
                      placeholder={TARGET_PLACEHOLDER[scene]}
                      className="mt-2 w-full resize-none rounded-[14px] bg-fg/[0.04] px-3.5 py-2.5 text-[14px] leading-snug outline-none ring-accent/40 transition-shadow placeholder:text-faint focus:bg-surface focus:ring-2 disabled:opacity-60"
                    />
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {TARGET_CHIPS[scene].map((c) => (
                        <button
                          key={c}
                          type="button"
                          disabled={running}
                          onClick={() => addChip(r.id, c)}
                          className="rounded-full px-2.5 py-0.5 text-[12px] font-medium text-ink-2 ring-1 ring-fg/[0.08] transition-all duration-200 hover:bg-fg/[0.04] hover:text-ink active:scale-95 disabled:opacity-50"
                        >
                          + {c}
                        </button>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {rows.length < MAX_ROWS && (
            <button
              onClick={() => setRows((rs) => [...rs, newRow()])}
              disabled={running}
              className="mt-3 flex w-full items-center justify-center gap-1.5 rounded-full py-2 text-[13px] font-medium text-ink-2 ring-1 ring-fg/[0.08] transition-colors hover:bg-fg/[0.04] hover:text-ink disabled:opacity-50"
            >
              <Plus className="size-4" /> Ещё образец
            </button>
          )}

          <div className="my-6 h-px bg-line" />

          <motion.button
            whileTap={canGenerate ? { scale: 0.98 } : undefined}
            onClick={generate}
            disabled={!canGenerate}
            className="relative flex w-full items-center justify-center gap-2 overflow-hidden rounded-full bg-accent py-3.5 text-[16px] font-medium text-white transition-all duration-300 ease-apple hover:bg-accent-hover disabled:bg-fg/[0.08] disabled:text-faint"
          >
            {running ? (
              <><RefreshCw className="size-[18px] animate-spin" /> Кладём материал…</>
            ) : result ? (
              <><RotateCcw className="size-[18px]" /> Сгенерировать заново</>
            ) : (
              <><Sparkles className="size-[18px]" /> Применить</>
            )}
          </motion.button>
          <p className="mt-3 text-center text-[12px] text-faint">
            {!source
              ? 'Загрузите клей, технический или готовый рендер'
              : incomplete || ready.length === 0
                ? 'К каждому образцу — куда его положить'
                : `${activeModel.name} · ${activeModel.time}${ready.length > 1 ? ` × ${ready.length}` : ''} · ${fmt(activeModel.cost * ready.length)} ток.`}
          </p>

          <AnimatePresence>
            {error && (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-3 rounded-xl bg-danger/10 px-3 py-2 text-[13px] text-danger">
                {error}
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      </aside>
    </motion.section>
  )
}

/** Размер элементов материала: мельче — образец мостится, крупнее — берётся его часть (словами 2511 масштаб не слушает) */
function SizeSlider({ value, onChange, disabled }: { value: number; onChange: (v: number) => void; disabled?: boolean }) {
  const k = sizeScale(value)
  const same = Math.abs(value) < 0.01
  return (
    <div className="mt-2.5 px-0.5">
      <div className="mb-1.5 flex items-baseline justify-between text-[12.5px]">
        <span className="text-mute">Размер элементов</span>
        <button
          type="button"
          disabled={disabled || same}
          onClick={() => onChange(0)}
          title={same ? undefined : 'Как на образце'}
          className="font-semibold tabular-nums text-ink transition-colors enabled:hover:text-accent"
        >
          {same ? 'как на образце' : `×${k.toLocaleString('ru-RU')}`}
        </button>
      </div>
      <input
        // шаг 0.05 от −2: ×1 (0) лежит на сетке — до него легко дойти и мышью, и стрелками
        type="range" min={SIZE_MIN} max={SIZE_MAX} step={0.05} value={value} disabled={disabled}
        aria-label="Размер элементов материала"
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={() => onChange(0)}
        className="range w-full"
        style={{ '--fill': `${((value - SIZE_MIN) / (SIZE_MAX - SIZE_MIN)) * 100}%` } as React.CSSProperties}
      />
      <div className="mt-1 flex justify-between text-[11px] text-faint">
        <span>Мельче</span>
        <span>Крупнее</span>
      </div>
    </div>
  )
}

/** Кисть отката на кадре: инструменты слева, тонкие настройки снизу, действия сверху — как в Inpainting. */
function BrushOverlay({
  tool, onTool, shape, onShape, size, onSize, hardness, onHardness, undoable, onUndo, onClear, nudge, onResetNudge, onClose, onApply,
}: {
  tool: DrawTool
  onTool: (t: DrawTool) => void
  shape: Shape
  onShape: (s: Shape) => void
  size: number
  onSize: (n: number) => void
  hardness: number
  onHardness: (v: number) => void
  undoable: number
  onUndo: () => void
  onClear: () => void
  nudge: { x: number; y: number }
  onResetNudge: () => void
  onClose: () => void
  onApply: () => void
}) {
  const tools: { id: DrawTool; icon: typeof Brush; label: string; key: string }[] = [
    { id: 'brush', icon: Brush, label: 'Кисть — Shift+клик: прямая от конца мазка, Shift в мазке: по горизонтали / вертикали', key: 'B' },
    { id: 'rect', icon: SquareDashed, label: 'Прямоугольник — протяните; Shift — квадрат', key: 'U' },
    { id: 'poly', icon: Pentagon, label: 'Многоугольник — точки кликами; Enter или клик в первую точку', key: 'P' },
  ]
  const shifted = nudge.x !== 0 || nudge.y !== 0
  return (
    <>
      {/* сверху: что делает кисть и действия */}
      <div className="pointer-events-none absolute inset-x-0 top-4 z-20 flex justify-center px-16">
        <div className="glass pointer-events-auto flex items-center gap-1 rounded-full p-1 shadow-float ring-1 ring-fg/5">
          <span className="hidden items-center gap-1.5 pl-3 pr-1 text-[12.5px] text-ink-2 md:flex">
            <Eraser className="size-3.5" /> Вернуть кадр до материала · Alt — результат · стрелки — сдвиг
            {shifted && (
              <button onClick={onResetNudge} className="tabular-nums font-medium text-ink hover:text-accent" title="Сбросить сдвиг">
                {nudge.x > 0 ? '→' : '←'}{Math.abs(nudge.x)} {nudge.y > 0 ? '↓' : '↑'}{Math.abs(nudge.y)} px
              </button>
            )}
          </span>
          <button onClick={onClose} className="rounded-full px-3 py-1.5 text-[12.5px] font-medium text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink">
            Закрыть
          </button>
          <button onClick={onApply} className="flex items-center gap-1.5 rounded-full bg-ink px-3.5 py-1.5 text-[12.5px] font-medium text-on-ink transition-colors hover:bg-ink-hover">
            <Check className="size-3.5" /> Применить
          </button>
        </div>
      </div>

      {/* слева: инструменты */}
      <div className="absolute left-4 top-1/2 z-20 -translate-y-1/2">
        <div className="glass flex flex-col gap-1 rounded-[18px] p-1.5 shadow-float ring-1 ring-fg/5">
          {tools.map(({ id, icon: Icon, label, key }) => (
            <ToolButton key={id} label={label} shortcut={key} onClick={() => onTool(id)} active={tool === id}>
              {tool === id && <motion.span layoutId="mat-brush-tool" className="absolute inset-0 -z-10 rounded-xl bg-ink" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
              <Icon className="size-[18px]" strokeWidth={1.75} />
            </ToolButton>
          ))}
          <div className="mx-2 my-1 h-px bg-fg/10" />
          <ToolButton label="Отменить" shortcut="Ctrl Z" onClick={onUndo} disabled={undoable === 0}>
            <Undo2 className="size-[18px]" strokeWidth={1.75} />
          </ToolButton>
          <ToolButton label="Сбросить все мазки" shortcut="" onClick={onClear}>
            <Trash2 className="size-[18px]" strokeWidth={1.75} />
          </ToolButton>
        </div>
      </div>

      {/* снизу: форма, размер, край */}
      <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-4">
        <div className="pointer-events-auto">
          <BrushOptions shape={shape} onShape={onShape} size={size} onSize={onSize} min={10} max={300} hardness={hardness} onHardness={onHardness} />
        </div>
      </div>
    </>
  )
}
