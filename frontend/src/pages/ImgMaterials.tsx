import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Check, Download, Eraser, Plus, RefreshCw, RotateCcw, Sparkles, Trash2, Undo2, X } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { CompareSlider } from '../components/CompareSlider'
import { ZoomView } from '../components/ZoomView'
import { GeneratingOverlay } from '../components/GeneratingOverlay'
import { RestoreBrush, type RestoreBrushHandle } from '../components/RestoreBrush'
import { TextureDrop } from '../components/inpaint/ActionPopover'
import { IconButton, SecondaryButton, Step } from '../components/StudioControls'
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
  const [undoable, setUndoable] = useState(0)
  const brush = useRef<RestoreBrushHandle>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => () => abort.current?.abort(), [])
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
    setResult(null)
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
    setResult(null)
    setStatus('idle')
    setRows([newRow()])
  }

  // мазки кисти отката — в результат: дальше скачивается, продолжается и уходит в Inpainting уже он
  async function applyBrush() {
    if (!result || !brush.current) return
    const blob = await brush.current.toBlob()
    setResult({ ...result, url: URL.createObjectURL(blob) })
    setBrushing(false)
    setView('result')
  }

  function undo() {
    const prev = history[history.length - 1]
    if (!prev) return
    setHistory((h) => h.slice(0, -1))
    setSource(prev)
    setResult(null)
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
                          ref={brush} before={source.url} after={result.url} imageStyle={imageStyle} size={brushSize}
                          onHistory={setUndoable}
                        />
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

                {!running && (
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
                      <IconButton label="Убрать изображение" onClick={() => { abort.current?.abort(); setSource(null); setHistory([]); setResult(null); setStatus('idle') }}>
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
          {result && status === 'done' && brushing && (
            <motion.div
              key="brush"
              initial={{ opacity: 0, y: -8, height: 0 }} animate={{ opacity: 1, y: 0, height: 'auto' }} exit={{ opacity: 0, y: -8, height: 0 }}
              transition={spring}
            >
              <div className="flex flex-col gap-4 rounded-[22px] bg-surface p-4 shadow-card ring-1 ring-fg/[0.04] sm:flex-row sm:items-center sm:justify-between sm:pl-5">
                <div className="min-w-0 flex-1">
                  <div className="text-[15px] font-semibold tracking-[-0.01em]">Кисть отката</div>
                  <div className="text-[13px] text-mute">Проведите — вернётся кадр до материала. Alt — вернуть результат, пробел — двигать кадр</div>
                  <label className="mt-2.5 flex items-center gap-3 text-[12.5px] text-mute">
                    <span className="shrink-0">Размер</span>
                    <input
                      type="range" min={10} max={300} step={2} value={brushSize}
                      onChange={(e) => setBrushSize(Number(e.target.value))}
                      className="range w-full max-w-[220px]"
                      style={{ '--fill': `${((brushSize - 10) / 290) * 100}%` } as React.CSSProperties}
                    />
                    <span className="w-12 shrink-0 tabular-nums text-ink-2">{brushSize} px</span>
                  </label>
                </div>
                <div className="flex flex-wrap gap-2">
                  <SecondaryButton onClick={() => brush.current?.undo()} disabled={undoable === 0}>
                    <Undo2 className="size-4" /> Отменить
                  </SecondaryButton>
                  <SecondaryButton onClick={() => brush.current?.clear()}>
                    <RotateCcw className="size-4" /> Сбросить
                  </SecondaryButton>
                  <SecondaryButton onClick={() => setBrushing(false)}>Закрыть</SecondaryButton>
                  <motion.button
                    whileTap={{ scale: 0.97 }}
                    onClick={applyBrush}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-5 py-2.5 text-[14px] font-medium text-on-ink transition-colors hover:bg-ink-hover sm:flex-none"
                  >
                    <Check className="size-4" /> Применить
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
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
                  <SecondaryButton onClick={() => { setUndoable(0); setBrushing(true) }}>
                    <Eraser className="size-4" /> Кисть отката
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
