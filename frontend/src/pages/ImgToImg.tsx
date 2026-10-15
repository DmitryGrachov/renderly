import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Dices, Download, Lock, LockOpen, RotateCcw, Sparkles, X, RefreshCw } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { PresetCard } from '../components/PresetCard'
import { CompareSlider } from '../components/CompareSlider'
import { ZoomView } from '../components/ZoomView'
import { GeneratingOverlay } from '../components/GeneratingOverlay'
import { CUSTOM_PRESET, presetById, presetsFor, type PresetId } from '../lib/presets'
import { SCENE_LABEL, useScene, type Scene } from '../lib/scene'
import { img2img } from '../lib/api'
import { fmt, modelById, useAccount } from '../lib/account'
import { ModelSelect } from '../components/ModelSelect'
import { formatSize, toStudioImage, urlToStudioImage, type StudioImage } from '../lib/types'

type Status = 'idle' | 'running' | 'done' | 'error'

interface Props {
  onSendToInpaint: (image: StudioImage) => void
}

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const

const SUBTITLE: Record<Scene, string> = {
  exterior: 'Загрузите визуализацию фасада — нейросеть пересветит сцену, сохранив архитектуру и материалы.',
  interior: 'Загрузите визуализацию интерьера — нейросеть перестроит свет, сохранив мебель, отделку и материалы.',
}

export function ImgToImg({ onSendToInpaint }: Props) {
  const { scene } = useScene()
  const { charge, model, setModel } = useAccount()
  const activeModel = modelById(model)
  const [source, setSource] = useState<StudioImage | null>(null)
  const [reference, setReference] = useState<StudioImage | null>(null)
  const [preset, setPreset] = useState<PresetId>('golden')
  const [strength, setStrength] = useState(0.7)
  const [materials, setMaterials] = useState('')
  const [keepSky, setKeepSky] = useState(false)
  const [keepBg, setKeepBg] = useState(false)
  const [keepSur, setKeepSur] = useState(false)
  // Сид: пока замок закрыт, каждая генерация идёт с ним — картинку можно «добивать» материалами и пресетом
  const [seed, setSeed] = useState('')
  const [seedLocked, setSeedLocked] = useState(false)
  const [status, setStatus] = useState<Status>('idle')
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('')
  const [result, setResult] = useState<{ url: string; preset: PresetId; scene: Scene; seed: number } | null>(null)
  const [view, setView] = useState<'compare' | 'result'>('compare')
  const [error, setError] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const hintRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => () => abort.current?.abort(), [])

  const running = status === 'running'
  const canGenerate = !!source && !running && (preset !== 'custom' || !!reference)

  // чип подсказки: дописывает «Кровля — » и ставит курсор в конец; пустой незаполненный пункт заменяется
  function addHint(label: string) {
    const base = materials.trimEnd().replace(/[,;]$/, '').replace(/(^|,\s*)[^,—]*—$/, '').trimEnd()
    const next = (base ? `${base}, ${label.toLowerCase()} — ` : `${label} — `).slice(0, 600)
    setMaterials(next)
    requestAnimationFrame(() => {
      const el = hintRef.current
      if (!el) return
      el.focus()
      el.setSelectionRange(next.length, next.length)
    })
  }

  async function pickSource(file: File) {
    if (running) return
    const img = await toStudioImage(file)
    setSource(img)
    setResult(null)
    setStatus('idle')
    setError(null)
  }

  function reset() {
    abort.current?.abort()
    setSource(null)
    setResult(null)
    setStatus('idle')
  }

  async function generate() {
    if (!source || !canGenerate) return
    const ctrl = new AbortController()
    abort.current = ctrl
    setStatus('running')
    setProgress(0)
    setError(null)
    try {
      const fixedSeed = seedLocked && seed !== '' ? Number(seed) : null
      const res = await img2img(
        { image: source.file, imageUrl: source.url, model, scene, preset, strength, materials, keepSky, keepBg, keepSur, seed: fixedSeed, reference: reference?.file, referenceUrl: reference?.url },
        (p, s) => { setProgress(p); setStage(s) },
        ctrl.signal,
      )
      setResult({ url: res.url, preset, scene, seed: res.seed })
      charge('render', `Img to Img · ${presetById(preset, scene).title}`)
      if (!seedLocked) setSeed(String(res.seed))
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

  function download() {
    if (!result || !source) return
    const a = document.createElement('a')
    a.href = result.url
    a.download = `inego_${result.scene}_${result.preset}_${source.name.replace(/\.[^.]+$/, '')}.jpg`
    a.click()
  }

  async function sendToInpaint() {
    if (!result || !source) return
    const name = `inego_${result.scene}_${result.preset}_${source.name.replace(/\.[^.]+$/, '')}.jpg`
    onSendToInpaint(await urlToStudioImage(result.url, name))
  }

  const activePreset = presetById(preset, scene)

  return (
    <div className="mx-auto max-w-[1280px] px-5 pb-24 sm:px-8">
      {/* Hero */}
      <section className="pb-10 pt-12 text-center sm:pb-12 sm:pt-16">
        <motion.p
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.05 }}
          className="text-[15px] font-semibold text-gradient sm:text-[17px]"
        >
          Img to Img · {SCENE_LABEL[scene]}
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.1 }}
          className="mt-3 text-[44px] font-semibold leading-[1.04] tracking-[-0.035em] sm:text-[64px]"
        >
          Один рендер.
          <br />
          <span className="text-faint">Любой свет.</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.18 }}
          className="mx-auto mt-5 max-w-[560px] text-[17px] leading-relaxed text-mute sm:text-[19px]"
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={scene} className="block" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              {SUBTITLE[scene]}
            </motion.span>
          </AnimatePresence>
        </motion.p>
      </section>

      {/* How it works */}
      <motion.section
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.22 }}
        className="mb-6 grid gap-px overflow-hidden rounded-[28px] bg-line ring-1 ring-fg/[0.04] sm:grid-cols-3">
        {[
          ['01', 'Загрузите', 'Рендер из 3ds Max, Revit, SketchUp или даже скриншот вьюпорта.'],
          ['02', 'Выберите свет', 'Четыре выверенных сценария или собственный референс.'],
          ['03', 'Доведите', 'Отправьте результат в Inpainting и поправьте детали кистью.'],
        ].map(([n, t, d]) => (
          <div key={n} className="bg-surface p-6 sm:px-7">
            <div className="text-[13px] font-semibold text-gradient">{n}</div>
            <div className="mt-2 text-[19px] font-semibold tracking-[-0.02em]">{t}</div>
            <p className="mt-1.5 text-[14px] leading-relaxed text-mute">{d}</p>
          </div>
        ))}
      </motion.section>

      {/* Workspace */}
      <motion.section
        initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.3 }}
        className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px] lg:gap-6"
      >
        {/* Canvas */}
        <div className="flex min-w-0 flex-col gap-4">
          <div className="relative aspect-[4/3] min-h-[400px] overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04] sm:aspect-[16/10] sm:min-h-0">
            <AnimatePresence mode="wait" initial={false}>
              {!source ? (
                <motion.div key="drop" className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
                  <Dropzone onFile={pickSource} listenPaste />
                </motion.div>
              ) : (
                <motion.div
                  key={source.url}
                  className="checker absolute inset-0"
                  initial={{ opacity: 0, scale: 1.02 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
                >
                  {/* масштаб сохраняется при переключении «Сравнить / Результат» — удобно смотреть одну деталь */}
                  <ZoomView aspect={source.width / source.height} panWithLeft={!(result && view === 'compare')}>
                    {(imageStyle) =>
                      result && status !== 'running' ? (
                        view === 'compare' ? (
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
                          presetSky={activePreset.sky}
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
                        {result && (
                          <div className="glass flex rounded-full p-1 shadow-soft ring-1 ring-fg/5">
                            {(['compare', 'result'] as const).map((v) => (
                              <button
                                key={v}
                                onClick={() => setView(v)}
                                className={`relative rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors ${view === v ? 'text-on-ink' : 'text-ink-2 hover:text-ink'}`}
                              >
                                {view === v && <motion.span layoutId="view-pill" className="absolute inset-0 -z-10 rounded-full bg-ink" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
                                {v === 'compare' ? 'Сравнить' : 'Результат'}
                              </button>
                            ))}
                          </div>
                        )}
                        <IconButton label="Убрать изображение" onClick={reset}>
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
            {result && status === 'done' && (
              <motion.div
                initial={{ opacity: 0, y: -8, height: 0 }} animate={{ opacity: 1, y: 0, height: 'auto' }} exit={{ opacity: 0, y: -8, height: 0 }}
                transition={spring}
              >
                <div className="flex flex-col gap-4 rounded-[22px] bg-surface p-4 shadow-card ring-1 ring-fg/[0.04] sm:flex-row sm:items-center sm:justify-between sm:pl-5">
                  <div className="flex items-center gap-3">
                    <span className="size-10 shrink-0 rounded-xl ring-1 ring-fg/5" style={{ background: presetById(result.preset, result.scene).sky }} />
                    <div>
                      <div className="text-[15px] font-semibold tracking-[-0.01em]">Готово · {presetById(result.preset, result.scene).title} · {SCENE_LABEL[result.scene]}</div>
                      <div className="flex flex-wrap items-center gap-x-2 text-[13px] text-mute">
                        <span>{source?.width}×{source?.height}</span>
                        <span aria-hidden>·</span>
                        <span>Сид <span className="font-medium tabular-nums text-ink-2">{result.seed}</span></span>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <SecondaryButton onClick={download}>
                      <Download className="size-4" /> Скачать
                    </SecondaryButton>
                    <motion.button
                      whileTap={{ scale: 0.97 }}
                      onClick={sendToInpaint}
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

          {/* Seed + prompt hint */}
          <div className="overflow-hidden rounded-[22px] bg-surface shadow-card ring-1 ring-fg/[0.04]">
            <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:gap-6 sm:pl-5">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-[15px] font-semibold tracking-[-0.01em]">
                  Сид
                  <span
                    className={`rounded-full px-2 py-0.5 text-[11px] font-semibold transition-colors duration-300 ${
                      seedLocked ? 'bg-accent/10 text-accent' : 'bg-fg/[0.05] text-mute'
                    }`}
                  >
                    {seedLocked ? 'Зафиксирован' : 'Случайный'}
                  </span>
                </div>
                <div className="text-[13px] text-mute">
                  {seedLocked ? 'Повторяет композицию — меняйте свет и подсказку поверх неё' : 'Каждая генерация — новый вариант'}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  aria-label="Сид"
                  inputMode="numeric"
                  value={seed}
                  disabled={running}
                  placeholder="случайный"
                  onChange={(e) => {
                    const v = e.target.value.replace(/\D/g, '').slice(0, 14)
                    setSeed(v)
                    setSeedLocked(v !== '')
                  }}
                  className="w-full min-w-0 rounded-full bg-fg/[0.04] px-4 py-2 text-[14px] tabular-nums outline-none ring-accent/40 transition-shadow placeholder:text-faint focus:bg-surface focus:ring-2 disabled:opacity-60 sm:w-[168px]"
                />
                <button
                  type="button"
                  onClick={() => { setSeed(String(Math.floor(Math.random() * 2 ** 32))); setSeedLocked(true) }}
                  disabled={running}
                  aria-label="Случайный сид"
                  title="Новый случайный сид"
                  className="group grid size-9 shrink-0 place-items-center rounded-full bg-fg/[0.05] text-ink-2 transition-all duration-300 ease-apple hover:bg-fg/[0.08] hover:text-ink active:scale-95 disabled:opacity-40"
                >
                  <Dices className="size-4 transition-transform duration-500 ease-apple group-hover:rotate-[24deg]" />
                </button>
                <button
                  type="button"
                  onClick={() => setSeedLocked((v) => !v && seed !== '')}
                  disabled={running || seed === ''}
                  aria-label={seedLocked ? 'Снять фиксацию сида' : 'Зафиксировать сид'}
                  title={seedLocked ? 'Сид зафиксирован — повторяет картинку' : 'Каждый раз новый сид'}
                  className={`grid size-9 shrink-0 place-items-center rounded-full transition-all duration-300 ease-apple active:scale-95 disabled:opacity-40 ${
                    seedLocked ? 'bg-accent text-white' : 'bg-fg/[0.05] text-ink-2 hover:bg-fg/[0.08]'
                  }`}
                >
                  {seedLocked ? <Lock className="size-4" /> : <LockOpen className="size-4" />}
                </button>
              </div>
            </div>

            <div className="mx-4 h-px bg-line sm:mx-5" />

            <div className="p-4 sm:px-5 sm:pb-5">
              <div className="flex items-baseline justify-between gap-3">
                <div>
                  <div className="text-[15px] font-semibold tracking-[-0.01em]">Подсказка</div>
                  <div className="text-[13px] text-mute">Необязательно — материалы и детали, иначе берём с изображения</div>
                </div>
                <span className="shrink-0 text-[12px] tabular-nums text-faint">{materials.length}/600</span>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {HINT_CHIPS[scene].map((c) => (
                  <button
                    key={c}
                    type="button"
                    disabled={running}
                    onClick={() => addHint(c)}
                    className="rounded-full px-3 py-1 text-[12.5px] font-medium text-ink-2 ring-1 ring-fg/[0.08] transition-all duration-200 hover:bg-fg/[0.04] hover:text-ink active:scale-95 disabled:opacity-50"
                  >
                    + {c}
                  </button>
                ))}
              </div>
              <textarea
                ref={hintRef}
                value={materials}
                onChange={(e) => setMaterials(e.target.value)}
                disabled={running}
                rows={3}
                maxLength={600}
                placeholder={HINT_PLACEHOLDER[scene]}
                className="mt-3 w-full resize-none rounded-[14px] bg-fg/[0.04] px-3.5 py-2.5 text-[14px] leading-snug outline-none ring-accent/40 transition-shadow placeholder:text-faint focus:bg-surface focus:ring-2 disabled:opacity-60"
              />
            </div>
          </div>
        </div>

        {/* Control panel */}
        <aside className="lg:sticky lg:top-20 lg:self-start">
          <div className="rounded-[28px] bg-surface p-5 shadow-card ring-1 ring-fg/[0.04] sm:p-6">
            <Step n={1} title="Модель" hint="Скорость против детализации" />
            <ModelSelect value={model} onChange={setModel} disabled={running} />

            <div className="my-6 h-px bg-line" />

            <Step n={2} title="Освещение" hint="Выберите сценарий света" />
            <div className="mt-4 grid grid-cols-2 gap-2.5">
              {presetsFor(scene).map((p) => (
                <PresetCard key={`${scene}-${p.id}`} preset={p} selected={preset === p.id} onSelect={() => setPreset(p.id)} disabled={running} />
              ))}
              <PresetCard preset={CUSTOM_PRESET} selected={preset === 'custom'} onSelect={() => setPreset('custom')} disabled={running} wide />
            </div>

            <AnimatePresence initial={false}>
              {preset === 'custom' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  transition={spring} className="overflow-hidden"
                >
                  <div className="pt-3">
                    {reference ? (
                      <div className="flex items-center gap-3 rounded-2xl bg-canvas p-2 pr-3">
                        <img src={reference.url} alt="" className="size-12 rounded-xl object-cover" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-[13px] font-medium">{reference.name}</div>
                          <div className="text-[12px] text-faint">Референс света</div>
                        </div>
                        <IconButton label="Убрать референс" onClick={() => setReference(null)} plain>
                          <X className="size-4" />
                        </IconButton>
                      </div>
                    ) : (
                      <Dropzone compact onFile={async (f) => setReference(await toStudioImage(f))} title="Загрузить референс" hint="Фото с нужным настроением света" />
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="mt-3 divide-y divide-line overflow-hidden rounded-2xl bg-canvas">
              <KeepToggle
                title="Сохранить небо"
                hint={keepSky ? 'Те же облака, свет и цвет — по референсу' : 'Небо — по референсу'}
                checked={keepSky} onChange={setKeepSky} disabled={running}
              />
              <KeepToggle
                title="Сохранить фон"
                hint={keepBg ? 'Та же застройка позади, свет и цвет — по референсу' : 'Фон — по референсу'}
                checked={keepBg} onChange={setKeepBg} disabled={running}
              />
              <KeepToggle
                title="Сохранить окружение"
                hint={keepSur ? 'Те же дороги, озеленение и машины, свет — по референсу' : 'Окружение — по референсу'}
                checked={keepSur} onChange={setKeepSur} disabled={running}
              />
            </div>

            <div className="my-6 h-px bg-line" />

            <Step n={3} title="Сила эффекта" hint="Насколько сильно менять сцену" />
            <div className="mt-4">
              <div className="mb-3 flex items-baseline justify-between">
                <span className="text-[13px] text-mute">{strength < 0.4 ? 'Деликатно' : strength < 0.75 ? 'Сбалансированно' : 'Выразительно'}</span>
                <span className="text-[15px] font-semibold tabular-nums">{Math.round(strength * 100)}%</span>
              </div>
              <input
                type="range" min={0.1} max={1} step={0.01} value={strength} disabled={running}
                onChange={(e) => setStrength(Number(e.target.value))}
                className="range w-full"
                style={{ '--fill': `${((strength - 0.1) / 0.9) * 100}%` } as React.CSSProperties}
              />
            </div>

            <div className="my-6 h-px bg-line" />

            <motion.button
              whileTap={canGenerate ? { scale: 0.98 } : undefined}
              onClick={generate}
              disabled={!canGenerate}
              className="relative flex w-full items-center justify-center gap-2 overflow-hidden rounded-full bg-accent py-3.5 text-[16px] font-medium text-white transition-all duration-300 ease-apple hover:bg-accent-hover disabled:bg-fg/[0.08] disabled:text-faint"
            >
              {running ? (
                <>
                  <RefreshCw className="size-[18px] animate-spin" /> Рендерим…
                </>
              ) : result ? (
                <>
                  <RotateCcw className="size-[18px]" /> Сгенерировать заново
                </>
              ) : (
                <>
                  <Sparkles className="size-[18px]" /> Сгенерировать
                </>
              )}
            </motion.button>
            <p className="mt-3 text-center text-[12px] text-faint">
              {!source
                ? 'Сначала загрузите изображение'
                : preset === 'custom' && !reference
                  ? 'Добавьте референс для своего пресета'
                  : `${activeModel.name} · ${activeModel.time} · ${fmt(activeModel.cost)} ток.`}
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
    </div>
  )
}

const HINT_CHIPS: Record<Scene, string[]> = {
  exterior: ['Стены', 'Цоколь', 'Кровля', 'Окна', 'Озеленение'],
  interior: ['Пол', 'Стены', 'Потолок', 'Мебель', 'Текстиль'],
}

const HINT_PLACEHOLDER: Record<Scene, string> = {
  exterior: 'Стены — серая штукатурка, цоколь — бежевый рваный камень, кровля — графитовый фальц',
  interior: 'Пол — дубовая ёлочка, стены — тёплая известковая штукатурка, диван — бежевое букле',
}

function KeepToggle({ title, hint, checked, onChange, disabled }: {
  title: string
  hint: string
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <label className={`flex cursor-pointer items-center gap-3 px-3.5 py-2.5 ${disabled ? 'pointer-events-none opacity-60' : ''}`}>
      <div className="min-w-0 flex-1">
        <div className="text-[14px] font-medium">{title}</div>
        <div className="text-[12px] text-faint">{hint}</div>
      </div>
      <input type="checkbox" className="peer sr-only" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span
        aria-hidden
        className={`relative h-[26px] w-[44px] shrink-0 rounded-full transition-colors duration-300 ease-apple peer-focus-visible:ring-2 peer-focus-visible:ring-accent/40 ${checked ? 'bg-accent' : 'bg-fg/[0.12]'}`}
      >
        <span
          className={`absolute top-[3px] left-[3px] size-5 rounded-full bg-white shadow-sm transition-transform duration-300 ease-apple ${checked ? 'translate-x-[18px]' : ''}`}
        />
      </span>
    </label>
  )
}

function Step({ n, title, hint }: { n: number; title: string; hint: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-ink text-[12px] font-semibold text-on-ink">{n}</span>
      <div>
        <div className="text-[17px] font-semibold tracking-[-0.015em]">{title}</div>
        <div className="text-[13px] text-mute">{hint}</div>
      </div>
    </div>
  )
}

function IconButton({ children, label, onClick, plain }: { children: React.ReactNode; label: string; onClick: () => void; plain?: boolean }) {
  return (
    <button
      aria-label={label}
      title={label}
      onClick={onClick}
      className={`grid size-9 place-items-center rounded-full text-ink-2 transition-all duration-300 ease-apple hover:text-ink active:scale-95 ${
        plain ? 'hover:bg-fg/[0.06]' : 'glass shadow-soft ring-1 ring-fg/5 hover:bg-surface'
      }`}
    >
      {children}
    </button>
  )
}

function SecondaryButton({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <motion.button
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className="flex flex-1 items-center justify-center gap-2 rounded-full bg-fg/[0.05] px-5 py-2.5 text-[14px] font-medium text-ink transition-colors hover:bg-fg/[0.08] sm:flex-none"
    >
      {children}
    </motion.button>
  )
}
