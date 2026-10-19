import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Columns2, Download, Eye, X } from 'lucide-react'
import { BrushOptions, MODE_COLOR, ModeSwitch, Toolbar, type Shape, type Tool } from './Toolbar'
import { ZoomControls } from '../ZoomControls'
import { CompareSlider } from '../CompareSlider'
import { ActionPopover, POPOVER_WIDTH } from './ActionPopover'
import { inpaint, type InpaintMode } from '../../lib/inpaint'
import type { StudioImage } from '../../lib/types'
import { useScene } from '../../lib/scene'
import { useAccount } from '../../lib/account'
import { useZoomPan, ZOOM_MAX, ZOOM_MIN } from '../../lib/zoom'

interface Pt { x: number; y: number }
interface Box { x: number; y: number; w: number; h: number }
interface Version { url: string; label: string; mode?: InpaintMode }

const MASK_MAX = 2048
const UNDO_LIMIT = 12
const PAD_TOP = 72
const PAD_BOTTOM = 76
const BRUSH_MIN = 4
const BRUSH_MAX = 200
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v))

interface Props {
  image: StudioImage
  onClose: () => void
}

export function Editor({ image, onClose }: Props) {
  const { scene } = useScene()
  const { model, llm, charge } = useAccount()
  const stageRef = useRef<HTMLDivElement>(null)
  const surfaceRef = useRef<HTMLDivElement>(null)
  const maskRef = useRef<HTMLCanvasElement>(null)
  const cursorRef = useRef<HTMLDivElement>(null)

  const [stage, setStage] = useState({ w: 0, h: 0 })
  const [tool, setTool] = useState<Tool>('lasso')
  const [mode, setMode] = useState<InpaintMode>('remove')
  const [shape, setShape] = useState<Shape>('circle')
  const [size, setSize] = useState(48)
  const [bbox, setBbox] = useState<Box | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [lasso, setLasso] = useState<Pt[]>([])
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState<{ progress: number; stage: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [versions, setVersions] = useState<Version[]>([{ url: image.url, label: 'Исходник' }])
  const [current, setCurrent] = useState(0)
  const [peek, setPeek] = useState(false)
  // шторка «до / после»: исходник против текущей версии
  const [compareOn, setCompareOn] = useState(false)
  const [hist, setHist] = useState({ undo: 0, redo: 0 })

  const undoStack = useRef<ImageData[]>([])
  const redoStack = useRef<ImageData[]>([])
  const drawingRef = useRef(false)
  const lastPt = useRef<Pt | null>(null)
  const lassoPts = useRef<Pt[]>([])
  const abortRef = useRef<AbortController | null>(null)

  // размеры маски — в пикселях исходника (с ограничением)
  const ms = Math.min(1, MASK_MAX / Math.max(image.width, image.height))
  const maskW = Math.round(image.width * ms)
  const maskH = Math.round(image.height * ms)

  // вписываем изображение в сцену
  const padX = stage.w < 640 ? 16 : 84
  const availW = Math.max(1, stage.w - padX * 2)
  const availH = Math.max(1, stage.h - PAD_TOP - PAD_BOTTOM)
  const fit = Math.min(availW / image.width, availH / image.height)
  const baseW = image.width * fit
  const baseH = image.height * fit
  // шторка доступна, когда есть с чем сравнивать и ничего не генерируется
  const compare = compareOn && current > 0 && !busy
  // инструмент рисования выключает шторку — на ней левая кнопка двигает границу
  const chooseTool = (t: Tool) => {
    setTool(t)
    if (t !== 'hand') setCompareOn(false)
  }

  // масштаб и перемещение: изображение растёт от вписанного размера, сдвиг — от центра рабочей области
  const { view, zoomIn, zoomOut, zoomFit, panning, panTool, spaceHeld, panHandlers } = useZoomPan(
    stageRef,
    { baseW, baseH, availW, availH, cx: stage.w / 2, cy: PAD_TOP + availH / 2 },
    { panWithLeft: tool === 'hand' && !compareOn },
  )
  const dispW = baseW * view.zoom
  const dispH = baseH * view.zoom
  const left = (stage.w - dispW) / 2 + view.x
  const top = PAD_TOP + (availH - dispH) / 2 + view.y
  const k = maskW / Math.max(1, dispW) // экранные px → px маски (кисть — постоянного экранного размера)

  useLayoutEffect(() => {
    const el = stageRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setStage({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => () => abortRef.current?.abort(), [])

  /* ---------------- mask helpers ---------------- */

  const ctx = () => maskRef.current!.getContext('2d', { willReadFrequently: true })!

  const recolor = useCallback((m: InpaintMode) => {
    const c = ctx()
    c.globalCompositeOperation = 'source-in'
    c.fillStyle = MODE_COLOR[m]
    c.fillRect(0, 0, maskW, maskH)
    c.globalCompositeOperation = 'source-over'
  }, [maskW, maskH])

  useEffect(() => recolor(mode), [mode, recolor])
  // подсказка удаления и описание правки — разные вещи, при смене режима поле очищается
  useEffect(() => setPrompt(''), [mode])

  const computeBox = useCallback((): Box | null => {
    const d = ctx().getImageData(0, 0, maskW, maskH).data
    let minX = maskW, minY = maskH, maxX = -1, maxY = -1
    for (let y = 0; y < maskH; y += 2) {
      const row = y * maskW * 4
      for (let x = 0; x < maskW; x += 2) {
        if (d[row + x * 4 + 3] > 8) {
          if (x < minX) minX = x
          if (x > maxX) maxX = x
          if (y < minY) minY = y
          if (y > maxY) maxY = y
        }
      }
    }
    return maxX < 0 ? null : { x: minX, y: minY, w: maxX - minX + 2, h: maxY - minY + 2 }
  }, [maskW, maskH])

  const syncHist = () => setHist({ undo: undoStack.current.length, redo: redoStack.current.length })
  const snapshot = () => ctx().getImageData(0, 0, maskW, maskH)

  const pushUndo = () => {
    undoStack.current.push(snapshot())
    if (undoStack.current.length > UNDO_LIMIT) undoStack.current.shift()
    redoStack.current = []
    syncHist()
  }

  const restore = (from: ImageData[], to: ImageData[]) => {
    const s = from.pop()
    if (!s) return
    to.push(snapshot())
    ctx().putImageData(s, 0, 0)
    recolor(mode)
    setBbox(computeBox())
    syncHist()
  }
  const undo = () => restore(undoStack.current, redoStack.current)
  const redo = () => restore(redoStack.current, undoStack.current)

  const clearMask = (record = true) => {
    if (record && bbox) pushUndo()
    ctx().clearRect(0, 0, maskW, maskH)
    setBbox(null)
  }

  /* ---------------- drawing ---------------- */

  const toMask = (e: React.PointerEvent): Pt => {
    const r = surfaceRef.current!.getBoundingClientRect()
    return { x: ((e.clientX - r.left) / r.width) * maskW, y: ((e.clientY - r.top) / r.height) * maskH }
  }

  const stampSegment = (a: Pt, b: Pt) => {
    const c = ctx()
    const r = (size / 2) * k
    c.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over'
    c.fillStyle = MODE_COLOR[mode]
    const n = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / Math.max(1, r / 4)))
    for (let i = 0; i <= n; i++) {
      const x = a.x + ((b.x - a.x) * i) / n
      const y = a.y + ((b.y - a.y) * i) / n
      if (shape === 'circle') {
        c.beginPath()
        c.arc(x, y, r, 0, Math.PI * 2)
        c.fill()
      } else {
        c.fillRect(x - r, y - r, r * 2, r * 2)
      }
    }
    c.globalCompositeOperation = 'source-over'
  }

  const moveCursor = (e: React.PointerEvent) => {
    const el = cursorRef.current
    const r = surfaceRef.current?.getBoundingClientRect()
    if (!el || !r) return
    el.style.transform = `translate(${e.clientX - r.left - size / 2}px, ${e.clientY - r.top - size / 2}px)`
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (busy || e.button !== 0 || panTool) return
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* синтетические события */ }
    pushUndo()
    drawingRef.current = true
    setDrawing(true)
    setError(null)
    const p = toMask(e)
    if (tool === 'lasso') {
      lassoPts.current = [p]
      setLasso([p])
    } else {
      lastPt.current = p
      stampSegment(p, p)
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    moveCursor(e)
    if (!drawingRef.current) return
    const p = toMask(e)
    if (tool === 'lasso') {
      const last = lassoPts.current[lassoPts.current.length - 1]
      if (Math.hypot(p.x - last.x, p.y - last.y) > 2 * k) {
        lassoPts.current.push(p)
        setLasso([...lassoPts.current])
      }
    } else if (lastPt.current) {
      stampSegment(lastPt.current, p)
      lastPt.current = p
    }
  }

  const onPointerUp = () => {
    if (!drawingRef.current) return
    drawingRef.current = false
    setDrawing(false)
    if (tool === 'lasso') {
      const pts = lassoPts.current
      if (pts.length > 2) {
        const c = ctx()
        c.fillStyle = MODE_COLOR[mode]
        c.beginPath()
        pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)))
        c.closePath()
        c.fill()
      } else {
        undoStack.current.pop()
        syncHist()
      }
      lassoPts.current = []
      setLasso([])
    }
    lastPt.current = null
    setBbox(computeBox())
  }

  /* ---------------- generation ---------------- */

  async function apply() {
    if (!bbox || busy) return
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setBusy({ progress: 0, stage: '' })
    setError(null)
    try {
      const text = prompt.trim()
      const url = await inpaint(
        { imageUrl: versions[current].url, scene, model, llm, mask: maskRef.current!, mode, prompt: text || undefined },
        (progress, stage) => setBusy({ progress, stage }),
        ctrl.signal,
      )
      const label = mode === 'remove' ? (text ? `Удаление · ${text}` : 'Удаление') : text
      charge('inpaint', `Inpainting · ${label.toLowerCase()}`)
      setVersions((v) => [...v.slice(0, current + 1), { url, mode, label }])
      setCurrent((c) => c + 1)
      clearMask(false)
      undoStack.current = []
      redoStack.current = []
      syncHist()
      setPrompt('')
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  function download() {
    const a = document.createElement('a')
    a.href = versions[current].url
    a.download = `inego_inpaint_v${current}_${image.name.replace(/\.[^.]+$/, '')}.jpg`
    a.click()
  }

  /* ---------------- shortcuts ---------------- */

  const handlers = useRef({ undo, redo, clearMask, setSize, chooseTool })
  handlers.current = { undo, redo, clearMask, setSize, chooseTool }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT') return
      const h = handlers.current
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.code === 'KeyZ') { e.preventDefault(); e.shiftKey ? h.redo() : h.undo(); return }
      if (mod && e.code === 'KeyY') { e.preventDefault(); h.redo(); return }
      if (mod) return
      if (e.code === 'KeyL') h.chooseTool('lasso')
      else if (e.code === 'KeyB') h.chooseTool('brush')
      else if (e.code === 'KeyE') h.chooseTool('eraser')
      else if (e.code === 'KeyH') h.chooseTool('hand')
      else if (e.code === 'BracketLeft') h.setSize((s) => Math.max(BRUSH_MIN, s - 6))
      else if (e.code === 'BracketRight') h.setSize((s) => Math.min(BRUSH_MAX, s + 6))
      else if (e.key === 'Escape') h.clearMask()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  /* ---------------- popover placement ---------------- */

  const showPopover = !!bbox && !drawing && !busy && !peek && !panning && !compare
  let pop = { x: 0, y: 0, placement: 'bottom' as 'top' | 'bottom' }
  if (bbox) {
    const bx = left + bbox.x / k
    const by = top + bbox.y / k
    const bw = bbox.w / k
    const bh = bbox.h / k
    const est = mode === 'edit' ? 250 : 230
    let placement: 'top' | 'bottom' = 'bottom'
    let y = by + bh + 14
    if (y + est > stage.h - 8) {
      placement = 'top'
      y = by - 14
      if (y - est < 8) y = stage.h - 16
    }
    // при приближении выделение может уйти за край — окно остаётся в сцене
    y = placement === 'top' ? clamp(y, est + 8, stage.h - 16) : clamp(y, 8, stage.h - est - 8)
    // не наезжаем на панели инструментов и масштаба по бокам
    const minX = (stage.w < 640 ? 12 : 76) + POPOVER_WIDTH / 2
    const maxX = Math.max(minX, stage.w - POPOVER_WIDTH / 2 - (stage.w < 640 ? 12 : 76))
    pop = { x: Math.min(Math.max(bx + bw / 2, minX), maxX), y, placement }
  }

  const shown = peek && current > 0 ? versions[current - 1] : versions[current]
  const brushLike = (tool === 'brush' || tool === 'eraser') && !spaceHeld && !compare
  const color = MODE_COLOR[mode]

  return (
    <div className="flex flex-col gap-4">
      <div
        ref={stageRef}
        className={`checker relative h-[min(78vh,760px)] min-h-[480px] touch-none select-none overflow-hidden rounded-[28px] shadow-card ring-1 ring-fg/[0.04] ${
          panning ? 'cursor-grabbing' : panTool ? 'cursor-grab' : ''
        }`}
        {...panHandlers}
      >
        {/* изображение + маска */}
        <div className="absolute" style={{ left, top, width: dispW, height: dispH }}>
          <AnimatePresence initial={false}>
            <motion.img
              key={shown.url}
              src={shown.url}
              alt={shown.label}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35 }}
              draggable={false}
              className="absolute inset-0 size-full rounded-[4px] object-contain shadow-[0_8px_40px_-12px_rgba(0,0,0,0.35)]"
            />
          </AnimatePresence>

          <div
            className={`absolute inset-0 transition-opacity duration-300 ${peek || compare ? 'opacity-0' : ''}`}
            style={{ filter: drawing ? undefined : `drop-shadow(0 0 1.5px rgba(255,255,255,0.9)) drop-shadow(0 0 10px ${color})` }}
          >
            <canvas
              ref={maskRef}
              width={maskW}
              height={maskH}
              className={`absolute inset-0 size-full ${busy || (showPopover && mode === 'remove') ? 'animate-breathe' : 'opacity-50'}`}
            />
          </div>

          {/* превью лассо */}
          {lasso.length > 1 && (
            <svg viewBox={`0 0 ${maskW} ${maskH}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 size-full">
              <path d={`M${lasso.map((p) => `${p.x},${p.y}`).join('L')}Z`} fill={color} fillOpacity={0.22} stroke="none" />
              <path d={`M${lasso.map((p) => `${p.x},${p.y}`).join('L')}`} fill="none" stroke="#000" strokeOpacity={0.35} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              <path
                d={`M${lasso.map((p) => `${p.x},${p.y}`).join('L')}`}
                fill="none" stroke="#fff" strokeWidth={1.5} strokeDasharray="6 6" vectorEffect="non-scaling-stroke"
                className="animate-ants"
              />
            </svg>
          )}

          {/* мерцание во время генерации */}
          {busy && (
            <div className="pointer-events-none absolute inset-0 overflow-hidden">
              <div className="animate-shimmer absolute inset-y-0 w-1/2 bg-gradient-to-r from-transparent via-white/40 to-transparent" />
            </div>
          )}

          {/* поверхность для рисования */}
          <div
            ref={surfaceRef}
            className={`absolute inset-0 touch-none ${
              panning ? 'cursor-grabbing' : panTool ? 'cursor-grab' : busy ? 'cursor-wait' : brushLike ? 'cursor-none' : 'cursor-crosshair'
            }`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onPointerEnter={() => cursorRef.current && (cursorRef.current.style.opacity = '1')}
            onPointerLeave={() => cursorRef.current && (cursorRef.current.style.opacity = '0')}
          >
            {brushLike && !busy && (
              <div
                ref={cursorRef}
                className={`pointer-events-none absolute left-0 top-0 opacity-0 transition-[width,height,border-radius] duration-150 ${shape === 'circle' ? 'rounded-full' : 'rounded-[3px]'}`}
                style={{
                  width: size,
                  height: size,
                  border: `1.5px ${tool === 'eraser' ? 'dashed' : 'solid'} #fff`,
                  boxShadow: '0 0 0 1px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(0,0,0,0.15)',
                  background: tool === 'eraser' ? 'rgba(255,255,255,0.12)' : `${color}22`,
                }}
              />
            )}
          </div>
        </div>

        {compare && (
          <div className="absolute inset-0">
            <CompareSlider
              before={versions[0].url}
              after={versions[current].url}
              beforeLabel={versions[0].label}
              afterLabel={versions[current].label}
              labelsAt="bottom"
              // картинки — ровно в рамке изображения сцены (с масштабом и сдвигом), граница — в координатах экрана
              imageStyle={{ inset: 'auto', left, top, width: dispW, height: dispH }}
            />
          </div>
        )}

        {/* верх: режим / прогресс */}
        <div className="absolute inset-x-0 top-4 flex flex-col items-center gap-2 px-16">
          <AnimatePresence mode="wait" initial={false}>
            {busy ? (
              <motion.div
                key="busy"
                initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                className="glass flex items-center gap-3 rounded-full py-1.5 pl-2 pr-1.5 shadow-float ring-1 ring-fg/5"
              >
                <ProgressRing value={busy.progress} />
                <span className="min-w-[150px] text-[13px] font-medium">{busy.stage || 'Подготовка'}…</span>
                <span className="w-9 text-right text-[13px] font-semibold tabular-nums">{Math.round(busy.progress * 100)}%</span>
                <button
                  onClick={() => abortRef.current?.abort()}
                  aria-label="Отменить"
                  className="grid size-7 place-items-center rounded-full text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink"
                >
                  <X className="size-3.5" />
                </button>
              </motion.div>
            ) : (
              <motion.div key="mode" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                <ModeSwitch mode={mode} onMode={setMode} />
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {!bbox && !busy && !drawing && !compare && (
              <motion.div
                initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="glass rounded-full px-3.5 py-1.5 text-[12.5px] text-ink-2 shadow-soft ring-1 ring-fg/5"
              >
                {tool === 'hand'
                  ? 'Перетаскивайте изображение · колесо — масштаб'
                  : `${tool === 'lasso' ? 'Обведите' : 'Закрасьте'} ${mode === 'edit' ? 'область, которую нужно изменить' : 'объект, который нужно удалить'}`}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* слева: инструменты */}
        <div className="absolute left-4 top-1/2 z-20 -translate-y-1/2">
          <Toolbar
            tool={tool}
            onTool={chooseTool}
            canUndo={hist.undo > 0}
            canRedo={hist.redo > 0}
            hasMask={!!bbox}
            onUndo={undo}
            onRedo={redo}
            onClear={() => clearMask()}
            disabled={!!busy}
          />
        </div>

        {/* справа: масштаб */}
        <div className="absolute right-4 top-1/2 z-20 -translate-y-1/2">
          <ZoomControls zoom={view.zoom} min={ZOOM_MIN} max={ZOOM_MAX} onZoomIn={zoomIn} onZoomOut={zoomOut} onFit={zoomFit} />
        </div>

        {/* справа сверху: закрыть */}
        <button
          onClick={onClose}
          aria-label="Закрыть изображение"
          title="Закрыть изображение"
          className="glass absolute right-4 top-4 grid size-9 place-items-center rounded-full text-ink-2 shadow-soft ring-1 ring-fg/5 transition hover:bg-surface hover:text-ink"
        >
          <X className="size-4" />
        </button>

        {/* низ: параметры кисти */}
        <AnimatePresence>
          {brushLike && !busy && (
            <motion.div
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
              transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              className="absolute inset-x-0 bottom-4 flex justify-center"
            >
              <BrushOptions shape={shape} onShape={setShape} size={size} onSize={setSize} min={BRUSH_MIN} max={BRUSH_MAX} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* окно действия */}
        <AnimatePresence>
          {showPopover && (
            <ActionPopover
              key={mode}
              mode={mode}
              prompt={prompt}
              onPrompt={setPrompt}
              onConfirm={apply}
              onReset={() => clearMask()}
              x={pop.x}
              y={pop.y}
              placement={pop.placement}
            />
          )}
        </AnimatePresence>
      </div>

      {/* история версий */}
      <div className="flex flex-col gap-3 rounded-[22px] bg-surface p-3 shadow-card ring-1 ring-fg/[0.04] sm:flex-row sm:items-center sm:pl-4">
        <div className="text-[13px] font-semibold text-ink-2 sm:w-16">История</div>
        {/* отступ под рамку выбранной версии (ring + offset = 4px): иначе прокрутка её обрезает */}
        <div className="-m-1 flex min-w-0 flex-1 gap-2 overflow-x-auto p-1 pb-1.5">
          {versions.map((v, i) => (
            <button
              key={v.url}
              onClick={() => !busy && setCurrent(i)}
              title={v.label}
              className="group flex w-[76px] shrink-0 flex-col gap-1 text-left"
            >
              <span
                className={`relative block h-12 w-full overflow-hidden rounded-xl ring-offset-2 transition-all duration-200 ${
                  i === current ? 'ring-2 ring-ink' : 'ring-1 ring-fg/10 group-hover:ring-fg/25'
                }`}
              >
                <img src={v.url} alt="" className="size-full object-cover" />
                {v.mode && (
                  <span className="absolute right-1 top-1 size-2 rounded-full ring-2 ring-white" style={{ background: MODE_COLOR[v.mode] }} />
                )}
              </span>
              <span className={`truncate text-[11px] ${i === current ? 'font-medium text-ink' : 'text-faint'}`}>{v.label}</span>
            </button>
          ))}
        </div>
        {error && <span className="rounded-full bg-danger/10 px-3 py-1.5 text-[12px] text-danger">{error}</span>}
        <div className="flex gap-2">
          <button
            disabled={current === 0}
            onPointerDown={() => setPeek(true)}
            onPointerUp={() => setPeek(false)}
            onPointerLeave={() => setPeek(false)}
            className="flex flex-1 select-none items-center justify-center gap-2 rounded-full bg-fg/[0.05] px-4 py-2.5 text-[13.5px] font-medium transition-colors hover:bg-fg/[0.08] active:bg-fg/[0.12] disabled:opacity-40 sm:flex-none"
            title="Удерживайте, чтобы увидеть предыдущую версию"
          >
            <Eye className="size-4" /> {peek ? 'Предыдущая' : 'Удерживайте'}
          </button>
          <button
            disabled={current === 0 || !!busy}
            onClick={() => setCompareOn((v) => !v)}
            aria-pressed={compare}
            className={`flex flex-1 select-none items-center justify-center gap-2 rounded-full px-4 py-2.5 text-[13.5px] font-medium transition-colors disabled:opacity-40 sm:flex-none ${
              compare ? 'bg-ink text-on-ink hover:bg-ink-hover' : 'bg-fg/[0.05] hover:bg-fg/[0.08] active:bg-fg/[0.12]'
            }`}
            title="Шторка: исходник и текущая версия"
          >
            <Columns2 className="size-4" /> До / после
          </button>
          <button
            onClick={download}
            className="flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-4 py-2.5 text-[13.5px] font-medium text-on-ink transition-colors hover:bg-ink-hover sm:flex-none"
          >
            <Download className="size-4" /> Скачать
          </button>
        </div>
      </div>
    </div>
  )
}

function ProgressRing({ value }: { value: number }) {
  const r = 10
  const c = 2 * Math.PI * r
  return (
    <svg viewBox="0 0 26 26" className="size-7 -rotate-90">
      <circle cx="13" cy="13" r={r} fill="none" stroke="var(--color-fg)" strokeOpacity={0.08} strokeWidth="3" />
      <circle cx="13" cy="13" r={r} fill="none" stroke="url(#ip-ring)" strokeWidth="3" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - value)} />
      <defs>
        <linearGradient id="ip-ring" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ff8a3d" />
          <stop offset=".5" stopColor="#ff4f7b" />
          <stop offset="1" stopColor="#6a5cff" />
        </linearGradient>
      </defs>
    </svg>
  )
}
