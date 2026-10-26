import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Brush, Check, Columns2, Download, Eye, EyeOff, History, Pentagon, SquareDashed, X } from 'lucide-react'
import { BrushOptions, MODE_COLOR, ModeSwitch, Toolbar, type Shape, type Tool } from './Toolbar'
import { ZoomControls } from '../ZoomControls'
import { CompareSlider } from '../CompareSlider'
import { ActionPopover, POPOVER_WIDTH, TEXTURE_MAX, isTextureFile } from './ActionPopover'
import { inpaint, planInpaint, type InpaintMode } from '../../lib/inpaint'
import { buildMaterials, DIVIDER_ID, pickMask, type MaterialMap } from '../../lib/materials'
import { PlanBox, type Box as PlanRect } from './PlanBox'
import type { StudioImage } from '../../lib/types'
import { loadImage } from '../../lib/grade'
import { useScene } from '../../lib/scene'
import { useAccount } from '../../lib/account'
import { useZoomPan, ZOOM_MAX, ZOOM_MIN } from '../../lib/zoom'
import { fillPolygon, HARDNESS_DEFAULT, lockAxis, rectPoints, type DrawTool } from '../../lib/brush'
import { changedRegion, colorize, combineRegion, NEUTRAL, newEdits, REGION_HARDNESS, type ColorValues, type RegionEdits } from '../../lib/colorAdjust'
import { RegionBrush, type RegionBrushHandle, type RegionMode } from '../RegionBrush'
import { ColorPanel } from '../ColorPanel'

interface Pt { x: number; y: number }
interface Box { x: number; y: number; w: number; h: number }
/** откат кистью: из чего собран и с каким краем — край можно поменять и после «Применить» */
interface RestoreEdit { cur: string; prev: string; mask: HTMLCanvasElement; hardness: number }
/** цвет сгенерированного: версия «Цвет» — base с поправленным цветом там, где base отличается от prev */
interface ColorEdit { base: string; prev: string; values: ColorValues; hardness?: number; offset?: number }
interface Version { url: string; label: string; mode?: InpaintMode; restore?: RestoreEdit; color?: ColorEdit }
/** выбранное инструментом «Выбор маски»: элементы и их маска (в размере карты), нарисованная на канвасе маски */
interface Pick { ids: Set<number>; mask: Uint8Array | null }

const MASK_MAX = 2048
const UNDO_LIMIT = 12
const PAD_TOP = 72
const PAD_BOTTOM = 76
const BRUSH_MIN = 4
const BRUSH_MAX = 200
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v))
// кисть отката: мазки с жёстким краем, мягкость — размытием всей маски на эту долю длинной стороны кадра при нулевой
// жёсткости (ползунок «Край»). Не зависит ни от размера кисти, ни от масштаба — и меняется после «Применить»
const RESTORE_FEATHER = 0.012

/** Откат в полном разрешении: текущая версия + предыдущая под маской с мягким краем → object URL. */
async function composeRestore(e: RestoreEdit): Promise<string> {
  const [cur, prev] = await Promise.all([loadImage(e.cur), loadImage(e.prev)])
  const W = cur.naturalWidth
  const H = cur.naturalHeight
  const out = Object.assign(document.createElement('canvas'), { width: W, height: H })
  const o = out.getContext('2d')!
  o.drawImage(cur, 0, 0)
  const layer = Object.assign(document.createElement('canvas'), { width: W, height: H })
  const l = layer.getContext('2d')!
  l.drawImage(prev, 0, 0, W, H)
  l.globalCompositeOperation = 'destination-in'
  const blur = (1 - e.hardness) * RESTORE_FEATHER * Math.max(W, H)
  if (blur > 0.5) l.filter = `blur(${blur}px)`
  l.drawImage(e.mask, 0, 0, W, H)
  o.drawImage(layer, 0, 0)
  const blob = await new Promise<Blob>((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/jpeg', 0.95))
  return URL.createObjectURL(blob)
}
// многоугольник замыкается кликом не дальше стольких экранных px от первой точки
const POLY_CLOSE = 10

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
  // кисть отката: мазки (альфа, в размере маски) и превью — предыдущая версия, проступающая под мазками
  const restoreRef = useRef<HTMLCanvasElement | null>(null)
  const previewRef = useRef<HTMLCanvasElement>(null)
  const prevImgRef = useRef<HTMLImageElement | null>(null)
  const restoreUndo = useRef<ImageData[]>([])
  const restoreRedo = useRef<ImageData[]>([])
  const previewFrame = useRef(0)
  const cursorRef = useRef<HTMLDivElement>(null)

  const [stage, setStage] = useState({ w: 0, h: 0 })
  const [tool, setTool] = useState<Tool>('lasso')
  const [mode, setMode] = useState<InpaintMode>('remove')
  const [shape, setShape] = useState<Shape>('circle')
  const [size, setSize] = useState(48)
  const [hardness, setHardness] = useState(HARDNESS_DEFAULT)
  // многоугольник: поставленные точки и «резинка» до курсора
  const [poly, setPoly] = useState<Pt[]>([])
  const [rubber, setRubber] = useState<Pt | null>(null)
  const [bbox, setBbox] = useState<Box | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [lasso, setLasso] = useState<Pt[]>([])
  const [prompt, setPrompt] = useState('')
  // образец материала: остаётся после применения — один образец кладут на несколько поверхностей подряд
  const [texture, setTexture] = useState<File | null>(null)
  const [textureWidth, setTextureWidth] = useState('')
  const [busy, setBusy] = useState<{ progress: number; stage: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [versions, setVersions] = useState<Version[]>([{ url: image.url, label: 'Исходник' }])
  const [current, setCurrent] = useState(0)
  const [peek, setPeek] = useState(false)
  // план правки: рамка объекта от бэкенда, которую можно поправить перед генерацией
  const [plan, setPlan] = useState<PlanRect | null>(null)
  // шторка «до / после»: исходник против текущей версии
  const [compareOn, setCompareOn] = useState(false)
  const [hist, setHist] = useState({ undo: 0, redo: 0 })
  const [restoreDirty, setRestoreDirty] = useState(false)
  // маски по схеме (как Render Pass) и версия, по которой они построены
  const [schema, setSchema] = useState<{ url: string; map: MaterialMap } | null>(null)
  const [buildingMasks, setBuildingMasks] = useState(false)
  const [hoverId, setHoverId] = useState(0)
  const hoverRef = useRef<HTMLCanvasElement>(null)
  const pick = useRef<Pick>({ ids: new Set(), mask: null })
  const pickUndo = useRef<Pick[]>([])
  const pickRedo = useRef<Pick[]>([])
  // при выборе маски окно действия не всплывает на каждый клик (закрывало элементы позади) — открывается «Далее»
  const [pickCount, setPickCount] = useState(0)
  const [pickOpen, setPickOpen] = useState(false)

  const undoStack = useRef<ImageData[]>([])
  const redoStack = useRef<ImageData[]>([])
  const drawingRef = useRef(false)
  const lastPt = useRef<Pt | null>(null)
  const lassoPts = useRef<Pt[]>([])
  // прямые: конец прошлого мазка (Shift+клик — прямая от него), начало мазка и ось (Shift в мазке)
  const lastEnd = useRef<Pt | null>(null)
  const anchor = useRef<Pt | null>(null)
  const axis = useRef<'x' | 'y' | null>(null)
  const rectStart = useRef<Pt | null>(null)
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
  // правка меняет кадр локально — маски прежней версии годятся и дальше; перестроить можно той же кнопкой
  const canPick = !!schema
  // инструмент рисования выключает шторку — на ней левая кнопка двигает границу
  const chooseTool = (t: Tool) => {
    if (t === 'restore' && current === 0) return
    if (t === 'pick' && !canPick) return
    if (t !== tool) { setPoly([]); setRubber(null); lastEnd.current = null }
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
  useEffect(() => { setPrompt(''); setPlan(null) }, [mode])

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
    pickUndo.current.push(pick.current)
    if (undoStack.current.length > UNDO_LIMIT) { undoStack.current.shift(); pickUndo.current.shift() }
    redoStack.current = []
    pickRedo.current = []
    syncHist()
  }

  const restore = (from: ImageData[], to: ImageData[]) => {
    const s = from.pop()
    if (!s) return
    to.push(snapshot())
    // выбор маски — тем же шагом, что и сам канвас
    const [pf, pt] = from === undoStack.current ? [pickUndo.current, pickRedo.current] : [pickRedo.current, pickUndo.current]
    pt.push(pick.current)
    pick.current = pf.pop() ?? { ids: new Set(), mask: null }
    setPickCount(pick.current.ids.size)
    ctx().putImageData(s, 0, 0)
    recolor(mode)
    setBbox(computeBox())
    syncHist()
  }
  const restoreStep = (from: ImageData[], to: ImageData[]) => {
    const st = from.pop()
    if (!st) return
    const c = restoreCtx()
    to.push(c.getImageData(0, 0, maskW, maskH))
    c.putImageData(st, 0, 0)
    syncRestore()
  }
  const undo = () => (tool === 'restore' ? restoreStep(restoreUndo.current, restoreRedo.current) : restore(undoStack.current, redoStack.current))
  const redo = () => (tool === 'restore' ? restoreStep(restoreRedo.current, restoreUndo.current) : restore(redoStack.current, undoStack.current))

  const clearMask = (record = true) => {
    if (record && bbox) pushUndo()
    ctx().clearRect(0, 0, maskW, maskH)
    pick.current = { ids: new Set(), mask: null }
    setPickCount(0)
    setPickOpen(false)
    setBbox(null)
    setPlan(null)
  }

  /* ---------------- маски по схеме ---------------- */

  // кадр текущей версии → /api/materials (тот же Render Pass): схема 2511 и маски по ней, ~минута
  async function buildMasks() {
    if (busy) return
    const url = versions[current].url
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setError(null)
    setBuildingMasks(true)
    const start = performance.now()
    setBusy({ progress: 0, stage: 'Рисуем схему' })
    const timer = setInterval(() => {
      const p = 0.95 * (1 - Math.exp(-2.2 * ((performance.now() - start) / 100000)))
      setBusy({ progress: p, stage: p < 0.85 ? 'Рисуем схему' : 'Режем маски' })
    }, 300)
    try {
      const blob = await fetch(url).then((r) => r.blob())
      const map = await buildMaterials(new File([blob], 'frame.png', { type: blob.type || 'image/png' }), ctrl.signal)
      pick.current = { ids: new Set(), mask: null }
      setSchema({ url, map })
      setTool('pick')
      setCompareOn(false)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      clearInterval(timer)
      setBusy(null)
      setBuildingMasks(false)
    }
  }

  // элемент схемы под точкой маски; на линиях-разделителях — ничего
  const pickAt = (p: Pt) => {
    if (!schema) return 0
    const { map } = schema
    const x = Math.floor((p.x / maskW) * map.width)
    const y = Math.floor((p.y / maskH) * map.height)
    if (x < 0 || y < 0 || x >= map.width || y >= map.height) return 0
    const id = map.ids[y * map.width + x]
    return id === DIVIDER_ID ? 0 : id
  }

  // клик: элемент добавляется в маску или убирается; на канвас ложится разница со старым выбором
  const togglePick = (id: number) => {
    if (!schema) return
    const { map } = schema
    const ids = new Set(pick.current.ids)
    if (ids.has(id)) ids.delete(id)
    else ids.add(id)
    const next = pickMask(map, ids)
    const prev = pick.current.mask
    const layer = (keep: (i: number) => boolean) => {
      const c = document.createElement('canvas')
      c.width = map.width
      c.height = map.height
      const lc = c.getContext('2d')!
      const img = lc.createImageData(map.width, map.height)
      for (let i = 0; i < next.length; i++) if (keep(i)) img.data[i * 4 + 3] = 255
      lc.putImageData(img, 0, 0)
      return c
    }
    pushUndo()
    const c = ctx()
    c.imageSmoothingEnabled = false
    if (prev) {
      c.globalCompositeOperation = 'destination-out'
      c.drawImage(layer((i) => prev[i] === 1 && next[i] === 0), 0, 0, maskW, maskH)
    }
    c.globalCompositeOperation = 'source-over'
    c.drawImage(layer((i) => next[i] === 1), 0, 0, maskW, maskH)
    recolor(mode)
    pick.current = { ids, mask: next }
    setPickCount(ids.size)
    setPickOpen(false)
    setBbox(computeBox())
  }

  // подсветка элемента под курсором — цветом режима, без обводки
  useEffect(() => {
    const cv = hoverRef.current
    if (!cv || !schema) return
    const { map } = schema
    if (cv.width !== map.width || cv.height !== map.height) {
      cv.width = map.width
      cv.height = map.height
    }
    const hc = cv.getContext('2d')!
    if (!hoverId || tool !== 'pick') { hc.clearRect(0, 0, map.width, map.height); return }
    const img = hc.createImageData(map.width, map.height)
    const [r, g, b] = [1, 3, 5].map((o) => parseInt(MODE_COLOR[mode].slice(o, o + 2), 16))
    for (let i = 0; i < map.ids.length; i++) if (map.ids[i] === hoverId) img.data.set([r, g, b, 110], i * 4)
    hc.putImageData(img, 0, 0)
  }, [schema, hoverId, tool, mode])



  /* ---------------- restore brush ---------------- */

  const restoreCtx = () => {
    if (!restoreRef.current) {
      const c = document.createElement('canvas')
      c.width = maskW
      c.height = maskH
      restoreRef.current = c
    }
    return restoreRef.current.getContext('2d', { willReadFrequently: true })!
  }
  const restoreFeather = () => (1 - hardness) * RESTORE_FEATHER * Math.max(maskW, maskH)

  // превью: предыдущая версия, видимая только под мазками (с мягким краем)
  const drawPreview = () => {
    cancelAnimationFrame(previewFrame.current)
    previewFrame.current = requestAnimationFrame(() => {
      const pv = previewRef.current
      const img = prevImgRef.current
      if (!pv || !restoreRef.current) return
      const c = pv.getContext('2d')!
      c.clearRect(0, 0, maskW, maskH)
      if (!img) return
      c.globalCompositeOperation = 'source-over'
      c.filter = 'none'
      // «Показать прошлый»: под холстом уже прошлая версия целиком — закрашенное только подсвечивается
      if (restorePeekRef.current) {
        c.globalAlpha = 0.4
        c.fillStyle = '#f59e0b'
        c.fillRect(0, 0, maskW, maskH)
        c.globalAlpha = 1
      } else {
        c.drawImage(img, 0, 0, maskW, maskH)
      }
      c.globalCompositeOperation = 'destination-in'
      c.filter = `blur(${restoreFeather()}px)`
      c.drawImage(restoreRef.current, 0, 0)
      c.filter = 'none'
      c.globalCompositeOperation = 'source-over'
    })
  }

  // «Показать прошлый» у кисти отката: прошлая версия целиком, закрашенное подсвечено — вести границы по ней
  const [restorePeek, setRestorePeek] = useState(false)
  // кисть отката рисует кистью, прямоугольником или многоугольником — прямые границы по прошлой версии
  const [restoreShape, setRestoreShape] = useState<DrawTool>('brush')
  useEffect(() => { setPoly([]); setRubber(null) }, [restoreShape])
  const restorePeekRef = useRef(restorePeek)
  restorePeekRef.current = restorePeek
  const peekPrev = restorePeek && tool === 'restore' && current > 0
  useEffect(() => { drawPreview() }, [restorePeek]) // eslint-disable-line react-hooks/exhaustive-deps

  const restoreHasStrokes = () => {
    const d = restoreCtx().getImageData(0, 0, maskW, maskH).data
    for (let i = 3; i < d.length; i += 16) if (d[i] > 8) return true
    return false
  }

  const syncRestore = () => {
    setRestoreDirty(restoreHasStrokes())
    setHist({ undo: restoreUndo.current.length, redo: restoreRedo.current.length })
    drawPreview()
  }

  const clearRestore = (record = true) => {
    const c = restoreCtx()
    if (record && restoreDirty) {
      restoreUndo.current.push(c.getImageData(0, 0, maskW, maskH))
      restoreRedo.current = []
    }
    c.clearRect(0, 0, maskW, maskH)
    syncRestore()
  }

  // предыдущая версия для кисти — при смене текущей; мазки к другой паре версий не относятся
  useEffect(() => {
    prevImgRef.current = null
    restoreUndo.current = []
    restoreRedo.current = []
    if (restoreRef.current) restoreCtx().clearRect(0, 0, maskW, maskH)
    setRestoreDirty(false)
    if (current === 0) {
      if (tool === 'restore') setTool('brush')
      drawPreview()
      return
    }
    let alive = true
    loadImage(versions[current - 1].url).then((img) => {
      if (!alive) return
      prevImgRef.current = img
      drawPreview()
    })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, versions])

  // мягкость края отката — сразу в превью
  useEffect(() => { drawPreview() }, [hardness]) // eslint-disable-line react-hooks/exhaustive-deps

  // стек отмены и кнопки панели — свои у кисти отката
  useEffect(() => {
    if (tool === 'restore') setHist({ undo: restoreUndo.current.length, redo: restoreRedo.current.length })
    else syncHist()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool])

  // откат: текущая версия + предыдущая под мазками, в полном разрешении — новая версия в истории
  async function applyRestore() {
    if (!restoreDirty || busy || current === 0) return
    setBusy({ progress: 0.5, stage: 'Откат кистью' })
    try {
      const mask = Object.assign(document.createElement('canvas'), { width: maskW, height: maskH })
      mask.getContext('2d')!.drawImage(restoreRef.current!, 0, 0)
      const edit = { cur: versions[current].url, prev: versions[current - 1].url, mask, hardness }
      const url = await composeRestore(edit)
      setVersions((v) => [...v.slice(0, current + 1), { url, label: 'Откат кистью', restore: edit }])
      setCurrent((c) => c + 1)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  // край у применённого отката: та же версия пересобирается с другой мягкостью (ползунок отпущен — без задержки)
  const [edgeDraft, setEdgeDraft] = useState<number | null>(null)
  const edgeTimer = useRef(0)
  const editedRestore = versions[current]?.restore
  useEffect(() => { setEdgeDraft(null) }, [current])
  function changeRestoreEdge(v: number) {
    if (!editedRestore) return
    setEdgeDraft(v)
    clearTimeout(edgeTimer.current)
    const at = current
    edgeTimer.current = window.setTimeout(async () => {
      const edit = { ...editedRestore, hardness: v }
      const url = await composeRestore(edit)
      setVersions((vs) => vs.map((x, n) => (n === at ? { ...x, url, restore: edit } : x)))
    }, 250)
  }

  // цвет: версия «Цвет» пересобирается на месте, пока тянут ползунки; область — где правка отличается от прошлой версии
  const [colorOpen, setColorOpen] = useState(false)
  const [colorWorking, setColorWorking] = useState(false)
  const [colorDraft, setColorDraft] = useState<ColorValues | null>(null)
  const [showRegion, setShowRegion] = useState(false)
  // кисть области: автоматическая область + правки кистью, свои у каждой версии «Цвет» (по паре prev / base)
  const [regionView, setRegionView] = useState<{ auto: HTMLCanvasElement; edits: RegionEdits } | null>(null)
  const [regionMode, setRegionMode] = useState<RegionMode | null>(null)
  const [regionSize, setRegionSize] = useState(48)
  const [regionTool, setRegionTool] = useState<DrawTool>('brush')
  const [regionUndo, setRegionUndo] = useState(0)
  const regionBrush = useRef<RegionBrushHandle>(null)
  const colorRegion = useRef<{ key: string; auto: HTMLCanvasElement } | null>(null)
  const regionEdits = useRef(new Map<string, RegionEdits>())
  const colorTimer = useRef(0)
  const editedColor = versions[current]?.color
  useEffect(() => { setColorDraft(null); setShowRegion(false); setRegionView(null); setRegionMode(null); setRegionUndo(0) }, [current])

  async function regionParts(e: ColorEdit) {
    const key = `${e.prev} ${e.base}`
    if (colorRegion.current?.key !== key) colorRegion.current = { key, auto: await changedRegion(e.prev, e.base) }
    const auto = colorRegion.current.auto
    let ed = regionEdits.current.get(key)
    if (!ed) { ed = newEdits(auto.width, auto.height); regionEdits.current.set(key, ed) }
    return { auto, edits: ed }
  }

  async function colorRegionFor(e: ColorEdit) {
    const { auto, edits } = await regionParts(e)
    return combineRegion(auto, edits, e.hardness ?? REGION_HARDNESS, e.offset ?? 0)
  }

  // «Цвет»: на версии с цветом — открыть ползунки; иначе новая версия «Цвет» поверх текущей (пока без изменений)
  function openColor() {
    if (current === 0 || busy) return
    if (!editedColor) {
      const edit: ColorEdit = { base: versions[current].url, prev: versions[current - 1].url, values: NEUTRAL }
      setVersions((v) => [...v.slice(0, current + 1), { url: edit.base, label: 'Цвет', color: edit }])
      setCurrent((c) => c + 1)
    }
    setColorOpen(true)
    setCompareOn(false)
  }

  function changeColor(values: ColorValues) {
    if (!editedColor) return
    setColorDraft(values)
    clearTimeout(colorTimer.current)
    const at = current
    const edit = { ...editedColor, values }
    colorTimer.current = window.setTimeout(async () => {
      setColorWorking(true)
      try {
        const url = await colorize(edit.base, await colorRegionFor(edit), values)
        setVersions((vs) => vs.map((x, n) => (n === at ? { ...x, url, color: edit } : x)))
      } finally {
        setColorWorking(false)
      }
    }, 120)
  }

  async function toggleRegion(on: boolean) {
    setShowRegion(on)
    if (on && editedColor) setRegionView(await regionParts(editedColor))
  }

  const toggleColor = () => (colorOpen && editedColor ? (setColorOpen(false), setRegionMode(null), setShowRegion(false)) : openColor())

  async function chooseRegionMode(m: RegionMode | null) {
    setRegionMode(m)
    if (m && editedColor) {
      setShowRegion(true)
      setRegionView(await regionParts(editedColor))
    }
  }

  // край области и его сдвиг — свойства версии «Цвет»: черновик сразу в панели и подсветке, пересборка — с задержкой
  const [regionDraft, setRegionDraft] = useState<{ hardness?: number; offset?: number }>({})
  const regionTimer = useRef(0)
  useEffect(() => { setRegionDraft({}) }, [current])
  function changeRegionEdge(patch: { hardness?: number; offset?: number }) {
    setRegionDraft((d) => ({ ...d, ...patch }))
    clearTimeout(regionTimer.current)
    regionTimer.current = window.setTimeout(() => regionChanged({ ...regionDraftRef.current, ...patch }), 150)
  }
  const regionDraftRef = useRef(regionDraft)
  regionDraftRef.current = regionDraft
  const regionHardness = regionDraft.hardness ?? editedColor?.hardness ?? REGION_HARDNESS
  const regionOffset = regionDraft.offset ?? editedColor?.offset ?? 0

  // мазок по области — версия «Цвет» пересобирается с теми же ползунками по новой области
  async function regionChanged(patch: { hardness?: number; offset?: number } = {}) {
    if (!editedColor) return
    const at = current
    const edit = {
      ...editedColor, values: colorDraft ?? editedColor.values,
      hardness: patch.hardness ?? regionDraftRef.current.hardness ?? editedColor.hardness,
      offset: patch.offset ?? regionDraftRef.current.offset ?? editedColor.offset,
    }
    setColorWorking(true)
    try {
      const url = await colorize(edit.base, await colorRegionFor(edit), edit.values)
      setVersions((vs) => vs.map((x, n) => (n === at ? { ...x, url, color: edit } : x)))
    } finally {
      setColorWorking(false)
    }
  }

  /* ---------------- drawing ---------------- */

  const toMask = (e: React.PointerEvent): Pt => {
    const r = surfaceRef.current!.getBoundingClientRect()
    return { x: ((e.clientX - r.left) / r.width) * maskW, y: ((e.clientY - r.top) / r.height) * maskH }
  }

  const stampSegment = (a: Pt, b: Pt) => {
    const restoring = tool === 'restore'
    const c = restoring ? restoreCtx() : ctx()
    const r = (size / 2) * k
    c.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over'
    c.fillStyle = restoring ? '#fff' : MODE_COLOR[mode]
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
    if (restoring) drawPreview()
  }

  const moveCursor = (e: React.PointerEvent) => {
    const el = cursorRef.current
    const r = surfaceRef.current?.getBoundingClientRect()
    if (!el || !r) return
    el.style.transform = `translate(${e.clientX - r.left - size / 2}px, ${e.clientY - r.top - size / 2}px)`
  }

  // фигура (прямоугольник / многоугольник): инструменты маски — или кисть отката в режиме фигуры
  const drawShape: 'rect' | 'poly' | null =
    tool === 'rect' || tool === 'poly' ? tool : tool === 'restore' && restoreShape !== 'brush' ? restoreShape : null

  const pushRestoreUndo = () => {
    restoreUndo.current.push(restoreCtx().getImageData(0, 0, maskW, maskH))
    if (restoreUndo.current.length > UNDO_LIMIT) restoreUndo.current.shift()
    restoreRedo.current = []
  }

  /** фигура → маска правки (цвет режима) или мазки отката */
  const fillShape = (pts: Pt[]) => {
    if (tool === 'restore') fillPolygon(restoreCtx(), pts, '#fff')
    else fillPolygon(ctx(), pts, MODE_COLOR[mode])
  }

  /** многоугольник → маска или откат, точки сбрасываются */
  const closePoly = (pts = poly) => {
    if (pts.length >= 3) {
      if (tool === 'restore') {
        pushRestoreUndo()
        fillShape(pts)
        syncRestore()
      } else {
        pushUndo()
        fillShape(pts)
        setBbox(computeBox())
      }
    }
    setPoly([])
    setRubber(null)
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (busy || e.button !== 0 || panTool) return
    if (drawShape === 'poly') {
      setError(null)
      const p = toMask(e)
      // двойной клик или клик в первую точку — замкнуть
      if (poly.length >= 3 && (e.detail >= 2 || Math.hypot(p.x - poly[0].x, p.y - poly[0].y) <= POLY_CLOSE * k)) {
        closePoly()
        return
      }
      if (e.detail >= 2) return
      setPoly((ps) => [...ps, p])
      return
    }
    if (tool === 'pick') {
      setError(null)
      const id = pickAt(toMask(e))
      if (id) togglePick(id)
      return
    }
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* синтетические события */ }
    if (tool === 'restore') pushRestoreUndo()
    else pushUndo()
    drawingRef.current = true
    setDrawing(true)
    setError(null)
    const p = toMask(e)
    if (tool === 'lasso') {
      lassoPts.current = [p]
      setLasso([p])
    } else if (drawShape === 'rect') {
      rectStart.current = p
      setLasso(rectPoints(p, p))
    } else {
      // Shift+клик — прямая от конца прошлого мазка (как в Photoshop)
      const from = e.shiftKey && lastEnd.current ? lastEnd.current : p
      lastPt.current = p
      anchor.current = p
      axis.current = null
      stampSegment(from, p)
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    moveCursor(e)
    if (tool === 'pick') {
      const id = busy || panTool ? 0 : pickAt(toMask(e))
      if (id !== hoverId) setHoverId(id)
      return
    }
    if (drawShape === 'poly') {
      if (poly.length && !busy && !panTool) setRubber(toMask(e))
      return
    }
    if (!drawingRef.current) return
    let p = toMask(e)
    if (drawShape === 'rect' && rectStart.current) {
      const a = rectStart.current
      if (e.shiftKey) {
        // Shift — квадрат
        const d = Math.max(Math.abs(p.x - a.x), Math.abs(p.y - a.y))
        p = { x: a.x + Math.sign(p.x - a.x || 1) * d, y: a.y + Math.sign(p.y - a.y || 1) * d }
      }
      setLasso(rectPoints(a, p))
      return
    }
    if (tool !== 'lasso' && e.shiftKey && anchor.current) {
      // Shift в мазке — строго по горизонтали или вертикали
      const locked = lockAxis(anchor.current, p, axis.current, 4 * k)
      p = locked.p
      axis.current = locked.axis
    } else {
      axis.current = null
    }
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
    } else if (drawShape === 'rect') {
      const pts = lasso
      if (pts.length === 4 && Math.abs(pts[2].x - pts[0].x) > 2 * k && Math.abs(pts[2].y - pts[0].y) > 2 * k) {
        fillShape(pts)
      } else if (tool === 'restore') {
        restoreUndo.current.pop()
      } else {
        undoStack.current.pop()
        syncHist()
      }
      rectStart.current = null
      setLasso([])
    }
    if (lastPt.current) lastEnd.current = lastPt.current
    lastPt.current = null
    if (tool === 'restore') syncRestore()
    else setBbox(computeBox())
  }

  /* ---------------- generation ---------------- */

  async function apply() {
    if (!bbox || busy) return
    const text = prompt.trim()
    // добавить / заменить: сначала план — рамка, где и какого размера встанет объект, её можно поправить
    if (mode === 'edit' && !plan) {
      const ctrl = new AbortController()
      abortRef.current = ctrl
      setBusy({ progress: 0.05, stage: 'Планирование' })
      setError(null)
      try {
        const p = await planInpaint(
          { imageUrl: versions[current].url, scene, model, llm, mask: maskRef.current!, mode, prompt: text }, ctrl.signal,
        )
        setBusy(null)
        if (p.box) { setPlan(p.box); return }
      } catch (e) {
        setBusy(null)
        if ((e as Error).name !== 'AbortError') setError((e as Error).message)
        return
      }
    }
    await generate(plan ?? undefined)
  }

  async function generate(box?: PlanRect) {
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setPlan(null)
    setBusy({ progress: 0, stage: '' })
    setError(null)
    try {
      const text = prompt.trim()
      const url = await inpaint(
        {
          imageUrl: versions[current].url, scene, model, llm, mask: maskRef.current!, mode, prompt: text || undefined, box,
          texture: mode === 'material' ? texture ?? undefined : undefined,
          textureWidthCm: mode === 'material' && texture && Number(textureWidth) > 0 ? Number(textureWidth) : undefined,
          exactMask: pick.current.ids.size > 0,
        },
        (progress, stage) => setBusy({ progress, stage }),
        ctrl.signal,
      )
      const label = mode === 'remove' ? (text ? `Удаление · ${text}` : 'Удаление') : mode === 'material' ? `Материал · ${text || (texture ? 'по образцу' : '')}` : text
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

  const clearCurrent = () => (tool === 'restore' ? clearRestore() : clearMask())
  // многоугольник в работе: Enter — замкнуть, Backspace — убрать точку, Esc — отменить (а не очистить маску)
  const polyKey = (key: string) => {
    if (drawShape !== 'poly' || !poly.length) return false
    if (key === 'Enter') closePoly()
    else if (key === 'Backspace') setPoly((ps) => ps.slice(0, -1))
    else if (key === 'Escape') { setPoly([]); setRubber(null) }
    else return false
    return true
  }
  const togglePeek = () => { if (tool === 'restore') setRestorePeek((v) => !v) }
  const handlers = useRef({ undo, redo, clearCurrent, setSize, chooseTool, polyKey, togglePeek, toggleColor })
  handlers.current = { undo, redo, clearCurrent, setSize, chooseTool, polyKey, togglePeek, toggleColor }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT') return
      const h = handlers.current
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.code === 'KeyZ') { e.preventDefault(); e.shiftKey ? h.redo() : h.undo(); return }
      if (mod && e.code === 'KeyY') { e.preventDefault(); h.redo(); return }
      if (mod) return
      if (h.polyKey(e.key)) { e.preventDefault(); return }
      if (e.code === 'KeyL') h.chooseTool('lasso')
      else if (e.code === 'KeyP') h.chooseTool('poly')
      else if (e.code === 'KeyU') h.chooseTool('rect')
      else if (e.code === 'KeyV') h.togglePeek()
      else if (e.code === 'KeyC') h.toggleColor()
      else if (e.code === 'KeyB') h.chooseTool('brush')
      else if (e.code === 'KeyE') h.chooseTool('eraser')
      else if (e.code === 'KeyH') h.chooseTool('hand')
      else if (e.code === 'KeyR') h.chooseTool('restore')
      else if (e.code === 'KeyM') h.chooseTool('pick')
      else if (e.code === 'BracketLeft') h.setSize((s) => Math.max(BRUSH_MIN, s - 6))
      else if (e.code === 'BracketRight') h.setSize((s) => Math.min(BRUSH_MAX, s + 6))
      else if (e.key === 'Escape') h.clearCurrent()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  /* ---------------- popover placement ---------------- */

  const showPopover = !!bbox && !drawing && !busy && !peek && !panning && !compare && !plan && tool !== 'restore' && !poly.length
    && (tool !== 'pick' || pickOpen)
  const showPickBar = tool === 'pick' && !!bbox && !pickOpen && !busy && !peek && !compare && !plan
  let pop = { x: 0, y: 0, placement: 'bottom' as 'top' | 'bottom' }
  if (bbox) {
    const bx = left + bbox.x / k
    const by = top + bbox.y / k
    const bw = bbox.w / k
    const bh = bbox.h / k
    const est = mode === 'remove' ? 230 : mode === 'material' ? 330 : 250
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

  const shown = (peek || peekPrev) && current > 0 ? versions[current - 1] : versions[current]
  const brushLike = (tool === 'brush' || tool === 'eraser' || tool === 'restore') && !spaceHeld && !compare
  const brushRing = brushLike && !drawShape
  const color = MODE_COLOR[mode]

  return (
    <div className="flex flex-col gap-4">
      <div
        ref={stageRef}
        className={`checker relative h-[min(78vh,760px)] min-h-[480px] touch-none select-none overflow-hidden rounded-[28px] shadow-card ring-1 ring-fg/[0.04] ${
          panning ? 'cursor-grabbing' : panTool ? 'cursor-grab' : ''
        }`}
        {...panHandlers}
        // файл, брошенный мимо поля образца, браузер открыл бы вместо страницы; в «Материалах» это и есть образец
        onDragOver={(e) => { if (e.dataTransfer.types.includes('Files')) e.preventDefault() }}
        onDrop={(e) => {
          e.preventDefault()
          const f = e.dataTransfer.files[0]
          if (mode === 'material' && f && isTextureFile(f) && f.size <= TEXTURE_MAX) { setTexture(f); setTextureWidth('') }
        }}
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

          {/* цвет: где действуют ползунки */}
          {/* кисть отката: предыдущая версия, проступающая под мазками */}
          <canvas
            ref={previewRef}
            width={maskW}
            height={maskH}
            className={`pointer-events-none absolute inset-0 size-full ${peek || compare ? 'opacity-0' : ''}`}
          />

          <div
            className={`absolute inset-0 transition-opacity duration-300 ${peek || compare || tool === 'restore' ? 'opacity-0' : ''}`}
            style={{ filter: drawing ? undefined : `drop-shadow(0 0 1.5px rgba(255,255,255,0.9)) drop-shadow(0 0 10px ${color})` }}
          >
            <canvas
              ref={maskRef}
              width={maskW}
              height={maskH}
              className={`absolute inset-0 size-full ${busy || (showPopover && mode === 'remove') ? 'animate-breathe' : 'opacity-50'}`}
            />
          </div>

          {/* выбор маски: элемент схемы под курсором */}
          {canPick && (
            <canvas
              ref={hoverRef}
              className={`pointer-events-none absolute inset-0 size-full ${peek || compare || tool !== 'pick' ? 'opacity-0' : ''}`}
              style={{ imageRendering: 'pixelated' }}
            />
          )}

          {/* план правки: рамка объекта — подвинуть, растянуть, затем сгенерировать */}
          {plan && !busy && (
            <PlanBox
              box={plan}
              onChange={setPlan}
              onConfirm={() => generate(plan)}
              onCancel={() => setPlan(null)}
              width={dispW}
              height={dispH}
            />
          )}

          {/* превью многоугольника: точки и «резинка» до курсора */}
          {poly.length > 0 && (
            <svg viewBox={`0 0 ${maskW} ${maskH}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 size-full">
              <path d={`M${[...poly, ...(rubber ? [rubber] : [])].map((p) => `${p.x},${p.y}`).join('L')}Z`} fill={color} fillOpacity={0.18} stroke="none" />
              <path
                d={`M${[...poly, ...(rubber ? [rubber] : [])].map((p) => `${p.x},${p.y}`).join('L')}`}
                fill="none" stroke="#fff" strokeWidth={1.5} strokeDasharray="6 6" vectorEffect="non-scaling-stroke" className="animate-ants"
              />
              {poly.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={(i ? 3 : 5) * k} fill={i ? '#fff' : color} stroke="#000" strokeOpacity={0.4} strokeWidth={1} vectorEffect="non-scaling-stroke" />
              ))}
            </svg>
          )}

          {/* превью лассо и прямоугольника */}
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
              panning ? 'cursor-grabbing' : panTool ? 'cursor-grab' : busy ? 'cursor-wait' : brushRing ? 'cursor-none' : tool === 'pick' ? (hoverId ? 'cursor-pointer' : 'cursor-default') : 'cursor-crosshair'
            }`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onPointerEnter={() => cursorRef.current && (cursorRef.current.style.opacity = '1')}
            onPointerLeave={() => { if (cursorRef.current) cursorRef.current.style.opacity = '0'; setHoverId(0) }}
          >
            {brushRing && !busy && (
              <div
                ref={cursorRef}
                className={`pointer-events-none absolute left-0 top-0 opacity-0 transition-[width,height,border-radius] duration-150 ${shape === 'circle' ? 'rounded-full' : 'rounded-[3px]'}`}
                style={{
                  width: size,
                  height: size,
                  border: `1.5px ${tool === 'eraser' ? 'dashed' : tool === 'restore' ? 'dotted' : 'solid'} #fff`,
                  boxShadow: '0 0 0 1px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(0,0,0,0.15)',
                  background: tool === 'eraser' ? 'rgba(255,255,255,0.12)' : tool === 'restore' ? 'rgba(255,255,255,0.08)' : `${color}22`,
                }}
              />
            )}
          </div>

          {/* область цвета: подсветка и кисть «добавить / убрать» — поверх поверхности рисования, пока кисть включена */}
          {colorOpen && editedColor && (showRegion || regionMode) && regionView && !peek && !busy && (
            <RegionBrush
              ref={regionBrush} auto={regionView.auto} edits={regionView.edits} mode={regionMode} size={regionSize}
              tool={regionTool} hardness={regionHardness} offset={regionOffset}
              onCommit={() => regionChanged()} onHistory={setRegionUndo} className="z-10"
            />
          )}
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
            {(tool === 'restore' ? !restoreDirty : !bbox || (tool === 'poly' && poly.length > 0)) && !busy && !drawing && !compare && (
              <motion.div
                initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="glass rounded-full px-3.5 py-1.5 text-[12.5px] text-ink-2 shadow-soft ring-1 ring-fg/5"
              >
                {tool === 'hand'
                  ? 'Перетаскивайте изображение · колесо — масштаб'
                  : tool === 'restore'
                  ? `Закрасьте места, где вернуть «${versions[current - 1]?.label ?? ''}»`
                  : tool === 'pick'
                  ? `Кликните по элементам схемы: ${mode === 'edit' ? 'что изменить' : mode === 'material' ? 'поверхности, которым сменить материал' : 'что удалить'}`
                  : tool === 'poly'
                  ? poly.length ? 'Кликайте точки · Enter или клик в первую точку — замкнуть · Backspace — убрать точку · Esc — отменить' : 'Ставьте точки кликами — стороны прямые'
                  : tool === 'rect'
                  ? 'Протяните прямоугольник · Shift — квадрат'
                  : `${tool === 'lasso' ? 'Обведите' : 'Закрасьте'} ${mode === 'edit' ? 'область, которую нужно изменить' : mode === 'material' ? 'поверхность, которой нужно сменить материал' : 'объект, который нужно удалить'}`}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* цвет сгенерированного */}
        <AnimatePresence>
          {colorOpen && editedColor && !busy && !compare && (
            <motion.div
              initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              className="absolute right-16 top-16 z-30"
            >
              <ColorPanel
                values={colorDraft ?? editedColor.values} onChange={changeColor} working={colorWorking}
                showRegion={showRegion} onShowRegion={toggleRegion}
                regionMode={regionMode} onRegionMode={chooseRegionMode}
                brushSize={regionSize} onBrushSize={setRegionSize}
                undoable={regionUndo} onUndo={() => regionBrush.current?.undo()}
                regionTool={regionTool} onRegionTool={setRegionTool}
                regionHardness={regionHardness} onRegionHardness={(v) => changeRegionEdge({ hardness: v })}
                regionOffset={regionOffset} onRegionOffset={(v) => changeRegionEdge({ offset: v })}
                onDone={() => { setColorOpen(false); setShowRegion(false); setRegionMode(null) }}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* слева: инструменты */}
        <div className="absolute left-4 top-1/2 z-20 -translate-y-1/2">
          <Toolbar
            tool={tool}
            onTool={chooseTool}
            canUndo={hist.undo > 0}
            canRedo={hist.redo > 0}
            hasMask={tool === 'restore' ? restoreDirty : !!bbox}
            onUndo={undo}
            onRedo={redo}
            onClear={clearCurrent}
            canRestore={current > 0}
            canPick={canPick}
            onBuildMasks={buildMasks}
            buildingMasks={buildingMasks}
            disabled={!!busy}
            onColor={toggleColor}
            colorActive={colorOpen && !!editedColor}
            canColor={current > 0}
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

        {/* низ: край применённого отката (без кисти — отдельно, с кистью — над её настройками) */}
        <AnimatePresence>
          {editedRestore && !brushLike && !busy && !showPickBar && !compare && (
            <motion.div
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
              transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              className="absolute inset-x-0 bottom-4 flex justify-center"
            >
              <RestoreEdge value={edgeDraft ?? editedRestore.hardness} onChange={changeRestoreEdge} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* низ: параметры кисти */}
        <AnimatePresence>
          {brushLike && !busy && (
            <motion.div
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
              transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              className="absolute inset-x-0 bottom-4 flex justify-center"
            >
              <div className="flex flex-col items-center gap-2">
                {editedRestore && !restoreDirty && <RestoreEdge value={edgeDraft ?? editedRestore.hardness} onChange={changeRestoreEdge} />}
                {tool === 'restore' && (
                  <button
                    onClick={() => setRestorePeek((v) => !v)}
                    title="Показать прошлую версию целиком и вести границы по ней (V)"
                    className={`glass flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[12.5px] font-medium shadow-float ring-1 ring-fg/5 transition-colors ${restorePeek ? 'bg-[#f59e0b] text-white' : 'text-ink-2 hover:text-ink'}`}
                  >
                    {restorePeek ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />} Прошлый «{versions[current - 1]?.label}»
                  </button>
                )}
                {tool === 'restore' && (
                  <div className="glass flex items-center gap-0.5 rounded-full p-1 shadow-float ring-1 ring-fg/5">
                    {([
                      ['brush', Brush, 'Кисть отката'],
                      ['rect', SquareDashed, 'Откат прямоугольником — протяните; Shift — квадрат'],
                      ['poly', Pentagon, 'Откат многоугольником — точки кликами; Enter или клик в первую точку'],
                    ] as const).map(([t, Icon, title]) => (
                      <button
                        key={t}
                        onClick={() => setRestoreShape(t)}
                        title={title}
                        className={`grid size-8 place-items-center rounded-full transition-colors ${restoreShape === t ? 'bg-ink text-on-ink' : 'text-ink-2 hover:bg-fg/[0.06] hover:text-ink'}`}
                      >
                        <Icon className="size-4" strokeWidth={1.75} />
                      </button>
                    ))}
                  </div>
                )}
                {tool === 'restore' && restoreDirty && (
                  <div className="glass flex items-center gap-1 rounded-full p-1 shadow-float ring-1 ring-fg/5">
                    <span className="flex items-center gap-1.5 pl-2.5 pr-1 text-[12.5px] text-ink-2">
                      <History className="size-3.5" /> Вернуть «{versions[current - 1]?.label}»
                    </span>
                    <button
                      onClick={() => clearRestore()}
                      className="rounded-full px-3 py-1.5 text-[12.5px] font-medium text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink"
                    >
                      Сбросить
                    </button>
                    <button
                      onClick={applyRestore}
                      className="flex items-center gap-1.5 rounded-full bg-ink px-3.5 py-1.5 text-[12.5px] font-medium text-on-ink transition-colors hover:bg-ink-hover"
                    >
                      <Check className="size-3.5" /> Применить откат
                    </button>
                  </div>
                )}
                <BrushOptions
                  shape={shape} onShape={setShape} size={size} onSize={setSize} min={BRUSH_MIN} max={BRUSH_MAX}
                  {...(tool === 'restore' ? { hardness, onHardness: setHardness } : {})}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* выбор маски: сколько выбрано и переход к действию */}
        <AnimatePresence>
          {showPickBar && (
            <motion.div
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
              transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              className="absolute inset-x-0 bottom-4 flex justify-center"
            >
              <div className="glass flex items-center gap-1 rounded-full p-1 shadow-float ring-1 ring-fg/5">
                <span className="flex items-center gap-2 pl-3 pr-1.5 text-[12.5px] text-ink-2">
                  <span className="size-2 rounded-full" style={{ background: color }} />
                  {pickCount ? `Выбрано элементов: ${pickCount}` : 'Маска готова'}
                </span>
                <button
                  onClick={() => clearMask()}
                  className="rounded-full px-3 py-1.5 text-[12.5px] font-medium text-mute transition-colors hover:bg-fg/[0.06] hover:text-ink"
                >
                  Сбросить
                </button>
                <button
                  onClick={() => setPickOpen(true)}
                  className="flex items-center gap-1.5 rounded-full bg-ink px-3.5 py-1.5 text-[12.5px] font-medium text-on-ink transition-colors hover:bg-ink-hover"
                >
                  Далее <ArrowRight className="size-3.5" />
                </button>
              </div>
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
              texture={texture}
              onTexture={(f) => { setTexture(f); setTextureWidth('') }}
              textureWidth={textureWidth}
              onTextureWidth={setTextureWidth}
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
              key={`${i}-${v.url}`}
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

/** Край уже применённого отката кистью: версия пересобирается с другой мягкостью. */
function RestoreEdge({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="glass flex items-center gap-3 rounded-full py-2 pl-3.5 pr-4 shadow-float ring-1 ring-fg/5">
      <History className="size-3.5 text-ink-2" />
      <span className="text-[12.5px] font-medium text-ink-2">Край отката</span>
      <input
        type="range" min={0} max={1} step={0.05} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="range w-28 sm:w-36"
        style={{ '--fill': `${value * 100}%` } as React.CSSProperties}
        aria-label="Жёсткость края отката"
      />
      <span className="w-14 text-[12px] font-medium text-ink-2">{value > 0.8 ? 'жёсткий' : value < 0.3 ? 'мягкий' : 'средний'}</span>
    </div>
  )
}
