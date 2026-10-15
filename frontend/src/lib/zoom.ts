import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

/** Масштаб и сдвиг изображения относительно вписанного положения (экранные px) */
export interface View { zoom: number; x: number; y: number }

/** Геометрия сцены: вписанный размер изображения, рабочая область и её центр (в координатах сцены) */
export interface ZoomGeometry {
  baseW: number
  baseH: number
  availW: number
  availH: number
  cx: number
  cy: number
}

interface Pt { x: number; y: number }

export const ZOOM_MIN = 1
export const ZOOM_MAX = 8
// шаги кнопок и клавиш; колесо масштабирует плавно
const ZOOM_STEPS = [1, 1.5, 2, 3, 4, 6, 8]
const FIT: View = { zoom: 1, x: 0, y: 0 }
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v))

// поля ввода и окна поверх сцены не масштабируют и не перетаскивают её
const IGNORE = 'button, textarea, input, select, [data-no-zoom]'

/**
 * Масштаб и перемещение изображения в сцене (Inpainting, Img to Img):
 * колесо / пинч — плавный масштаб под курсором, клавиши = − 0, перемещение — пробел + перетаскивание,
 * средняя кнопка мыши, а с panWithLeft — и левая (при масштабе больше 100%).
 * Обработчики panHandlers вешаются на сцену; capture-фаза не даёт рисованию и ползункам перехватить перетаскивание.
 */
export function useZoomPan(stageRef: RefObject<HTMLElement | null>, geometry: ZoomGeometry, opts: { panWithLeft?: boolean } = {}) {
  const [view, setView] = useState<View>(FIT)
  const [panning, setPanning] = useState(false)
  const [spaceHeld, setSpaceHeld] = useState(false)
  const panRef = useRef<Pt | null>(null)
  const geo = useRef(geometry)
  geo.current = geometry

  /** Сдвиг не даёт увести изображение: в рабочей области остаётся не меньше половины его (или её) ширины и высоты.
   *  Свободнее, чем «край к краю»: иначе масштаб у края кадра не держит точку под курсором. На 100% — по центру. */
  const clampView = useCallback((v: View): View => {
    if (v.zoom <= ZOOM_MIN) return FIT
    const g = geo.current
    const w = g.baseW * v.zoom
    const h = g.baseH * v.zoom
    const mx = (w + g.availW) / 2 - Math.min(w, g.availW) / 2
    const my = (h + g.availH) / 2 - Math.min(h, g.availH) / 2
    return { zoom: v.zoom, x: clamp(v.x, -mx, mx), y: clamp(v.y, -my, my) }
  }, [])

  /** Масштаб вокруг точки сцены (по умолчанию — центр области): точка изображения под ней остаётся на месте */
  const zoomWith = useCallback((next: (z: number) => number, at?: Pt) => {
    setView((v) => {
      const g = geo.current
      const z = clamp(next(v.zoom), ZOOM_MIN, ZOOM_MAX)
      if (z === v.zoom) return v
      const px = (at?.x ?? g.cx) - g.cx
      const py = (at?.y ?? g.cy) - g.cy
      const r = z / v.zoom
      return clampView({ zoom: z, x: px - (px - v.x) * r, y: py - (py - v.y) * r })
    })
  }, [clampView])

  const zoomIn = useCallback(() => zoomWith((z) => ZOOM_STEPS.find((s) => s > z + 0.01) ?? ZOOM_MAX), [zoomWith])
  const zoomOut = useCallback(() => zoomWith((z) => [...ZOOM_STEPS].reverse().find((s) => s < z - 0.01) ?? ZOOM_MIN), [zoomWith])
  const zoomFit = useCallback(() => setView(FIT), [])

  // колесо / пинч тачпада (нативный слушатель: React вешает wheel пассивным, а прокрутку страницы надо отменить)
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if ((e.target as HTMLElement).closest('textarea, [data-no-zoom]')) return
      e.preventDefault()
      const r = el.getBoundingClientRect()
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY
      const f = Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.002))
      zoomWith((z) => z * f, { x: e.clientX - r.left, y: e.clientY - r.top })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [stageRef, zoomWith])

  // при смене размера сцены сдвиг остаётся в допустимых границах
  useEffect(() => setView(clampView), [geometry.availW, geometry.availH, geometry.baseW, geometry.baseH, clampView])

  // клавиши — только пока сцена на экране (вкладка Img to Img остаётся смонтированной скрытой) и не в поле ввода
  useEffect(() => {
    const typing = (t: EventTarget | null) => t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target) || !stageRef.current?.offsetParent) return
      if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) setSpaceHeld(true) }
      else if (e.code === 'Equal' || e.code === 'NumpadAdd') zoomIn()
      else if (e.code === 'Minus' || e.code === 'NumpadSubtract') zoomOut()
      else if (e.code === 'Digit0' || e.code === 'Numpad0') zoomFit()
    }
    const onKeyUp = (e: KeyboardEvent) => e.code === 'Space' && setSpaceHeld(false)
    const onBlur = () => setSpaceHeld(false)
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [stageRef, zoomIn, zoomOut, zoomFit])

  /* ---------------- перемещение ---------------- */

  const panTool = spaceHeld || (!!opts.panWithLeft && view.zoom > ZOOM_MIN)

  const panHandlers = {
    onPointerDownCapture: (e: React.PointerEvent) => {
      if (!(e.button === 1 || (e.button === 0 && panTool))) return
      if ((e.target as HTMLElement).closest(IGNORE)) return
      e.preventDefault()
      e.stopPropagation() // не рисуем и не двигаем ползунок сравнения
      try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* синтетические события */ }
      panRef.current = { x: e.clientX, y: e.clientY }
      setPanning(true)
    },
    onPointerMove: (e: React.PointerEvent) => {
      const p = panRef.current
      if (!p) return
      const dx = e.clientX - p.x
      const dy = e.clientY - p.y
      panRef.current = { x: e.clientX, y: e.clientY }
      setView((v) => clampView({ ...v, x: v.x + dx, y: v.y + dy }))
    },
    onPointerUp: () => {
      if (!panRef.current) return
      panRef.current = null
      setPanning(false)
    },
    onPointerCancel: () => {
      panRef.current = null
      setPanning(false)
    },
    // без автопрокрутки средней кнопкой
    onMouseDown: (e: React.MouseEvent) => e.button === 1 && e.preventDefault(),
  }

  return { view, zoomIn, zoomOut, zoomFit, panning, panTool, spaceHeld, panHandlers }
}
