import { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { ArrowUp, ImagePlus } from 'lucide-react'

interface Props {
  onFile: (file: File) => void
  /** Слушать Ctrl+V по всему окну */
  listenPaste?: boolean
  compact?: boolean
  title?: string
  hint?: string
}

const isImage = (f: File) => f.type.startsWith('image/')

export function Dropzone({ onFile, listenPaste, compact, title, hint }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const root = useRef<HTMLDivElement>(null)
  const [over, setOver] = useState(false)
  const depth = useRef(0)

  const take = useCallback(
    (files: FileList | null | undefined) => {
      const f = files && Array.from(files).find(isImage)
      if (f) onFile(f)
    },
    [onFile],
  )

  useEffect(() => {
    if (!listenPaste) return
    // вставляем только в видимую зону (скрытые вкладки остаются смонтированными)
    const onPaste = (e: ClipboardEvent) => root.current?.offsetParent && take(e.clipboardData?.files)
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [listenPaste, take])

  const handlers = {
    onDragEnter: (e: React.DragEvent) => { e.preventDefault(); depth.current++; setOver(true) },
    onDragLeave: (e: React.DragEvent) => { e.preventDefault(); if (--depth.current <= 0) setOver(false) },
    onDragOver: (e: React.DragEvent) => e.preventDefault(),
    onDrop: (e: React.DragEvent) => { e.preventDefault(); depth.current = 0; setOver(false); take(e.dataTransfer.files) },
  }

  const picker = (
    <input ref={input} type="file" accept="image/*" hidden onChange={(e) => { take(e.target.files); e.target.value = '' }} />
  )

  if (compact) {
    return (
      <button
        type="button"
        {...handlers}
        onClick={() => input.current?.click()}
        className={`flex w-full items-center gap-3 rounded-2xl border border-dashed px-4 py-3.5 text-left transition-all duration-300 ease-apple ${
          over ? 'border-accent bg-accent/[0.04]' : 'border-fg/15 hover:border-fg/30 hover:bg-fg/[0.02]'
        }`}
      >
        <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-fg/[0.05]">
          <ImagePlus className="size-4 text-ink-2" strokeWidth={1.75} />
        </span>
        <span className="min-w-0">
          <span className="block text-[13px] font-medium">{title ?? 'Загрузить изображение'}</span>
          <span className="block truncate text-[12px] text-faint">{hint ?? 'PNG, JPG, WEBP'}</span>
        </span>
        {picker}
      </button>
    )
  }

  return (
    <div
      ref={root}
      {...handlers}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      className="group relative flex size-full cursor-pointer flex-col items-center justify-center overflow-hidden outline-none"
    >
      {/* мягкое свечение */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-700 ease-apple ${over ? 'opacity-100' : 'opacity-0'}`}
        style={{ background: 'radial-gradient(60% 60% at 50% 50%, rgba(0,113,227,0.10), transparent 70%)' }}
      />
      <div
        className={`pointer-events-none absolute inset-4 rounded-[22px] border-[1.5px] border-dashed transition-all duration-500 ease-apple sm:inset-6 ${
          over ? 'scale-[0.985] border-accent' : 'border-fg/[0.10] group-hover:border-fg/20'
        }`}
      />

      <motion.div
        animate={{ y: over ? -6 : 0, scale: over ? 1.06 : 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 22 }}
        className="relative mb-7"
      >
        <StackedCards active={over} />
      </motion.div>

      <h3 className="relative px-6 text-center text-[22px] font-semibold tracking-[-0.02em] sm:text-[26px]">
        {over ? 'Отпускайте' : 'Перетащите рендер сюда'}
      </h3>
      <p className="relative mt-2 px-6 text-center text-[15px] text-mute">
        или <span className="text-accent">выберите файл</span>
        <span className="hidden sm:inline"> · можно вставить через </span>
        <kbd className="hidden rounded-md border border-fg/10 bg-surface px-1.5 py-0.5 font-sans text-[12px] text-ink-2 shadow-[0_1px_0_rgba(0,0,0,0.06)] sm:inline">
          Ctrl V
        </kbd>
      </p>
      <p className="relative mt-6 text-[12px] tracking-wide text-faint">PNG · JPG · WEBP · до 50 МБ</p>
      {picker}
    </div>
  )
}

function StackedCards({ active }: { active: boolean }) {
  return (
    <div className="relative h-[104px] w-[132px]">
      <div
        className="absolute inset-0 rounded-2xl shadow-soft transition-transform duration-500 ease-apple"
        style={{
          background: 'linear-gradient(180deg,#ff9a5a,#ffe2a8)',
          transform: active ? 'rotate(-14deg) translate(-22px,6px)' : 'rotate(-8deg) translate(-12px,4px)',
        }}
      />
      <div
        className="absolute inset-0 rounded-2xl shadow-soft transition-transform duration-500 ease-apple"
        style={{
          background: 'linear-gradient(180deg,#2b3a78,#c7739a)',
          transform: active ? 'rotate(14deg) translate(22px,6px)' : 'rotate(8deg) translate(12px,4px)',
        }}
      />
      <div className="absolute inset-0 grid place-items-center rounded-2xl bg-surface shadow-card ring-1 ring-fg/[0.04]">
        <span className="grid size-11 place-items-center rounded-full bg-ink text-on-ink transition-transform duration-500 ease-apple group-hover:-translate-y-0.5">
          <ArrowUp className="size-5" strokeWidth={2.25} />
        </span>
      </div>
    </div>
  )
}
