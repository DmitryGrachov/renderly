import { useRef, useState } from 'react'
import { motion } from 'motion/react'
import { Download, Layers, Loader2, X } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { toStudioImage, type StudioImage } from '../lib/types'
import { buildMaterials, type RenderPassResult } from '../lib/materials'

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const

export function RenderPass() {
  const [image, setImage] = useState<StudioImage | null>(null)
  const [result, setResult] = useState<RenderPassResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  const start = async (file: File) => {
    abort.current?.abort()
    const ctrl = new AbortController()
    abort.current = ctrl
    setError(null)
    setResult(null)
    setImage(await toStudioImage(file))
    try {
      setResult(await buildMaterials(file, ctrl.signal))
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    }
  }

  const reset = () => {
    abort.current?.abort()
    setImage(null)
    setResult(null)
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
          Кадр плоской схемой.
          <br />
          <span className="text-faint">Без света, теней и фактуры.</span>
        </motion.h1>
      </section>

      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.12 }}>
        {!image ? (
          <div className="relative h-[400px] overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04] sm:aspect-[16/9] sm:h-auto">
            <Dropzone onFile={start} listenPaste />
          </div>
        ) : (
          <Viewer image={image} result={result} error={error} onReset={reset} onRetry={() => start(image.file)} />
        )}
      </motion.div>
    </div>
  )
}

interface ViewerProps {
  image: StudioImage
  result: RenderPassResult | null
  error: string | null
  onReset: () => void
  onRetry: () => void
}

function Viewer({ image, result, error, onReset, onRetry }: ViewerProps) {
  const [showPhoto, setShowPhoto] = useState(false)

  const download = () => {
    if (!result) return
    const a = document.createElement('a')
    a.href = result.schema
    a.download = `inego_flat_${image.name.replace(/\.[^.]+$/, '')}.png`
    a.click()
  }

  return (
    <div className="relative overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04]">
      <img
        src={result && !showPhoto ? result.schema : image.url}
        alt={image.name}
        className="block w-full select-none"
        draggable={false}
      />
      {!result && !error && (
        <div className="absolute inset-0 grid place-items-center bg-black/25 backdrop-blur-[2px]">
          <div className="flex items-center gap-2.5 rounded-full bg-surface/90 px-4 py-2 text-[14px] font-medium shadow-card">
            <Loader2 className="size-4 animate-spin text-accent" />
            Рисуем схему… около минуты
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
      <div className="absolute right-3 top-3 flex gap-2">
        {result && (
          <>
            <button
              onClick={() => setShowPhoto((v) => !v)}
              title={showPhoto ? 'Показать схему' : 'Показать кадр'}
              className={`grid size-9 place-items-center rounded-full bg-surface/85 shadow-soft backdrop-blur transition-colors hover:bg-surface ${showPhoto ? 'text-accent' : 'text-ink'}`}
            >
              <Layers className="size-4" />
            </button>
            <button
              onClick={download}
              title="Скачать схему"
              className="grid size-9 place-items-center rounded-full bg-surface/85 text-ink shadow-soft backdrop-blur transition-colors hover:bg-surface"
            >
              <Download className="size-4" />
            </button>
          </>
        )}
        <button
          onClick={onReset}
          aria-label="Другой кадр"
          className="grid size-9 place-items-center rounded-full bg-surface/85 text-ink shadow-soft backdrop-blur transition-colors hover:bg-surface"
        >
          <X className="size-4" />
        </button>
      </div>
    </div>
  )
}
