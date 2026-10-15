import { motion } from 'motion/react'
import { ArrowLeft } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { Editor } from '../components/inpaint/Editor'
import { toStudioImage, type StudioImage } from '../lib/types'

interface Props {
  image: StudioImage | null
  onImage: (image: StudioImage | null) => void
  onBack: () => void
}

const spring = { type: 'spring', stiffness: 260, damping: 30 } as const

export function Inpainting({ image, onImage, onBack }: Props) {
  return (
    <div className="mx-auto max-w-[1280px] px-5 pb-24 sm:px-8">
      <section className="pb-10 pt-12 text-center sm:pb-12 sm:pt-16">
        <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="text-[15px] font-semibold text-gradient sm:text-[17px]">
          Inpainting
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.06 }}
          className="mt-3 text-[44px] font-semibold leading-[1.04] tracking-[-0.035em] sm:text-[64px]"
        >
          Каждая деталь.
          <br />
          <span className="text-faint">Под контролем.</span>
        </motion.h1>
      </section>

      <motion.section
        initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.12 }}
        className="mb-6 grid gap-px overflow-hidden rounded-[28px] bg-line ring-1 ring-fg/[0.04] sm:grid-cols-3"
      >
        {[
          ['01', 'Выберите режим', 'Добавить или изменить — либо удалить лишнее.'],
          ['02', 'Выделите область', 'Лассо для точного контура, кисть для свободной маски.'],
          ['03', 'Подтвердите', 'Опишите изменение или просто удалите — результат появится в истории.'],
        ].map(([n, t, d]) => (
          <div key={n} className="bg-surface p-6 sm:px-7">
            <div className="text-[13px] font-semibold text-gradient">{n}</div>
            <div className="mt-2 text-[19px] font-semibold tracking-[-0.02em]">{t}</div>
            <p className="mt-1.5 text-[14px] leading-relaxed text-mute">{d}</p>
          </div>
        ))}
      </motion.section>

      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.2 }}>
        {image ? (
          <Editor key={image.url} image={image} onClose={() => onImage(null)} />
        ) : (
          <div className="relative h-[400px] sm:h-auto overflow-hidden rounded-[28px] bg-surface shadow-card ring-1 ring-fg/[0.04] sm:aspect-[16/9] sm:min-h-0">
            <Dropzone onFile={async (f) => onImage(await toStudioImage(f))} listenPaste />
            <button
              onClick={onBack}
              className="group absolute bottom-5 left-1/2 inline-flex -translate-x-1/2 items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-[13px] font-medium text-mute transition-colors hover:bg-fg/[0.04] hover:text-ink sm:bottom-8"
            >
              <ArrowLeft className="size-3.5 transition-transform duration-300 ease-apple group-hover:-translate-x-0.5" />
              или перенесите результат из Img to Img
            </button>
          </div>
        )}
      </motion.div>
    </div>
  )
}
