import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle2 } from 'lucide-react'
import { Nav, type Tab } from './components/Nav'
import { ImgToImg } from './pages/ImgToImg'
import { Inpainting } from './pages/Inpainting'
import type { StudioImage } from './lib/types'

const readTab = (): Tab => (location.hash === '#/inpainting' ? 'inpainting' : 'img2img')

export default function App() {
  const [tab, setTab] = useState<Tab>(readTab)
  const [inpaintImage, setInpaintImage] = useState<StudioImage | null>(null)
  const [toast, setToast] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setTab(readTab())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 2600)
    return () => clearTimeout(t)
  }, [toast])

  const go = (t: Tab) => {
    location.hash = `#/${t}`
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-dvh">
      <Nav tab={tab} onChange={go} inpaintBadge={!!inpaintImage} />

      {/* Обе страницы остаются смонтированными, чтобы не терять состояние */}
      <main>
        <div hidden={tab !== 'img2img'}>
          <ImgToImg
            onSendToInpaint={(img) => {
              setInpaintImage(img)
              setToast('Изображение перенесено в Inpainting')
              go('inpainting')
            }}
          />
        </div>
        <div hidden={tab !== 'inpainting'}>
          {tab === 'inpainting' && <Inpainting image={inpaintImage} onImage={setInpaintImage} onBack={() => go('img2img')} />}
        </div>
      </main>

      <footer className="border-t border-black/[0.06]">
        <div className="mx-auto flex max-w-[1280px] px-5 py-8 text-[12px] text-faint sm:px-8">
          <span>© {new Date().getFullYear()} INEGO.dev — AI-инструменты для архитектурной визуализации</span>
        </div>
      </footer>

      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 380, damping: 30 }}
            className="fixed inset-x-0 bottom-6 z-50 mx-auto flex w-fit items-center gap-2.5 rounded-full bg-ink/90 px-5 py-3 text-[14px] font-medium text-white shadow-float backdrop-blur-xl"
          >
            <CheckCircle2 className="size-[18px] text-emerald-400" />
            {toast}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
