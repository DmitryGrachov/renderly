import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle2 } from 'lucide-react'
import { Logo, Nav, type Tab } from './components/Nav'
import { ImgToImg } from './pages/ImgToImg'
import { Inpainting } from './pages/Inpainting'
import { RenderPass } from './pages/RenderPass'
import { Account } from './pages/Account'
import type { StudioImage } from './lib/types'
import { AuthPage } from './pages/Auth'
import { useAuth } from './lib/auth'

const readTab = (): Tab =>
  location.hash === '#/inpainting' ? 'inpainting' : location.hash === '#/renderpass' ? 'renderpass' : location.hash === '#/account' ? 'account' : 'img2img'

export default function App() {
  const { status } = useAuth()
  return (
    <AnimatePresence mode="wait">
      {status === 'loading' ? (
        <motion.div key="splash" exit={{ opacity: 0 }} className="grid min-h-dvh place-items-center">
          <motion.div animate={{ scale: [1, 1.06, 1] }} transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}>
            <Logo size={44} />
          </motion.div>
        </motion.div>
      ) : status === 'guest' ? (
        <motion.div key="auth" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
          <AuthPage />
        </motion.div>
      ) : (
        <motion.div key="studio" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
          <Studio />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function Studio() {
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
        <div hidden={tab !== 'renderpass'}>
          <RenderPass />
        </div>
        {tab === 'account' && <Account onToast={setToast} />}
      </main>

      <footer className="border-t border-fg/[0.06]">
        <div className="mx-auto flex max-w-[1880px] px-5 py-8 text-[12px] text-faint sm:px-8">
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
            className="fixed inset-x-0 bottom-6 z-50 mx-auto flex w-fit items-center gap-2.5 rounded-full bg-ink/90 px-5 py-3 text-[14px] font-medium text-on-ink shadow-float backdrop-blur-xl"
          >
            <CheckCircle2 className="size-[18px] text-emerald-400" />
            {toast}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
