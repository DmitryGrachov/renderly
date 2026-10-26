import { AnimatePresence, motion } from 'motion/react'
import { STAGES } from '../lib/api'

interface Props {
  image: string
  progress: number
  stage: string
  presetSky: string
  onCancel: () => void
  /** этапы для шкалы; по умолчанию — этапы рендера Img to Img */
  stages?: readonly string[]
}

export function GeneratingOverlay({ image, progress, stage, presetSky, onCancel, stages = STAGES }: Props) {
  const stageIndex = Math.max(0, stages.indexOf(stage))
  const R = 26
  const C = 2 * Math.PI * R

  return (
    <div className="absolute inset-0">
      <img src={image} alt="" className="absolute inset-0 size-full scale-105 object-contain blur-[2px] saturate-50" />
      {/* «свет» пресета проявляется по мере прогресса */}
      <div
        className="absolute inset-0 mix-blend-soft-light transition-opacity duration-300"
        style={{ background: presetSky, opacity: 0.25 + progress * 0.55 }}
      />
      <div className="absolute inset-0 bg-canvas/30" />
      <div className="absolute inset-0 overflow-hidden">
        <div className="animate-shimmer absolute inset-y-0 w-1/2 bg-gradient-to-r from-transparent via-white/45 to-transparent" />
      </div>

      <div className="absolute inset-0 grid place-items-center p-6">
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 26 }}
          className="glass w-full max-w-[340px] rounded-[26px] p-6 shadow-float ring-1 ring-fg/5"
        >
          <div className="flex items-center gap-4">
            <div className="relative size-16 shrink-0">
              <svg viewBox="0 0 64 64" className="size-full -rotate-90">
                <circle cx="32" cy="32" r={R} fill="none" stroke="var(--color-fg)" strokeOpacity={0.08} strokeWidth="4" />
                <circle
                  cx="32" cy="32" r={R} fill="none" stroke="url(#prog-g)" strokeWidth="4" strokeLinecap="round"
                  strokeDasharray={C} strokeDashoffset={C * (1 - progress)}
                />
                <defs>
                  <linearGradient id="prog-g" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stopColor="#ff8a3d" />
                    <stop offset=".5" stopColor="#ff4f7b" />
                    <stop offset="1" stopColor="#6a5cff" />
                  </linearGradient>
                </defs>
              </svg>
              <span className="absolute inset-0 grid place-items-center text-[14px] font-semibold tabular-nums">
                {Math.round(progress * 100)}
              </span>
            </div>
            <div className="min-w-0">
              <div className="text-[12px] font-medium uppercase tracking-[0.08em] text-faint">Рендеринг</div>
              <AnimatePresence mode="wait">
                <motion.div
                  key={stage}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.25 }}
                  className="mt-0.5 truncate text-[16px] font-semibold tracking-[-0.01em]"
                >
                  {stage}…
                </motion.div>
              </AnimatePresence>
            </div>
          </div>

          <div className="mt-5 flex gap-1.5">
            {stages.map((s, i) => (
              <div key={s} className="h-1 flex-1 overflow-hidden rounded-full bg-fg/[0.07]">
                <div
                  className="h-full rounded-full bg-ink transition-all duration-500 ease-apple"
                  style={{ width: i < stageIndex ? '100%' : i === stageIndex ? '55%' : '0%' }}
                />
              </div>
            ))}
          </div>

          <button onClick={onCancel} className="mt-5 w-full rounded-full py-2 text-[13px] font-medium text-mute transition-colors hover:bg-fg/[0.04] hover:text-ink">
            Отменить
          </button>
        </motion.div>
      </div>
    </div>
  )
}
