import { useEffect, useRef } from 'react'
import { motion } from 'motion/react'
import { ArrowUp, Sparkles, Trash2 } from 'lucide-react'
import type { InpaintMode } from '../../lib/inpaint'

import { useScene, type Scene } from '../../lib/scene'

const SUGGESTIONS: Record<Scene, string[]> = {
  exterior: ['Зелёные растения', 'Деревянная терраса', 'Люди', 'Автомобиль', 'Уличный фонарь', 'Бассейн'],
  interior: ['Комнатные растения', 'Диван', 'Картина на стене', 'Ковёр', 'Светильник', 'Декор на полке'],
}

const PLACEHOLDER: Record<Scene, string> = {
  exterior: 'Например: деревянная терраса с шезлонгами',
  interior: 'Например: мягкий диван из бежевого букле',
}

interface Props {
  mode: InpaintMode
  prompt: string
  onPrompt: (v: string) => void
  onConfirm: () => void
  onReset: () => void
  /** Позиция «носика» относительно сцены */
  x: number
  y: number
  placement: 'top' | 'bottom'
}

export const POPOVER_WIDTH = 340

export function ActionPopover({ mode, prompt, onPrompt, onConfirm, onReset, x, y, placement }: Props) {
  const input = useRef<HTMLTextAreaElement>(null)
  const { scene } = useScene()
  useEffect(() => {
    if (mode === 'edit') input.current?.focus({ preventScroll: true })
  }, [mode])

  const canConfirm = mode === 'remove' || prompt.trim().length > 0

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.94, y: placement === 'bottom' ? -8 : 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96, transition: { duration: 0.15 } }}
      transition={{ type: 'spring', stiffness: 420, damping: 32 }}
      style={{
        left: x,
        top: y,
        width: POPOVER_WIDTH,
        translateX: '-50%',
        translateY: placement === 'top' ? '-100%' : '0%',
        transformOrigin: placement === 'top' ? 'bottom center' : 'top center',
      }}
      className="absolute z-30 rounded-[22px] bg-surface/[0.88] p-4 backdrop-blur-2xl backdrop-saturate-150 shadow-float ring-1 ring-fg/[0.06]"
      onPointerDown={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && !e.shiftKey && canConfirm) { e.preventDefault(); onConfirm() }
        e.stopPropagation()
      }}
    >
      <div className="flex items-start gap-3">
        <span
          className="grid size-9 shrink-0 place-items-center rounded-full text-white"
          style={{
            background: mode === 'edit' ? 'linear-gradient(135deg,#5a9bff,#6a5cff)' : 'linear-gradient(135deg,#ff6b6b,#ff3b5c)',
          }}
        >
          {mode === 'edit' ? <Sparkles className="size-4" /> : <Trash2 className="size-4" />}
        </span>
        <div className="min-w-0 pt-0.5">
          <div className="text-[15px] font-semibold tracking-[-0.01em]">{mode === 'edit' ? 'Что сделать с областью?' : 'Удалить выделенное?'}</div>
          <div className="text-[12.5px] leading-snug text-mute">
            {mode === 'edit' ? 'Опишите, что должно появиться или измениться' : 'Объект будет заменён окружающим фоном'}
          </div>
        </div>
      </div>

      {mode === 'edit' && (
        <>
          <textarea
            ref={input}
            value={prompt}
            onChange={(e) => onPrompt(e.target.value)}
            rows={2}
            placeholder={PLACEHOLDER[scene]}
            className="mt-3.5 w-full resize-none rounded-[14px] bg-fg/[0.04] px-3.5 py-2.5 text-[14px] leading-snug outline-none ring-accent/40 transition-shadow placeholder:text-faint focus:bg-surface focus:ring-2"
          />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {SUGGESTIONS[scene].map((s) => (
              <button
                key={s}
                onClick={() => onPrompt(s)}
                className={`rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors ${
                  prompt === s ? 'bg-ink text-on-ink' : 'bg-fg/[0.05] text-ink-2 hover:bg-fg/[0.09]'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </>
      )}

      <div className="mt-4 flex items-center justify-between gap-2">
        <button onClick={onReset} className="rounded-full px-3.5 py-2 text-[13px] font-medium text-mute transition-colors hover:bg-fg/[0.05] hover:text-ink">
          Сбросить
        </button>
        <motion.button
          whileTap={canConfirm ? { scale: 0.96 } : undefined}
          onClick={onConfirm}
          disabled={!canConfirm}
          className={`flex items-center gap-1.5 rounded-full px-4 py-2 text-[13.5px] font-medium text-white transition-all duration-200 disabled:bg-fg/[0.08] disabled:text-faint ${
            mode === 'edit' ? 'bg-accent hover:bg-accent-hover' : 'bg-[#ff3b30] hover:bg-[#ff2d20]'
          }`}
        >
          {mode === 'edit' ? (
            <>Применить <ArrowUp className="size-3.5" strokeWidth={2.5} /></>
          ) : (
            <><Trash2 className="size-3.5" /> Удалить</>
          )}
        </motion.button>
      </div>
    </motion.div>
  )
}
