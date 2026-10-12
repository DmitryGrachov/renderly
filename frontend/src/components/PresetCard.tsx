import { motion } from 'motion/react'
import { Check } from 'lucide-react'
import type { Preset } from '../lib/presets'

interface Props {
  preset: Preset
  selected: boolean
  onSelect: () => void
  disabled?: boolean
  wide?: boolean
}

export function PresetCard({ preset, selected, onSelect, disabled, wide }: Props) {
  return (
    <motion.button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      whileTap={{ scale: 0.97 }}
      className={`group relative overflow-hidden rounded-[18px] bg-white text-left transition-shadow duration-300 ease-apple disabled:opacity-50 ${
        selected
          ? 'shadow-[0_0_0_2px_var(--color-ink),0_8px_24px_-8px_rgba(0,0,0,0.25)]'
          : 'shadow-[0_0_0_1px_rgba(0,0,0,0.06)] hover:shadow-[0_0_0_1px_rgba(0,0,0,0.1),0_8px_24px_-12px_rgba(0,0,0,0.2)]'
      } ${wide ? 'col-span-2 flex items-stretch' : ''}`}
    >
      <div className={`relative overflow-hidden ${wide ? 'w-[92px] shrink-0' : 'h-[76px]'}`} style={{ background: preset.sky }}>
        {preset.sun && (
          <span
            className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full transition-transform duration-700 ease-apple group-hover:scale-110"
            style={{
              left: preset.sun.x,
              top: preset.sun.y,
              width: preset.sun.size,
              height: preset.sun.size,
              background: preset.sun.color,
              boxShadow: `0 0 ${preset.sun.size}px ${preset.sun.size / 3}px ${preset.sun.color}`,
            }}
          />
        )}
        {/* силуэт здания */}
        {!wide && (
          <svg viewBox="0 0 200 40" preserveAspectRatio="none" className="absolute inset-x-0 bottom-0 h-[38%] w-full">
            <path d="M0 40 V26 H34 V14 H70 V22 H96 V6 H134 V18 H160 V24 H200 V40Z" fill="rgba(0,0,0,0.16)" />
          </svg>
        )}
        {wide && <span className="absolute inset-0 backdrop-blur-[6px]" />}
      </div>

      <div className={`p-3 ${wide ? 'flex-1 py-3.5 pl-4' : ''}`}>
        <div className="text-[13.5px] font-semibold leading-tight tracking-[-0.01em]">{preset.title}</div>
        <div className="mt-1 text-[11px] font-medium tracking-wide text-faint">{preset.meta}</div>
        {wide && <div className="mt-1.5 text-[12px] leading-snug text-mute">{preset.caption}</div>}
      </div>

      <motion.span
        initial={false}
        animate={{ scale: selected ? 1 : 0, opacity: selected ? 1 : 0 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        className="absolute right-2 top-2 grid size-5 place-items-center rounded-full bg-ink text-white shadow-md"
      >
        <Check className="size-3" strokeWidth={3} />
      </motion.span>
    </motion.button>
  )
}
